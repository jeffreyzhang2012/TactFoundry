from pathlib import Path

import numpy as np
import pybullet as p
import pytest
import xacro
from ament_index_python.packages import get_package_share_directory

from tactile_simulation.engine import World
from tactile_simulation.objects import KINDS


@pytest.fixture(params=['uf850', 'xarm6'])
def world(request):
    model = Path(get_package_share_directory('tactile_robot_description')) / 'urdf/xarm6_ag95.urdf.xacro'
    instance = World(xacro.process_file(str(model), mappings={'robot_model': request.param}).toxml())
    yield instance
    instance.close()


def test_catalogue_physics_and_clear(world):
    world.spawn('mixed', len(KINDS), 42)
    assert {obj['kind'] for obj in world.objects.values()} == set(KINDS)
    assert len({obj['slot'] for obj in world.objects.values()}) == len(KINDS)
    for _ in range(180):
        world.step()
    for obj in world.objects.values():
        low, high = p.getAABB(obj['body'], physicsClientId=world.client)
        assert 0.055 < low[2] < 0.07  # Resting on tabletop, not falling through it.
        assert high[2] > low[2]
    world.clear()
    assert not world.objects


def test_camera_and_joint_motion(world):
    world.spawn('mixed', 8, 42)
    before, depth, focal = world.render('d435_camera_color_optical_frame')
    assert before.shape == (240, 320, 3)
    assert depth.shape == (240, 320) and depth.dtype == np.float32
    assert focal > 0 and np.isfinite(depth).all()
    assert np.any(depth > 0)
    world.command(['joint1'], [0.5])
    for _ in range(120):
        world.step()
    names, values = world.joint_states()
    assert abs(dict(zip(names, values))['joint1'] - 0.5) < 0.03
    after, _, _ = world.render('d435_camera_color_optical_frame')
    assert not np.array_equal(before, after)


def test_invalid_requests_do_not_mutate_scene(world):
    for kind, count in [('unknown', 1), ('cube', 0), ('cube', 49)]:
        with pytest.raises(ValueError):
            world.spawn(kind, count, 42)
    assert not world.objects
    world.command(['joint1', 'joint2', 'bogus'], [100., float('nan'), 0.])
    assert world.targets['joint1'] == world.limits['joint1'][1]
    assert world.targets['joint2'] == -0.6


def test_cartesian_ik_moves_tool_and_respects_joint_limits(world):
    def pose():
        return np.array(p.getLinkState(world.robot, world.links['ag95_ag95_base_link'],
                       computeForwardKinematics=True, physicsClientId=world.client)[4])
    before = pose()
    for _ in range(120):
        world.cartesian_command([0., .025, 0., 0., 0., 0.], 0., 1/60)
        world.step()
    change = pose() - before
    assert change[1] > .015
    assert abs(change[0]) < .015 and abs(change[2]) < .015
    for _ in range(60):
        world.cartesian_command([0., 0., 0., .2, .2, .2], .5, 1/60)
        world.step()
    for name, value in world.targets.items():
        lower, upper, _ = world.limits[name]
        assert lower <= value <= upper


def test_camera_relative_motion_follows_wrist_orientation(world):
    frame = 'd435_camera_color_optical_frame'
    local = [0., 0., .02, .1, 0., 0.]
    def orientation():
        state = p.getLinkState(world.robot, world.links[frame], computeForwardKinematics=True,
                               physicsClientId=world.client)
        return np.asarray(p.getMatrixFromQuaternion(state[5])).reshape(3, 3)
    rotation = orientation()
    mapped = world.twist_to_world(local, frame)
    assert np.allclose(mapped[:3], rotation[:, 2] * .02)
    assert np.allclose(mapped[3:], rotation[:, 0] * .1)
    p.resetJointState(world.robot, world.joints['joint1'], .8, physicsClientId=world.client)
    assert not np.allclose(mapped, world.twist_to_world(local, frame))
    state = p.getLinkState(world.robot, world.links['ag95_ag95_base_link'],
                           computeForwardKinematics=True, physicsClientId=world.client)
    start = np.asarray(p.multiplyTransforms(state[4], state[5], [0., 0., .15], [0., 0., 0., 1.])[0])
    forward = orientation()[:, 2]
    for _ in range(120):
        world.cartesian_command([0., 0., .02, 0., 0., 0.], 0., 1/60, frame=frame)
        world.step()
    state = p.getLinkState(world.robot, world.links['ag95_ag95_base_link'],
                           computeForwardKinematics=True, physicsClientId=world.client)
    end = np.asarray(p.multiplyTransforms(state[4], state[5], [0., 0., .15], [0., 0., 0., 1.])[0])
    assert np.dot(end - start, forward) > .01
