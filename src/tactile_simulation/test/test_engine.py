from pathlib import Path

import numpy as np
import pybullet as p
import pytest
import xacro
from ament_index_python.packages import get_package_share_directory

from tactile_simulation.engine import World
from tactile_simulation.objects import KINDS


@pytest.fixture
def world():
    model = Path(get_package_share_directory('tactile_robot_description')) / 'urdf/xarm6_ag95.urdf.xacro'
    instance = World(xacro.process_file(str(model)).toxml())
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
