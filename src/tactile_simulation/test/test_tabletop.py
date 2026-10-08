from pathlib import Path
import numpy as np
import pybullet as p
import pytest
import xacro
from ament_index_python.packages import get_package_share_directory

from tactile_simulation.engine import World
from tactile_simulation.tabletop import EXTERNAL_EYE, EXTERNAL_TARGET, optical_pose


@pytest.fixture(params=['uf850', 'xarm6'])
def bench(request):
    model = Path(get_package_share_directory('tactile_robot_description')) / 'urdf/xarm6_ag95.urdf.xacro'
    description = xacro.process_file(str(model), mappings={
        'robot_model': request.param, 'base_xyz': '0 0 .75'}).toxml()
    world = World(description, 'tabletop', request.param)
    yield world
    world.close()


def test_robot_home_table_clearance_and_objects_settle(bench):
    state = p.getLinkState(bench.robot, bench.links['ag95_ag95_base_link'],
                          computeForwardKinematics=True, physicsClientId=bench.client)
    assert np.allclose(state[4], [.39, 0., 1.08], atol=.002)
    rotation = np.asarray(p.getMatrixFromQuaternion(state[5])).reshape(3, 3)
    assert np.allclose(rotation[:, 2], [0., 0., -1.], atol=.002)
    for name, angle in bench.targets.items():
        assert bench.limits[name][0] <= angle <= bench.limits[name][1]
    assert {o['name'] for o in bench.objects.values()} == {
        'target_plate', 'black_bowl', 'ramekin', 'bottle', 'distractor_block'}
    bowl = next(o for o in bench.objects.values() if o['name'] == 'black_bowl')
    low, high = p.getAABB(bowl['body'], physicsClientId=bench.client)
    assert .06 < high[0]-low[0] < .095  # Fits the AG95, including rim thickness.
    for _ in range(180):
        bench.step()
    for obj in bench.objects.values():
        low, _ = p.getAABB(obj['body'], physicsClientId=bench.client)
        assert .745 < low[2] < .76
    # Any base/table contact is allowed; moving links must stay clear at home.
    contacts = p.getContactPoints(bodyA=bench.robot, physicsClientId=bench.client)
    assert not [c for c in contacts if c[3] > bench.links['link_base'] and c[8] < -.0005]
    assert all(r['normal_load'] == 0. for r in bench.jaw_forces().values())
    assert all(r['point'][2] > .75 for r in bench.jaw_forces().values())


def test_external_camera_projection_and_reset(bench):
    xyz, rpy = optical_pose()
    rotation = np.asarray(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(rpy))).reshape(3, 3)
    forward = np.asarray(EXTERNAL_TARGET) - EXTERNAL_EYE
    assert np.allclose(rotation[:, 2], forward / np.linalg.norm(forward))
    # All task props project inside the fixed external camera's image.
    rgb, depth, focal = bench.render_external()
    assert rgb.shape == (360, 480, 3) and depth.shape == (360, 480)
    assert np.isfinite(depth).all() and np.any(depth > 0)
    for obj in bench.objects.values():
        point = p.getBasePositionAndOrientation(obj['body'], physicsClientId=bench.client)[0]
        x, y, z = rotation.T @ (np.asarray(point) - xyz)
        assert z > 0 and 0 < 240+focal*x/z < 480 and 0 < 180+focal*y/z < 360
    bench.spawn('cube', 2, 1)
    assert len(bench.objects) == 7
    bench.reset_scene()
    assert len(bench.objects) == 5
    bench.clear()
    assert not bench.objects
    assert len(bench.fixtures) == 8  # Floor, bench, legs, cabinet and backdrop survive.
