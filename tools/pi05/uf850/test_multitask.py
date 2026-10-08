import numpy as np
from pathlib import Path
import xacro
from ament_index_python.packages import get_package_share_directory
from tactile_simulation.engine import World
from generate import solve_pose
from multitask import episode
from tasks import OBJECTS, ORDERS, layout_spec


def test_layout_orders_are_excluded_and_each_identity_visits_each_training_slot():
    assert not set(ORDERS['train']) & set(ORDERS['heldout'])
    for identity in range(3):
        assert {order[identity] for order in ORDERS['train']}=={0,1,2}
    assert layout_spec(3001)==layout_spec(3001)


def test_matched_scene_supports_three_physical_tasks():
    demonstrations=[episode(3001,name,render=False) for name in OBJECTS]
    for result,data in demonstrations:
        assert result['success'] and result['bilateral_contact'] and result['lifted']
        assert result['stable_placement_seconds']>=1.
        assert data['state'].shape==data['actions'].shape==(result['frames'],7)
        assert result['object'] in result['prompt'] or {
            'black_bowl':'black bowl','bottle':'green bottle','distractor_block':'red block'}[result['object']] in result['prompt']
        assert result['slot_order']==[0,2,1]
    for _,data in demonstrations[1:]:
        np.testing.assert_allclose(data['state'][:13],demonstrations[0][1]['state'][:13],atol=1e-6)
    # At least two tasks need different first-reach actions from the same observation.
    assert not np.allclose(demonstrations[0][1]['actions'][12:22],demonstrations[1][1]['actions'][12:22])


def test_ik_near_zero_joints_preserves_live_state():
    model=Path(get_package_share_directory('tactile_robot_description'))/'urdf/xarm6_ag95.urdf.xacro'
    world=World(xacro.process_file(str(model),mappings={'robot_model':'uf850','base_xyz':'0 0 .75'}).toxml(),
                'tabletop',gripper_effort=3.)
    try:
        names,before=world.joint_states()
        result=solve_pose(world,[.38916529,.01158478,1.10])
        assert result.shape==(6,) and np.isfinite(result).all()
        after_names,after=world.joint_states()
        assert names==after_names
        np.testing.assert_allclose(before,after,atol=1e-10)
    finally:
        world.close()
