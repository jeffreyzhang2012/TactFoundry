from pathlib import Path
import numpy as np
import pybullet as p
import xacro
from ament_index_python.packages import get_package_share_directory
from tactile_simulation.engine import World
from generate import JOINTS, episode
from success import PlacementEvaluator, placement_checks


def test_placement_requires_target_contact_release_and_stability():
    good=dict(grasped_and_lifted=True,distance=.01,upright=1.,plate_contact=True,
              released=True,robot_contact=False,linear_speed=0.,angular_speed=0.)
    assert all(placement_checks(**good).values())
    for key,value in [('grasped_and_lifted',False),('distance',.03),('upright',.5),
                      ('plate_contact',False),('released',False),('robot_contact',True),
                      ('linear_speed',.02),('angular_speed',.2)]:
        assert not all(placement_checks(**dict(good,**{key:value})).values()),key


def test_physical_bowl_placement_does_not_count_as_bottle_success():
    _,data=episode(1,render=False)
    model=Path(get_package_share_directory('tactile_robot_description'))/'urdf/xarm6_ag95.urdf.xacro'
    world=World(xacro.process_file(str(model),mappings={'robot_model':'uf850','base_xyz':'0 0 .75'}).toxml(),
                'tabletop',gripper_effort=3.)
    try:
        props={obj['name']:obj['body'] for obj in world.objects.values()}
        xyz,quat=p.getBasePositionAndOrientation(props['black_bowl'],physicsClientId=world.client)
        offset=np.random.default_rng(1).uniform([-.015,-.015],[.015,.015])
        p.resetBasePositionAndOrientation(props['black_bowl'],(xyz[0]+offset[0],xyz[1]+offset[1],xyz[2]),quat,
                                         physicsClientId=world.client)
        bowl=PlacementEvaluator(world,props['black_bowl'],props['target_plate'])
        bottle=PlacementEvaluator(world,props['bottle'],props['target_plate'])
        for command in data['actions']:
            world.command(JOINTS,command)
            for _ in range(3): world.step()
            bowl.observe();bottle.observe()
        stable=0
        for _ in range(60):  # allow this physical teacher replay three seconds to settle
            for _ in range(3): world.step()
            bowl_checks,_,_=bowl.observe()
            bottle_checks,_,_=bottle.observe()
            stable=stable+1 if all(bowl_checks.values()) else 0
            assert not all(bottle_checks.values())
        assert stable>=20, bowl_checks
    finally:
        world.close()
