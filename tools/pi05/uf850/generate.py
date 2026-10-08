"""Generate contact-only 850 bowl-placement demonstrations (ROS Python 3.10)."""
import argparse
import json
from pathlib import Path

import numpy as np
import pybullet as p
import xacro
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from ament_index_python.packages import get_package_share_directory
from tactile_simulation.engine import World

ARM = [f'joint{i}' for i in range(1, 7)]
JOINTS = ARM + ['ag95_left_outer_knuckle_joint']
PROMPT = 'pick up the black bowl between the plate and the ramekin and place it on the plate'


def joint_state(world):
    names, values = world.joint_states()
    return np.array([dict(zip(names, values))[name] for name in JOINTS], dtype=np.float32)


def bowl_contact(world, bowl):
    contacts=p.getContactPoints(bodyA=world.robot,bodyB=bowl,physicsClientId=world.client)
    return all(sum(c[9] for c in contacts if c[3] in {
        world.links[f'ag95_{side}_finger'], world.links[f'ag95_{side}_finger_pad']})>.05
        for side in ('left','right'))


def solve_pose(world, xyz):
    """Solve downward tool pose without changing the live simulation state."""
    names, saved = world.joint_states()
    velocities = [s[1] for s in p.getJointStates(world.robot, list(world.joints.values()), physicsClientId=world.client)]
    target_rotation = Rotation.from_euler('xyz', [np.pi, 0., np.pi/2]).as_matrix()
    def residual(q):
        for name, value in zip(ARM, q):
            p.resetJointState(world.robot, world.joints[name], value, physicsClientId=world.client)
        state = p.getLinkState(world.robot, world.links['ag95_ag95_base_link'],
                              computeForwardKinematics=True, physicsClientId=world.client)
        rot = np.asarray(p.getMatrixFromQuaternion(state[5])).reshape(3, 3)
        return np.r_[np.asarray(state[4])-xyz, Rotation.from_matrix(target_rotation.T@rot).as_rotvec()*.3]
    try:
        lo = [world.limits[name][0]+.0001 for name in ARM]
        hi = [world.limits[name][1]-.0001 for name in ARM]
        seed = np.clip([world.targets[name] for name in ARM], lo, hi)
        result = least_squares(residual, seed, bounds=(lo, hi), diff_step=.001, max_nfev=200)
        if np.linalg.norm(residual(result.x)) > .002:
            raise RuntimeError(f'IK cannot reach {xyz}')
        return result.x
    finally:
        for name, value, velocity in zip(names, saved, velocities):
            p.resetJointState(world.robot, world.joints[name], value, velocity, physicsClientId=world.client)


def episode(seed, render=True, varied_start=False, grasp_dwell=0):
    model = Path(get_package_share_directory('tactile_robot_description'))/'urdf/xarm6_ag95.urdf.xacro'
    world = World(xacro.process_file(str(model), mappings={'robot_model':'uf850', 'base_xyz':'0 0 .75'}).toxml(),
                  'tabletop', gripper_effort=3.)
    frames, images, wrists, actions = [], [], [], []
    rng = np.random.default_rng(seed)
    props = {obj['name']:obj for obj in world.objects.values()}
    bowl, plate = props['black_bowl']['body'], props['target_plate']['body']
    offset = rng.uniform([-.015, -.015], [.015, .015])
    xyz, quat = p.getBasePositionAndOrientation(bowl, physicsClientId=world.client)
    p.resetBasePositionAndOrientation(bowl, (xyz[0]+offset[0],xyz[1]+offset[1],xyz[2]),quat,physicsClientId=world.client)
    initial_arm=np.asarray(world.home[:6])
    if varied_start:
        start_xyz=rng.uniform([.33,-.18,1.04],[.49,.08,1.12])
        initial_arm=solve_pose(world,start_xyz)
        world.command(JOINTS,np.r_[initial_arm,rng.uniform(0.,.35)])
        for name,value in world.positions_for_targets().items():
            p.resetJointState(world.robot,world.joints[name],value,physicsClientId=world.client)
    lifted, bilateral = False, False
    max_height = 0.
    def move(target, minimum=12):
        nonlocal lifted, bilateral, max_height
        start = np.array([world.targets[name] for name in JOINTS])
        count = max(minimum, int(np.ceil(np.max(np.abs(target-start)) / .025)))
        for i in range(1,count+1):
            command = start+(target-start)*i/count
            frames.append(joint_state(world))
            actions.append(command.astype(np.float32))
            if render:
                rgb,_,_=world.render_external(224,224)
                wrist,_,_=world.render('d435_camera_color_optical_frame',224,224)
                images.append(rgb); wrists.append(wrist)
            world.command(JOINTS,command)
            for _ in range(3): world.step()
            height = p.getBasePositionAndOrientation(bowl,physicsClientId=world.client)[0][2]
            max_height = max(max_height,height)
            lifted |= height > .82
            bilateral |= bowl_contact(world,bowl)
    try:
        move(np.r_[initial_arm,0.],minimum=12)
        point = p.getBasePositionAndOrientation(bowl,physicsClientId=world.client)[0]
        for z in (1.05,.96): move(np.r_[solve_pose(world,[point[0],point[1],z]),0.])
        if grasp_dwell:
            move(np.r_[solve_pose(world,[point[0],point[1],.96]),0.],minimum=grasp_dwell)
        move(np.r_[solve_pose(world,[point[0],point[1],.96]),.45],minimum=25)
        if grasp_dwell:
            move(np.r_[solve_pose(world,[point[0],point[1],.96]),.45],minimum=grasp_dwell)
        print('grasp',seed,'bowl',p.getBasePositionAndOrientation(bowl,physicsClientId=world.client)[0],
              'jaw loads',[v['normal_load'] for v in world.jaw_forces().values()],flush=True)
        move(np.r_[solve_pose(world,[point[0],point[1],1.08]),.45],minimum=30)
        target = p.getBasePositionAndOrientation(plate,physicsClientId=world.client)[0]
        move(np.r_[solve_pose(world,[target[0],target[1],1.08]),.45],minimum=30)
        move(np.r_[solve_pose(world,[target[0],target[1],.972]),.45],minimum=30)
        move(np.r_[solve_pose(world,[target[0],target[1],.972]),0.],minimum=25)
        move(np.r_[solve_pose(world,[target[0],target[1],1.08]),0.],minimum=30)
        for _ in range(120): world.step()
        position, quat = p.getBasePositionAndOrientation(bowl,physicsClientId=world.client)
        upright = np.asarray(p.getMatrixFromQuaternion(quat)).reshape(3,3)[2,2]>.8
        success = bool(lifted and bilateral and np.linalg.norm(np.asarray(position[:2])-target[:2])<.025
                       and .765<position[2]<.79 and upright)
        result = dict(seed=seed,success=success,lifted=bool(lifted),bilateral_contact=bool(bilateral),
                      max_bowl_height=max_height,final_bowl_position=position,frames=len(frames),prompt=PROMPT,
                      fps=20,robot='uf850',state='joint1..6 and AG95 master angle (radians)',
                      gripper_motor_effort_nm=3.,teacher='IK waypoints and physical jaw contacts, no attachment constraint',
                      varied_start=varied_start,grasp_dwell_frames=grasp_dwell,
                      actions='absolute next joint1..6 and AG95 master targets (radians)')
        return result, dict(state=np.asarray(frames),actions=np.asarray(actions),
                            image=np.asarray(images),wrist_image=np.asarray(wrists))
    finally:
        world.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--episodes',type=int,default=20)
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--probe',action='store_true')
    parser.add_argument('--varied-start',action='store_true')
    parser.add_argument('--grasp-dwell',type=int,default=0)
    args=parser.parse_args()
    if args.episodes<1 or args.grasp_dwell<0:
        parser.error('episodes must be positive and grasp-dwell must be nonnegative')
    args.output.mkdir(parents=True,exist_ok=True)
    accepted=0
    for seed in range(args.seed,args.seed+args.episodes*4):
        result, data = episode(seed, render=not args.probe,varied_start=args.varied_start,grasp_dwell=args.grasp_dwell)
        print(json.dumps(result),flush=True)
        (args.output/f'attempt_{seed:04d}.json').write_text(json.dumps(result,indent=2))
        if result['success'] and not args.probe:
            dest=args.output/f'episode_{seed:04d}.npz'
            if dest.exists(): raise FileExistsError(dest)
            np.savez_compressed(dest,**data)
            accepted+=1
        if args.probe or accepted>=args.episodes: break
    if not args.probe and accepted<args.episodes:
        raise RuntimeError(f'Only {accepted}/{args.episodes} verified successful demonstrations')


if __name__=='__main__': main()
