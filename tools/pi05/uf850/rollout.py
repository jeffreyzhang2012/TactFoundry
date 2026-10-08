"""Evaluate the tuned policy on the 850 physics scene, optionally live in RViz.

All commands go directly to this process's PyBullet World. No hardware driver
or external robot command topic is used.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import shutil

import numpy as np
import pybullet as p
import rclpy
import xacro
import yaml
from PIL import Image
from ament_index_python.packages import get_package_share_directory
from openpi_client import websocket_client_policy
from tactile_simulation.node import Playground
from generate import JOINTS, PROMPT, joint_state, bowl_contact


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8001)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=101)
    parser.add_argument('--episodes',type=int,default=3)
    parser.add_argument('--max-steps',type=int,default=480)
    parser.add_argument('--execute-steps',type=int,default=4,choices=range(1,11),
                        help='Actions to execute before replanning; use 1 or 2 for more frequent feedback')
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--hold',action='store_true',help='Keep the final simulated scene live until Ctrl+C')
    args=parser.parse_args()
    ffmpeg=shutil.which('ffmpeg')
    if ffmpeg is None:
        candidates=list((Path.home()/'projects/openpi/.venv/lib').glob(
            'python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux*'))
        if not candidates: raise FileNotFoundError('Install ffmpeg or imageio-ffmpeg in OpenPI')
        ffmpeg=str(candidates[0])
    args.output.mkdir(parents=True,exist_ok=True)
    os.environ['ROS_DOMAIN_ID']='44'
    os.environ['FASTRTPS_DEFAULT_PROFILES_FILE']=str(Path(get_package_share_directory('tactile_simulation'))/'config/fastdds_udp.xml')
    client=websocket_client_policy.WebsocketClientPolicy('127.0.0.1',args.port)
    metadata=client.get_server_metadata()
    if metadata.get('robot')!='uf850_ag95' or metadata.get('gripper_motor_effort_nm')!=3.:
        raise ValueError(f'Wrong policy server for this simulated robot: {metadata}')
    share=Path(get_package_share_directory('tactile_robot_description'))
    description=xacro.process_file(str(share/'urdf/xarm6_ag95.urdf.xacro'),
                                   mappings={'base_xyz':'0 0 .75','robot_model':'uf850'}).toxml()
    children=[]
    with tempfile.TemporaryDirectory(prefix='uf850-policy-') as scratch:
        scratch=Path(scratch)
        params=scratch/'params.yaml'
        params.write_text(yaml.safe_dump({'/**':{'ros__parameters':{
            'robot_description':description,'scene':'tabletop','robot_model':'uf850',
            'object_count':0,'gamepad':False,'camera_source':'sim','gripper_effort':3.}}}))
        rclpy.init(args=['--ros-args','--params-file',str(params)])
        node=Playground()
        for timer in node.timers: timer.cancel()
        world=node.world
        log=(args.output/'rviz.log').open('w')
        if not args.headless:
            children.append(subprocess.Popen(['ros2','run','robot_state_publisher','robot_state_publisher',
                '--ros-args','--params-file',str(params)],stdout=log,stderr=subprocess.STDOUT))
            children.append(subprocess.Popen(['rviz2','-d',str(share/'rviz/tabletop.rviz')],stdout=log,stderr=subprocess.STDOUT))
        results=[]
        try:
            for seed in range(args.seed,args.seed+args.episodes):
                world.reset_arm(); world.reset_scene(); client.reset()
                props={obj['name']:obj for obj in world.objects.values()}
                bowl,plate=props['black_bowl']['body'],props['target_plate']['body']
                rng=np.random.default_rng(seed)
                offset=rng.uniform([-.015,-.015],[.015,.015])
                xyz,quat=p.getBasePositionAndOrientation(bowl,physicsClientId=world.client)
                p.resetBasePositionAndOrientation(bowl,(xyz[0]+offset[0],xyz[1]+offset[1],xyz[2]),quat,physicsClientId=world.client)
                for _ in range(30): world.step()
                node.physics(); node.camera(); node.scene()
                lifted=False; bilateral=False; success=False; clipped=0; max_height=0.
                frames_dir=scratch/f'frames_{seed}'
                frames_dir.mkdir()
                trace=[]
                for step in range(args.max_steps):
                    rgb,_,_=world.render_external(224,224)
                    wrist,_,_=world.render('d435_camera_color_optical_frame',224,224)
                    Image.fromarray(np.concatenate((rgb,wrist),axis=1)).save(frames_dir/f'{step:06d}.png')
                    if step%args.execute_steps==0:
                        prediction=np.asarray(client.infer({'observation/state':joint_state(world),
                            'observation/image':rgb,'observation/wrist_image':wrist,'prompt':PROMPT})['actions'])
                        if prediction.shape!=(10,7) or not np.isfinite(prediction).all():
                            raise ValueError('Invalid action chunk from policy')
                    target=prediction[step%args.execute_steps]
                    actual=joint_state(world)
                    trace.append(np.r_[actual,target])
                    if step%40==0:
                        tool=p.getLinkState(world.robot,world.links['ag95_ag95_base_link'],
                            computeForwardKinematics=True,physicsClientId=world.client)[4]
                        print('trace',seed,step,'tool',tool,'grip actual/target',float(actual[6]),float(target[6]),flush=True)
                    # Match the teacher's command interpolation. Limiting an
                    # absolute setpoint relative to lagging measured joints
                    # changes the executed trajectory and grip/arm timing.
                    previous=np.array([world.targets[name] for name in JOINTS])
                    rate_limited=previous+np.clip(target-previous,-.025,.025)
                    clipped+=int(not np.allclose(target,rate_limited))
                    world.command(JOINTS,rate_limited)
                    for _ in range(2): world.step()
                    node.physics(); node.camera(); node.scene()
                    position,quat=p.getBasePositionAndOrientation(bowl,physicsClientId=world.client)
                    target_pos=p.getBasePositionAndOrientation(plate,physicsClientId=world.client)[0]
                    max_height=max(max_height,position[2]); lifted|=position[2]>.82
                    bilateral|=bowl_contact(world,bowl)
                    upright=np.asarray(p.getMatrixFromQuaternion(quat)).reshape(3,3)[2,2]>.8
                    near_plate=np.linalg.norm(np.asarray(position[:2])-target_pos[:2])<.025
                    released=world.targets[JOINTS[-1]]<.1
                    success=bool(lifted and bilateral and near_plate and .765<position[2]<.79 and upright and released)
                    if success: break
                    if not args.headless: time.sleep(.05)
                for _ in range(120): world.step()
                final=p.getBasePositionAndOrientation(bowl,physicsClientId=world.client)[0]
                # Require the placement to survive settling after the last action.
                success=bool(success and np.linalg.norm(np.asarray(final[:2])-target_pos[:2])<.025 and .765<final[2]<.79)
                subprocess.run([ffmpeg,'-loglevel','error','-framerate','20','-i',str(frames_dir/'%06d.png'),
                    '-c:v','libx264','-pix_fmt','yuv420p',str(args.output/f'rollout_{seed}.mp4')],check=True)
                result=dict(seed=seed,success=success,lifted=bool(lifted),bilateral_contact=bool(bilateral),
                            steps=step+1,rate_limited_steps=clipped,max_bowl_height=max_height,
                            final_bowl_position=final,policy_metadata=metadata)
                result['rate_limit_reference']='previous_command'
                result['execute_steps']=args.execute_steps
                results.append(result); print(json.dumps(result),flush=True)
                np.savez_compressed(args.output/f'trace_{seed}.npz',joints_and_targets=np.asarray(trace))
                (args.output/'results.json').write_text(json.dumps(results,indent=2))
            while args.hold:
                for _ in range(2): world.step()
                node.physics(); node.camera(); node.scene()
                time.sleep(.05)
        finally:
            for child in children:
                child.terminate()
                try: child.wait(timeout=5)
                except subprocess.TimeoutExpired: child.kill()
            log.close(); node.destroy_node(); rclpy.shutdown()

if __name__=='__main__': main()
