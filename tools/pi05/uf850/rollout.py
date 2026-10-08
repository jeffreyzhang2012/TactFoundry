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
from PIL import Image, ImageDraw
from ament_index_python.packages import get_package_share_directory
from openpi_client import websocket_client_policy
from tactile_simulation.node import Playground
from generate import JOINTS, PROMPT, joint_state
from success import PlacementEvaluator
from tasks import PROMPTS, apply_layout, layout_spec

TASKS = {'black_bowl': PROMPT,
         'bottle': 'pick up the green bottle and place it upright on the plate',
         'distractor_block': 'pick up the red block and place it on the plate'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8001)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=101)
    parser.add_argument('--object',choices=TASKS,default='black_bowl')
    parser.add_argument('--layout',choices=['original','train','heldout'],default='original')
    parser.add_argument('--prompt',help='Override the instruction for a language sensitivity test')
    parser.add_argument('--ros-domain-id',type=int,default=44)
    parser.add_argument('--episodes',type=int,default=3)
    parser.add_argument('--max-steps',type=int,default=480)
    parser.add_argument('--execute-steps',type=int,default=4,choices=range(1,11),
                        help='Actions to execute before replanning; use 1 or 2 for more frequent feedback')
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--publish-ros-images',action='store_true',
                        help='Publish full ROS camera streams during headless tests (policy RGB and videos are always rendered)')
    parser.add_argument('--hold',action='store_true',help='Keep the final simulated scene live until Ctrl+C')
    args=parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1:
        parser.error('episodes and max-steps must be positive')
    ffmpeg=shutil.which('ffmpeg')
    if ffmpeg is None:
        candidates=list((Path.home()/'projects/openpi/.venv/lib').glob(
            'python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux*'))
        if not candidates: raise FileNotFoundError('Install ffmpeg or imageio-ffmpeg in OpenPI')
        ffmpeg=str(candidates[0])
    args.output.mkdir(parents=True,exist_ok=True)
    os.environ['ROS_DOMAIN_ID']=str(args.ros_domain_id)
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
        def publish_camera():
            if not args.headless or args.publish_ros_images:node.camera()
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
                bowl,plate=props[args.object]['body'],props['target_plate']['body']
                rng=np.random.default_rng(seed)
                offset=rng.uniform([-.015,-.015],[.015,.015])
                spec=layout_spec(seed,args.layout) if args.layout!='original' else None
                if spec:
                    apply_layout(world,spec)
                else:
                    xyz,quat=p.getBasePositionAndOrientation(bowl,physicsClientId=world.client)
                    p.resetBasePositionAndOrientation(bowl,(xyz[0]+offset[0],xyz[1]+offset[1],xyz[2]),quat,physicsClientId=world.client)
                    for _ in range(30): world.step()
                prompt=args.prompt or (PROMPTS[args.object][seed%3] if spec else TASKS[args.object])
                evaluator=PlacementEvaluator(world,bowl,plate)
                other_evaluators={name:PlacementEvaluator(world,props[name]['body'],plate)
                                  for name in PROMPTS if name!=args.object}
                node.physics(); publish_camera(); node.scene()
                initial_objects={name:p.getBasePositionAndOrientation(obj['body'],physicsClientId=world.client)[0]
                                 for name,obj in props.items()}
                success=False; clipped=0
                frames_dir=scratch/f'frames_{seed}'
                frames_dir.mkdir()
                trace=[]
                object_trace=[]
                for step in range(args.max_steps):
                    rgb,_,_=world.render_external(224,224)
                    wrist,_,_=world.render('d435_camera_color_optical_frame',224,224)
                    frame=Image.fromarray(np.concatenate((rgb,wrist),axis=1))
                    ImageDraw.Draw(frame).text((4,4),f'{args.object} | step {step}',fill='white',stroke_width=1,stroke_fill='black')
                    frame.save(frames_dir/f'{step:06d}.png')
                    if step%args.execute_steps==0:
                        prediction=np.asarray(client.infer({'observation/state':joint_state(world),
                            'observation/image':rgb,'observation/wrist_image':wrist,'prompt':prompt})['actions'])
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
                    node.physics(); publish_camera(); node.scene()
                    checks,position,distance=evaluator.observe()
                    for other in other_evaluators.values(): other.observe()
                    object_trace.append([*position,distance,int(evaluator.bilateral),int(evaluator.grasped_and_lifted)])
                    success=all(checks.values())
                    if success: break
                    if not args.headless: time.sleep(.05)
                stable=0
                for settle in range(40):
                    for _ in range(3): world.step()
                    checks,position,distance=evaluator.observe()
                    stable=stable+1 if all(checks.values()) else 0
                    rgb,_,_=world.render_external(224,224)
                    wrist,_,_=world.render('d435_camera_color_optical_frame',224,224)
                    frame=Image.fromarray(np.concatenate((rgb,wrist),axis=1))
                    ImageDraw.Draw(frame).text((4,4),f'{args.object} | settling',fill='white',stroke_width=1,stroke_fill='black')
                    frame.save(frames_dir/f'{step+1+settle:06d}.png')
                final=p.getBasePositionAndOrientation(bowl,physicsClientId=world.client)[0]
                # Require the placement to survive settling after the last action.
                success=bool(stable>=20)  # at least one continuous second after release
                subprocess.run([ffmpeg,'-loglevel','error','-framerate','20','-i',str(frames_dir/'%06d.png'),
                    '-c:v','libx264','-pix_fmt','yuv420p',str(args.output/f'rollout_{seed}.mp4')],check=True)
                result=dict(seed=seed,object=args.object,prompt=prompt,layout=spec or 'original',success=success,
                            lifted=bool(evaluator.lifted),bilateral_contact=bool(evaluator.bilateral),
                            steps=step+1,rate_limited_steps=clipped,max_object_height=evaluator.max_height,
                            final_object_position=final,policy_metadata=metadata)
                result['rate_limit_reference']='previous_command'
                result['execute_steps']=args.execute_steps
                result['ros_camera_streams']=not args.headless or args.publish_ros_images
                result['final_checks']=checks
                result['distance_to_plate_m']=distance
                result['stable_placement_seconds']=stable*.05
                result['failure_reasons']=[key for key,value in checks.items() if not value]
                result['initial_object_positions']=initial_objects
                result['final_object_positions']={name:p.getBasePositionAndOrientation(obj['body'],physicsClientId=world.client)[0]
                                                 for name,obj in props.items()}
                result['object_activity']={name:{'bilateral_contact':bool(ev.bilateral),
                    'grasped_and_lifted':bool(ev.grasped_and_lifted),'max_height_m':ev.max_height}
                    for name,ev in {args.object:evaluator,**other_evaluators}.items()}
                result['orientation_goal']='upright' if evaluator.require_upright else 'any (cube symmetry)'
                result['success_rule']='target-specific bilateral grasp during >=4cm lift; plate contact within 25mm and required orientation; released, robot clear, stationary for >=1s'
                results.append(result); print(json.dumps(result),flush=True)
                np.savez_compressed(args.output/f'trace_{seed}.npz',joints_and_targets=np.asarray(trace),
                                    object_xyz_distance_bilateral_graspedlifted=np.asarray(object_trace))
                (args.output/'results.json').write_text(json.dumps(results,indent=2))
            while args.hold:
                for _ in range(2): world.step()
                node.physics(); publish_camera(); node.scene()
                time.sleep(.05)
        finally:
            for child in children:
                child.terminate()
                try: child.wait(timeout=5)
                except subprocess.TimeoutExpired: child.kill()
            log.close(); node.destroy_node(); rclpy.shutdown()

if __name__=='__main__': main()
