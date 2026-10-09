"""Controlled, evaluation-only color transfer probe of the unchanged V6 policy."""
import argparse,json
from pathlib import Path
import numpy as np
import pybullet as p
import xacro
from PIL import Image,ImageDraw
from ament_index_python.packages import get_package_share_directory
from openpi_client import websocket_client_policy
from tactile_simulation.engine import World
from diverse_tasks import color_transfer_spec,apply_layout,label,TEMPLATES,render_external
from generate import JOINTS,joint_state,solve_pose


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--port',type=int,default=8001)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    client=websocket_client_policy.WebsocketClientPolicy('127.0.0.1',args.port)
    metadata=client.get_server_metadata()
    if metadata.get('dataset')!='tactfoundry/uf850_diverse_sim_v6' or not metadata.get('supports_sampling_noise'):
        raise ValueError('Requires the unchanged V6 policy server with controlled sampling')
    model=Path(get_package_share_directory('tactile_robot_description'))/'urdf/xarm6_ag95.urdf.xacro'
    description=xacro.process_file(str(model),mappings={'robot_model':'uf850','base_xyz':'0 0 .75'}).toxml()
    noise=np.random.default_rng(7).standard_normal((10,32)).astype(np.float32)
    report={'policy_metadata':metadata,'retrained':False,'novel_color':'cyan',
        'known_color':'red','distractors':['blue','yellow'],'variants':{},
        'note':'Same shape for all three objects. Within a scene only text changes, with fixed RGB/state/noise. '
               'Between paired variants only target color and its word change. Branch agreement is not physical success.'}
    previews=[]
    for variant in ('known','cyan'):
        rows=[]
        for seed in range(42001,42013):
            spec=color_transfer_spec(seed,variant)
            world=World(description,'tabletop',gripper_effort=3.)
            try:
                apply_layout(world,spec)
                world.command(JOINTS,np.r_[world.home[:6],0.])
                for _ in range(36):world.step()  # The teacher's 12 initial idle frames.
                state=joint_state(world)
                rgb=render_external(world,metadata['dataset_profile']['external_roi'])
                wrist=world.render('d435_camera_color_optical_frame',224,224)[0]
                initial=np.array([world.targets[name] for name in JOINTS])
                branches=[]
                for item in spec['objects']:
                    body=next(obj['body'] for obj in world.objects.values() if obj['name']==item['name'])
                    xyz=p.getBasePositionAndOrientation(body,physicsClientId=world.client)[0]
                    approach=max(1.10,xyz[2]+.190+.10)
                    command=np.r_[solve_pose(world,[*xyz[:2],approach]),0.]
                    count=max(30,int(np.ceil(np.max(np.abs(command-initial))/.025)))
                    branches.append(np.array([initial+(i/count)*(command-initial) for i in range(1,11)])[:,:6])
                predictions=[];prompts=[]
                for item in spec['objects']:
                    prompt=TEMPLATES[0].format(label=label(item));prompts.append(prompt)
                    predictions.append(np.asarray(client.infer({'observation/state':state,
                        'observation/image':rgb,'observation/wrist_image':wrist,'prompt':prompt,
                        '_sampling_noise':noise})['actions'])[:,:6])
                errors=np.mean(np.abs(np.asarray(predictions)[:,None]-np.asarray(branches)[None]),axis=(2,3))
                selected=np.argmin(errors,axis=1)
                rows.append({'seed':seed,'target_label':label(spec['objects'][0]),'slot':spec['objects'][0]['slot'],
                    'kind':spec['objects'][0]['kind'],'prompts':prompts,'nearest_teacher_branch':selected.tolist(),
                    'arm_mae_matrix_rad':errors.tolist(),'target_branch_correct':bool(selected[0]==0),
                    'matching_branches':int(np.sum(selected==np.arange(3))),'layout':spec})
                print(variant,seed,label(spec['objects'][0]),selected.tolist(),flush=True)
                if seed in (42001,42004,42007,42010):
                    frame=Image.fromarray(rgb);ImageDraw.Draw(frame).text((2,2),label(spec['objects'][0]),fill='white')
                    previews.append(frame)
            finally:world.close()
        report['variants'][variant]={'layouts':12,'instructions':36,
            'target_branch_agreement':sum(r['target_branch_correct'] for r in rows),
            'all_instruction_branch_agreement':sum(r['matching_branches'] for r in rows),'probes':rows}
    (args.output/'instruction-color-probe.json').write_text(json.dumps(report,indent=2))
    canvas=Image.new('RGB',(224*4,224*2))
    for i,frame in enumerate(previews):canvas.paste(frame,((i%4)*224,(i//4)*224))
    canvas.save(args.output/'color-test-scenes.png')
    print({v:{k:r[k] for k in ('target_branch_agreement','all_instruction_branch_agreement')} for v,r in report['variants'].items()},flush=True)


if __name__=='__main__':main()
