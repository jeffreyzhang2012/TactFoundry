"""Generate balanced, physical pick-and-place demonstrations with swapped objects."""
import argparse
import json
from pathlib import Path
import numpy as np
import pybullet as p
import xacro
from ament_index_python.packages import get_package_share_directory
from tactile_simulation.engine import World
from generate import JOINTS, joint_state, solve_pose
from success import PlacementEvaluator
from tasks import OBJECTS, PROMPTS, apply_layout, layout_spec


def episode(layout_seed, target, *, split='train', render=True, close_angle=None, grasp_offset=.195, diverse=False, spec_override=None):
    model=Path(get_package_share_directory('tactile_robot_description'))/'urdf/xarm6_ag95.urdf.xacro'
    world=World(xacro.process_file(str(model),mappings={'robot_model':'uf850','base_xyz':'0 0 .75'}).toxml(),
                'tabletop',gripper_effort=3.)
    states,actions,images,wrists=[],[],[],[]
    if diverse:
        from diverse_tasks import layout_spec as diverse_spec,apply_layout as diverse_apply,label,TEMPLATES,render_external
        spec=spec_override if spec_override is not None else diverse_spec(layout_seed,split)
        item=next(item for item in spec['objects'] if item['name']==target)
        prompt=TEMPLATES[spec['prompt_variant']].format(label=label(item))
        initialize=diverse_apply
    else:
        spec=layout_spec(layout_seed,split)
        prompt=PROMPTS[target][layout_seed % len(PROMPTS[target])]
        item=None;initialize=apply_layout
    try:
        initialize(world,spec)
        props={obj['name']:obj['body'] for obj in world.objects.values()}
        body,plate=props[target],props['target_plate']
        evaluator=PlacementEvaluator(world,body,plate)
        is_bowl=item['kind']=='bowl' if item else target=='black_bowl'
        close_angle=close_angle if close_angle is not None else ((.55 if diverse else .45) if is_bowl else .8)
        stable=0
        def move(command, minimum=12):
            nonlocal stable
            initial=np.array([world.targets[name] for name in JOINTS])
            count=max(minimum,int(np.ceil(np.max(np.abs(command-initial))/.025)))
            for i in range(1,count+1):
                next_target=initial+(command-initial)*i/count
                states.append(joint_state(world));actions.append(next_target.astype(np.float32))
                if render:
                    images.append(render_external(world) if diverse else world.render_external(224,224)[0])
                    wrists.append(world.render('d435_camera_color_optical_frame',224,224)[0])
                world.command(JOINTS,next_target)
                for _ in range(3): world.step()
                checks,_,_=evaluator.observe()
                stable=stable+1 if all(checks.values()) else 0
        move(np.r_[world.home[:6],0.],12)
        xyz=p.getBasePositionAndOrientation(body,physicsClientId=world.client)[0]
        grasp_z=xyz[2]+grasp_offset
        approach=max(1.10,grasp_z+.10)
        move(np.r_[solve_pose(world,[*xyz[:2],approach]),0.],30)
        move(np.r_[solve_pose(world,[*xyz[:2],grasp_z]),0.],30)
        move(np.r_[solve_pose(world,[*xyz[:2],grasp_z]),0.],12)
        move(np.r_[solve_pose(world,[*xyz[:2],grasp_z]),close_angle],40)
        move(np.r_[solve_pose(world,[*xyz[:2],grasp_z]),close_angle],12)
        move(np.r_[solve_pose(world,[*xyz[:2],approach]),close_angle],30)
        plate_xyz=p.getBasePositionAndOrientation(plate,physicsClientId=world.client)[0]
        move(np.r_[solve_pose(world,[*plate_xyz[:2],approach]),close_angle],30)
        deposit_z=grasp_z+.012
        move(np.r_[solve_pose(world,[*plate_xyz[:2],deposit_z]),close_angle],30)
        move(np.r_[solve_pose(world,[*plate_xyz[:2],deposit_z]),0.],40)
        move(np.r_[solve_pose(world,[*plate_xyz[:2],approach]),0.],30)
        move(np.r_[solve_pose(world,[*plate_xyz[:2],approach]),0.],60 if diverse else 40)
        checks,final,distance=evaluator.observe()
        result=dict(success=stable>=20,object=target,prompt=prompt,**spec,
                    bilateral_contact=bool(evaluator.bilateral),lifted=bool(evaluator.lifted),
                    max_object_height=evaluator.max_height,final_object_position=final,
                    distance_to_plate_m=distance,stable_placement_seconds=stable*.05,
                    final_checks=checks,frames=len(states),fps=20,gripper_motor_effort_nm=3.,
                    close_angle=close_angle,grasp_offset=grasp_offset,
                    teacher='IK waypoints with physical contacts; no attachment or manipulation-time object reset')
        if item:result.update(color=item['color'],kind=item['kind'],target_label=label(item),
                              slot=item['slot'],novel_combination=item['novel_combination'])
        return result,dict(state=np.asarray(states),actions=np.asarray(actions),
                          image=np.asarray(images),wrist_image=np.asarray(wrists))
    finally:
        world.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--train-layouts',type=int,default=24)
    parser.add_argument('--heldout-layouts',type=int,default=6)
    parser.add_argument('--train-seed',type=int,default=3001)
    parser.add_argument('--heldout-seed',type=int,default=4001)
    parser.add_argument('--probe',action='store_true')
    parser.add_argument('--diverse',action='store_true')
    parser.add_argument('--object',choices=(*OBJECTS,'target_0','target_1','target_2'),default='bottle')
    parser.add_argument('--close-angle',type=float)
    parser.add_argument('--grasp-offset',type=float,default=.195)
    args=parser.parse_args()
    if min(args.train_layouts,args.heldout_layouts)<0 or args.train_layouts+args.heldout_layouts<1:
        parser.error('layout counts must be nonnegative with at least one layout requested')
    args.output.mkdir(parents=True,exist_ok=True)
    if args.diverse:
        from diverse_tasks import TARGETS,PROFILE
        targets=TARGETS
        if args.object=='bottle':args.object='target_0'
        (args.output/'dataset-profile.json').write_text(json.dumps(PROFILE,indent=2))
        if args.grasp_offset==.195:args.grasp_offset=.190
    else:targets=OBJECTS
    if args.probe:
        result,_=episode(args.train_seed,args.object,render=False,close_angle=args.close_angle,grasp_offset=args.grasp_offset,diverse=args.diverse)
        print(json.dumps(result),flush=True);return
    manifest={'train':[],'heldout':[],'matched_targets':list(targets),'discarded_layouts':[]}
    for split,total,start in [('train',args.train_layouts,args.train_seed),('heldout',args.heldout_layouts,args.heldout_seed)]:
        if total==0:continue
        accepted=0
        for seed in range(start,start+total*(12 if args.diverse else 6)):
            results=[]
            for target in targets:
                try:
                    result,data=episode(seed,target,split=split,render=False,diverse=args.diverse,
                                        close_angle=args.close_angle,grasp_offset=args.grasp_offset)
                except RuntimeError as error:
                    if not str(error).startswith('IK cannot reach'):raise
                    result=dict(success=False,object=target,layout_seed=seed,layout_split=split,reason=str(error))
                    data=None
                print(json.dumps(result),flush=True)
                results.append((result,data))
            if not all(result['success'] for result,_ in results):
                manifest['discarded_layouts'].append({'layout_seed':seed,'split':split,
                    'results':[result for result,_ in results]})
                (args.output/'layout_manifest.json').write_text(json.dumps(manifest,indent=2));continue
            results=[episode(seed,target,split=split,render=True,diverse=args.diverse,
                             close_angle=args.close_angle,grasp_offset=args.grasp_offset) for target in targets]
            if not all(result['success'] for result,_ in results):
                if args.diverse:
                    manifest['discarded_layouts'].append({'layout_seed':seed,'split':split,
                        'reason':'recorded replay did not remain successful','results':[result for result,_ in results]})
                    (args.output/'layout_manifest.json').write_text(json.dumps(manifest,indent=2));continue
                raise RuntimeError('Physical preflight changed outcome during image recording')
            for index,(result,data) in enumerate(results):
                episode_seed=60000+seed*3+index
                path=args.output/f'episode_{episode_seed:04d}.npz'
                if path.exists(): raise FileExistsError(path)
                result['seed']=episode_seed
                np.savez_compressed(path,**data)
                (args.output/f'attempt_{episode_seed:04d}.json').write_text(json.dumps(result,indent=2))
                manifest[split].append({'seed':episode_seed,'layout_seed':seed,'object':result['object'],
                    'slot_order':result['slot_order'],'frames':result['frames'],
                    **{key:result[key] for key in ('color','kind','slot','novel_combination') if key in result}})
            accepted+=1
            (args.output/'layout_manifest.json').write_text(json.dumps(manifest,indent=2))
            print(split,'accepted matched layout',accepted,'/',total,flush=True)
            if accepted>=total:break
        if accepted<total: raise RuntimeError(f'Only {accepted}/{total} balanced successful layouts for {split}')


if __name__=='__main__':main()
