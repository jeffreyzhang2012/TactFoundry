"""Measure held-out joint prediction error; this is not a grasp success test."""
import argparse
import dataclasses
import json
from pathlib import Path
import numpy as np
from openpi.policies import policy_config
from openpi.training import data_loader
from training import make_config

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--repo-id',default='tactfoundry/uf850_bowl_sim_v1')
    parser.add_argument('--samples',type=int,default=32)
    parser.add_argument('--language-probe-source',type=Path,
                        help='Raw matched-layout dataset for controlled instruction sensitivity measurements')
    parser.add_argument('--training-probe-layouts',type=int,default=0)
    args=parser.parse_args()
    if args.samples<1 or args.training_probe_layouts<0:
        parser.error('samples must be positive and training-probe-layouts nonnegative')
    cfg=make_config(args.root,args.repo_id,args.weights)
    policy=policy_config.create_trained_policy(cfg,args.checkpoint)
    data=cfg.data.create(cfg.assets_dirs,cfg.model)
    heldout=dataclasses.replace(data,repo_id=args.repo_id+'_heldout')
    dataset=data_loader.create_torch_dataset(heldout,cfg.model.action_horizon,cfg.model)
    dataset=data_loader.TransformedDataset(dataset,data.repack_transforms.inputs)
    indexes=np.linspace(0,len(dataset)-1,args.samples,dtype=int)
    errors=[]
    hold_errors=[]
    for index in indexes:
        observation=dataset[int(index)]
        target=np.asarray(observation.pop('actions'))
        hold_errors.append(np.asarray(observation['observation/state'])[None,:]-target)
        prediction=policy.infer(observation)['actions']
        errors.append(prediction-target)
    errors=np.asarray(errors)
    hold_errors=np.asarray(hold_errors)
    result={'checkpoint':str(args.checkpoint),'heldout_repo':heldout.repo_id,'samples':len(indexes),
            'arm_mae_rad':float(np.mean(np.abs(errors[...,:6]))),
            'arm_rmse_rad':float(np.sqrt(np.mean(errors[...,:6]**2))),
            'gripper_mae_rad':float(np.mean(np.abs(errors[...,6]))),
            'hold_position_arm_mae_rad':float(np.mean(np.abs(hold_errors[...,:6]))),
            'hold_position_gripper_mae_rad':float(np.mean(np.abs(hold_errors[...,6]))),
            'note':'Held-out action prediction error only; does not establish closed-loop grasp success.'}
    (args.root/'heldout-evaluation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    if args.language_probe_source:
        from language_probe import measure_grounding
        report=measure_grounding(policy.infer,args.language_probe_source,
                                 {**cfg.policy_metadata,'checkpoint':str(args.checkpoint)})
        (args.root/'language-grounding.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2),flush=True)
        if args.training_probe_layouts:
            report=measure_grounding(policy.infer,args.language_probe_source,
                {**cfg.policy_metadata,'checkpoint':str(args.checkpoint)},split='train',
                max_layouts=args.training_probe_layouts)
            (args.root/'training-language-probe.json').write_text(json.dumps(report,indent=2))
            print('Training instruction branch agreement',report['nearest_teacher_branch_agreement'],
                  '/',report['instructions'],flush=True)

if __name__=='__main__': main()
