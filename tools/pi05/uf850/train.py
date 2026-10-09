"""Run OpenPI normalization/training with a repository-local robot config."""
import argparse
import importlib.util
import logging
from pathlib import Path
import sys
import dataclasses
import json
import numpy as np
from training import make_config, register

def main():
    sys.stdout.reconfigure(line_buffering=True)
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['stats','train'])
    parser.add_argument('--openpi',type=Path,default=Path.home()/'projects/openpi')
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--repo-id',default='tactfoundry/uf850_bowl_sim_v1')
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=300)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--exp-name',default='pilot')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--pure-noise-probability',type=float)
    parser.add_argument('--freeze-vision',action='store_true')
    parser.add_argument('--diverse-sampling',action='store_true')
    parser.add_argument('--state-min-range',type=float,default=0.)
    parser.add_argument('--action-min-range',type=float,default=0.)
    parser.add_argument('--selection-manifest',type=Path,
                        help='Resample initial target-selection frames using a matched-layout training split')
    args=parser.parse_args()
    if args.diverse_sampling and not args.selection_manifest:
        parser.error('diverse sampling requires a selection manifest')
    if args.freeze_vision and args.pure_noise_probability is None:
        args.pure_noise_probability=0.
    if args.pure_noise_probability is not None:
        if not np.isfinite(args.pure_noise_probability) or not 0<=args.pure_noise_probability<=1:
            parser.error('pure noise probability must be finite and between zero and one')
        args.root.mkdir(parents=True,exist_ok=True)
        (args.root/'training-profile.json').write_text(json.dumps({
            'pure_noise_probability':args.pure_noise_probability,'freeze_vision':args.freeze_vision,
            'initializer':str(args.weights),'optimizer':'fresh unless --resume',
            'loss_axes':7,'objective':'standard flow plus pure-noise instruction-only cases'},indent=2))
    if args.steps<1 or args.batch_size<1 or not np.isfinite([args.state_min_range,args.action_min_range]).all() or min(args.state_min_range,args.action_min_range)<0:
        parser.error('steps and batch size must be positive; normalization ranges must be finite and nonnegative')
    cfg=make_config(args.root,args.repo_id,args.weights,steps=args.steps,batch_size=args.batch_size,
                    exp_name=args.exp_name,resume=args.resume)
    register(cfg)
    logging.basicConfig(level=logging.INFO)
    name='compute_norm_stats' if args.stage=='stats' else 'train'
    spec=importlib.util.spec_from_file_location('upstream_'+name,args.openpi/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module)
    if args.stage=='stats':
        module.main(cfg.name)
        if args.state_min_range or args.action_min_range:
            from openpi.shared import normalize
            path=cfg.assets_dirs/args.repo_id
            stats=normalize.load(path)
            for key,floor in [('state',args.state_min_range),('actions',args.action_min_range)]:
                item=stats[key]
                center=(item.q01+item.q99)/2
                half=np.maximum((item.q99-item.q01)/2,floor/2)
                stats[key]=dataclasses.replace(item,q01=center-half,q99=center+half,std=np.maximum(item.std,floor/2))
            normalize.save(path,stats)
            (path/'normalization_floor.json').write_text(json.dumps({'state_min_range_rad':args.state_min_range,
                                                                  'action_min_range_rad':args.action_min_range},indent=2))
    elif args.selection_manifest:
        if args.diverse_sampling:
            from diverse_sampling import selection_sampling
        else:
            from focus_sampling import selection_sampling
        with selection_sampling(args.repo_id,args.selection_manifest,args.root/'sampling-profile.json') as profile:
            cfg=dataclasses.replace(cfg,policy_metadata={**cfg.policy_metadata,'training_sampling':profile['name']})
            module.main(cfg)
    else: module.main(cfg)

if __name__=='__main__': main()
