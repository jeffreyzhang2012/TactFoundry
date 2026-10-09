"""Convert verified local demonstrations to separate train/held-out LeRobot sets."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--repo-id',default='tactfoundry/uf850_bowl_sim_v1')
    parser.add_argument('--heldout-seeds',default='',help='Comma-separated seeds to preserve a prior evaluation split')
    parser.add_argument('--reuse-repo-id',default='',help='Copy a prior local dataset and append only new episodes')
    parser.add_argument('--prior-split',type=Path)
    args=parser.parse_args()
    layout_manifest_path=args.source/'layout_manifest.json'
    layout_manifest=json.loads(layout_manifest_path.read_text()) if layout_manifest_path.exists() else None
    episodes=sorted(args.source.glob('episode_*.npz'))
    if len(episodes)<5: raise ValueError('Need at least five verified successful episodes')
    heldout=max(1,len(episodes)//5)
    heldout_seeds={int(v) for v in args.heldout_seeds.split(',') if v}
    if layout_manifest and not heldout_seeds:
        heldout_seeds={row['seed'] for row in layout_manifest['heldout']}
    present={int(path.stem.split('_')[1]) for path in episodes}
    if not heldout_seeds.issubset(present): raise ValueError('Some requested held-out seeds have no successful episode')
    validation=[path for path in episodes if int(path.stem.split('_')[1]) in heldout_seeds] if heldout_seeds else episodes[-heldout:]
    training=[path for path in episodes if path not in validation]
    if not training or not validation: raise ValueError('Need nonempty training and held-out splits')
    manifest={'train':[], 'heldout':[], 'fps':20, 'action_convention':'absolute 6 arm joint targets + AG95 angle, radians'}
    if layout_manifest:
        groups={split:{row['layout_seed'] for row in layout_manifest[split]} for split in ('train','heldout')}
        if groups['train'] & groups['heldout']: raise ValueError('Layout leakage across training and validation')
        selected_heldout={int(path.stem.split('_')[1]) for path in validation}
        if selected_heldout!={row['seed'] for row in layout_manifest['heldout']}:
            raise ValueError('All instructions from a layout must stay in the same split')
        manifest['layout_groups']={key:sorted(value) for key,value in groups.items()}
    prior=json.loads(args.prior_split.read_text()) if args.prior_split else {}
    if args.reuse_repo_id and not prior: raise ValueError('Reusing a dataset requires its prior split manifest')
    if args.reuse_repo_id:
        for split,paths in [('train',training),('heldout',validation)]:
            selected={int(path.stem.split('_')[1]) for path in paths}
            if not {row['seed'] for row in prior[split]}.issubset(selected):
                raise ValueError('Reusing data must preserve the previous train/held-out split')
    features={
        'image': {'dtype':'image','shape':(224,224,3),'names':['height','width','channel']},
        'wrist_image': {'dtype':'image','shape':(224,224,3),'names':['height','width','channel']},
        'state': {'dtype':'float32','shape':(7,),'names':['joints']},
        'actions': {'dtype':'float32','shape':(7,),'names':['joint_targets']},
    }
    for split, paths in [('train',training),('heldout',validation)]:
        repo=args.repo_id+('_heldout' if split=='heldout' else '')
        dest=HF_LEROBOT_HOME/repo
        if dest.exists(): raise FileExistsError(f'{dest} already exists; choose a new repo-id')
        if args.reuse_repo_id:
            base_repo=args.reuse_repo_id+('_heldout' if split=='heldout' else '')
            shutil.copytree(HF_LEROBOT_HOME/base_repo,dest)
            dataset=LeRobotDataset(repo_id=repo)
            dataset.start_image_writer(num_threads=4)
            manifest[split]=list(prior[split])
        else:
            dataset=LeRobotDataset.create(repo_id=repo,robot_type='uf850_ag95',fps=20,
                                          features=features,image_writer_threads=4,image_writer_processes=0)
        prior_seeds={row['seed'] for row in manifest[split]}
        for path in paths:
            seed=int(path.stem.split('_')[1])
            if seed in prior_seeds: continue
            meta=json.loads((args.source/f'attempt_{seed:04d}.json').read_text())
            if not meta['success']: raise ValueError('Refusing unsuccessful demonstration')
            # NpzFile does not cache decompressed arrays. Materialize each
            # feature once rather than reloading a whole RGB episode per frame.
            with np.load(path) as archive:
                data={key:archive[key] for key in features}
            if not all(len(data[key])==len(data['state']) for key in features): raise ValueError('Unsynchronized episode')
            for i in range(len(data['state'])):
                dataset.add_frame({key:data[key][i] for key in features}|{'task':meta['prompt']})
            dataset.save_episode()
            manifest[split].append({'seed':seed,'frames':len(data['state']),
                                    **{key:meta[key] for key in ('object','layout_seed','slot_order','color','kind','slot','novel_combination') if key in meta}})
            print(split,'saved seed',seed,flush=True)
        dataset.stop_image_writer()
        print(split,repo,len(paths),'episodes',flush=True)
    (args.source/'split.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
