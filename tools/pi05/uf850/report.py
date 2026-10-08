"""Summarize completed training and physical trials without conflating metrics."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tasks import OBJECTS


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--train-log',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--updates',type=int,required=True)
    parser.add_argument('--ancestry-updates',type=int,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    rows=re.findall(r'Step (\d+): grad_norm=([\d.]+), loss=([\d.]+)',args.train_log.read_text())
    if not rows:raise ValueError('No training loss records found')
    steps=np.array([int(row[0]) for row in rows]);loss=np.array([float(row[2]) for row in rows])
    metrics=json.loads((args.root/'heldout-evaluation.json').read_text())
    grounding=json.loads((args.root/'language-grounding.json').read_text())
    manifest=json.loads((args.source/'split.json').read_text())
    results={}
    for category in ('baseline','unseen_layouts','original_scene'):
        if not (args.root/category).is_dir():continue
        results[category]={}
        for obj in OBJECTS:
            trial=json.loads((args.root/category/obj/'results.json').read_text())
            results[category][obj]={'successes':sum(row['success'] for row in trial),'trials':len(trial),
                'seeds':[row['seed'] for row in trial],'results':trial,
                'requested_grasps':sum(row.get('object_activity',{}).get(obj,{}).get('grasped_and_lifted',False)
                                       for row in trial)}
    profile=args.root/'sampling-profile.json'
    train_probe_path=args.root/'training-language-probe.json'
    reference_path=args.root/'teacher-reference.json'
    reference=json.loads(reference_path.read_text()) if reference_path.exists() else None
    if reference:
        for obj in OBJECTS:
            expected=set(results['unseen_layouts'][obj]['seeds'])
            actual={row['layout_seed'] for row in reference['results'] if row['object']==obj}
            if actual!=expected:raise ValueError('Teacher reference does not match physical trial starts')
    report={'new_gradient_updates':args.updates,'gradient_updates_in_checkpoint_ancestry':args.ancestry_updates,
        'train_episodes':len(manifest['train']),'heldout_episodes':len(manifest['heldout']),
        'train_frames':sum(row['frames'] for row in manifest['train']),
        'heldout_frames':sum(row['frames'] for row in manifest['heldout']),
        'first_logged_loss':float(loss[0]),'last_logged_loss':float(loss[-1]),
        'last_100_updates_mean_loss':float(loss[-10:].mean()),
        'checkpoint':metrics['checkpoint'],'heldout_action_metrics':metrics,'language_grounding':grounding,
        'sampling':json.loads(profile.read_text()) if profile.exists() else 'uniform frames',
        'training_language_probe':json.loads(train_probe_path.read_text()) if train_probe_path.exists() else None,
        'physical_teacher_reference':reference,
        'closed_loop':results,
        'limitations':'Small stochastic simulation study. Familiar objects; fixed cameras, geometry and lighting. '
                       'Teacher-reachable demonstration layouts. No real-robot or novel-object validation. '
                       'The language probe uses validation layouts, not an independent final test.'}
    (args.output/'training-summary.json').write_text(json.dumps(report,indent=2))
    fig,axes=plt.subplots(1,3,figsize=(14,4.5))
    axes[0].semilogy(steps,loss,color='#246bb1',linewidth=1.3)
    axes[0].set(xlabel='Additional updates in this pass',ylabel='Training diffusion loss',title='Fine-tuning')
    x=np.arange(2)
    axes[1].bar(x-.18,[metrics['arm_mae_rad'],metrics['gripper_mae_rad']],width=.36,label='Policy',color='#246bb1')
    axes[1].bar(x+.18,[metrics['hold_position_arm_mae_rad'],metrics['hold_position_gripper_mae_rad']],
                width=.36,label='Hold position',color='#aaa')
    axes[1].set(xticks=x,xticklabels=['Arm','Gripper'],ylabel='MAE (radians)',title='Held-out action prediction')
    axes[1].legend(fontsize=8)
    scores=results['unseen_layouts'];n=max(row['trials'] for row in scores.values())
    x=np.arange(3)
    if reference:
        axes[2].bar(x-.18,[scores[obj]['successes'] for obj in OBJECTS],width=.36,color='#246bb1',label='Policy')
        axes[2].bar(x+.18,[sum(row['success'] for row in reference['results'] if row['object']==obj) for obj in OBJECTS],
                    width=.36,color='#aaa',label='IK teacher reference')
        axes[2].legend(fontsize=8)
    else:
        axes[2].bar(x,[scores[obj]['successes'] for obj in OBJECTS],color='#246bb1')
    axes[2].set(xticks=x,xticklabels=['Bowl','Bottle','Block'],ylabel=f'Successful placements / {n}',
                ylim=(0,n+.4),yticks=range(n+1),title='Excluded-layout physical trials')
    for ax in axes:ax.grid(axis='y',alpha=.15)
    agreement=grounding['nearest_teacher_branch_agreement'];total=grounding['instructions']
    fig.suptitle(f'UF850 + AG95: {args.updates:,} additional updates; initial instruction branch {agreement}/{total}',fontsize=13)
    fig.text(.5,.02,'Action error and instruction sensitivity do not establish physical task success. Simulation only.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,.94]);fig.savefig(args.output/'training-results.png',dpi=160);plt.close(fig)
    ffmpeg=next((Path.home()/'projects/openpi/.venv/lib').glob('python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux*'))
    for obj in OBJECTS:
        folder=args.root/'unseen_layouts'/obj
        playlist=folder/'videos.txt'
        playlist.write_text(''.join(f"file '{folder}/rollout_{row['seed']}.mp4'\n" for row in scores[obj]['results']))
        subprocess.run([str(ffmpeg),'-y','-loglevel','error','-f','concat','-safe','0','-i',str(playlist),
                        '-c','copy',str(args.output/f'unseen_{obj}.mp4')],check=True)
    print(json.dumps({'loss_first_last':[float(loss[0]),float(loss[-1])],
        'action_metrics':metrics,'language_agreement':[agreement,total],
        'placements':{category:{obj:[row['successes'],row['trials']] for obj,row in scores.items()}
                      for category,scores in results.items()}},indent=2),flush=True)


if __name__=='__main__':main()
