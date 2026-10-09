"""Summarize a completed fresh-base diverse experiment and physical trials."""
import argparse,json,re,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--train-log',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    records=re.findall(r'Step (\d+): grad_norm=([\d.]+), loss=([\d.]+)',args.train_log.read_text())
    if not records:raise ValueError('No training loss records')
    steps=np.array([int(r[0]) for r in records]);loss=np.array([float(r[2]) for r in records])
    def load(name):return json.loads((args.root/name).read_text())
    action=load('heldout-evaluation.json');grounding=load('language-grounding.json')
    train_grounding=load('training-language-probe.json');teacher=load('teacher-reference.json')
    split=json.loads((args.source/'split.json').read_text())
    trials={}
    for category in ('unseen_diverse','original_scene'):
        trials[category]={}
        for folder in sorted((args.root/category).iterdir()):
            rows=json.loads((folder/'results.json').read_text())
            trials[category][folder.name]={'successes':sum(r['success'] for r in rows),'trials':len(rows),
                'requested_grasps':sum(r['object_activity'][r['object']]['grasped_and_lifted'] for r in rows),
                'correct_first_grasp_attempts':sum(bool((r.get('first_grasp_attempt') or {}).get('correct')) for r in rows),
                'results':rows}
            for video in folder.glob('*.mp4'):shutil.copy2(video,args.output/f'{category}_{folder.name}_{video.name}')
    summary={'checkpoint':action['checkpoint'],'new_gradient_updates':4000,
        'initializer':load('training-profile.json')['initializer'],'continues_prior_tactfoundry_weights':False,
        'optimizer':'fresh','dataset':load('dataset-profile.json'),'coverage':load('data-coverage.json'),
        'train_frames':sum(r['frames'] for r in split['train']),
        'heldout_frames':sum(r['frames'] for r in split['heldout']),
        'first_logged_loss':float(loss[0]),'last_logged_loss':float(loss[-1]),
        'training_profile':load('training-profile.json'),'sampling_profile':load('sampling-profile.json'),
        'heldout_action_metrics':action,'language_grounding':grounding,'training_language_probe':train_grounding,
        'physical_teacher_reference':teacher,'closed_loop':trials,
        'limitations':'Simulation only. Fixed camera extrinsics and lighting, procedural geometry, success-filtered teacher demonstrations. '
            'Held-out color/shape pairs and layouts do not establish novel geometry, camera viewpoint or real-robot performance. '
            'Training loss uses seven real axes and a modified noise distribution; it is not directly comparable to earlier losses.'}
    (args.output/'training-summary.json').write_text(json.dumps(summary,indent=2))
    for name in ('heldout-evaluation.json','language-grounding.json','training-language-probe.json',
                 'training-profile.json','sampling-profile.json','data-coverage.json','teacher-reference.json'):
        shutil.copy2(args.root/name,args.output/name)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5))
    axes[0].semilogy(steps,loss,color='#246bb1');axes[0].set(title='Fresh base fine-tune',xlabel='Update',ylabel='Training flow loss')
    fractions=[train_grounding['nearest_teacher_branch_agreement']/train_grounding['instructions'],
               grounding['nearest_teacher_branch_agreement']/grounding['instructions']]
    axes[1].bar(['Train scenes','Held-out scenes'],fractions,color=['#888888','#246bb1'])
    axes[1].set(title='Instruction branch agreement',ylabel='Fraction',ylim=(0,1))
    objects=['target_0','target_1','target_2'];x=np.arange(3)
    policy=[trials['unseen_diverse'][o]['successes'] for o in objects]
    reference=[sum(r['success'] for r in teacher['results'] if r['object']==o) for o in objects]
    axes[2].bar(x-.18,policy,.36,label='Policy',color='#246bb1')
    axes[2].bar(x+.18,reference,.36,label='IK reference',color='#888888')
    axes[2].set(title='Fresh physical simulation trials',xticks=x,xticklabels=['Target 0','Target 1','Target 2'],
                ylabel='Placements / 3',yticks=range(4),ylim=(0,3.5));axes[2].legend(fontsize=8)
    for ax in axes:ax.grid(axis='y',alpha=.15)
    fig.suptitle('UF850 + AG95: 360 diverse demonstrations, 4,000 updates from pretrained π0.5')
    fig.text(.5,.01,'Instruction agreement is a first-reach proxy. Physical success requires grasp, lift and stable release on the plate.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.94]);fig.savefig(args.output/'training-results.png',dpi=160);plt.close(fig)
    print(json.dumps({'loss_first_last':[float(loss[0]),float(loss[-1])],
        'instruction_agreement':fractions,'placements':{k:{o:[r['successes'],r['trials']] for o,r in v.items()} for k,v in trials.items()}}),flush=True)


if __name__=='__main__':main()
