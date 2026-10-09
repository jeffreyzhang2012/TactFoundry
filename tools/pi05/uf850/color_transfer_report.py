"""Report paired familiar/unseen-color trials without overstating transfer."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    probe=json.loads((args.root/'instruction-color-probe.json').read_text())
    teacher=json.loads((args.root/'teacher-reference.json').read_text())
    results={};summary={}
    for variant in ('known','cyan'):
        rows=json.loads((args.root/variant/'results.json').read_text())
        if [r['seed'] for r in rows]!=list(range(42001,42013)):raise ValueError('Missing predetermined trials')
        results[variant]=rows
        summary[variant]={'color':'red' if variant=='known' else 'cyan','trials':len(rows),
            'correct_first_attempts':sum(bool((r.get('first_grasp_attempt') or {}).get('correct')) for r in rows),
            'attempts':sum(r.get('first_grasp_attempt') is not None for r in rows),
            'requested_grasps':sum(r['object_activity']['target_0']['grasped_and_lifted'] for r in rows),
            'wrong_object_grasps':sum(any(v['grasped_and_lifted'] for k,v in r['object_activity'].items() if k!='target_0') for r in rows),
            'successful_placements':sum(r['success'] for r in rows),
            'target_reach_branch_agreement':probe['variants'][variant]['target_branch_agreement'],
            'all_instruction_branch_agreement':probe['variants'][variant]['all_instruction_branch_agreement']}
        for video in (args.root/variant).glob('*.mp4'):shutil.copy2(video,args.output/f'{variant}_{video.name}')
    for a,b in zip(results['known'],results['cyan']):
        for key in ('xy','scale','mass','friction','yaw','slot','kind'):
            assert a['layout']['objects'][0][key]==b['layout']['objects'][0][key]
        assert a['sampling_seed']==b['sampling_seed']==170
        assert a['policy_metadata']['checkpoint']==b['policy_metadata']['checkpoint']==probe['policy_metadata']['checkpoint']
        for name,position in a['initial_object_positions'].items():
            np.testing.assert_allclose(position,b['initial_object_positions'][name],atol=1e-5,rtol=0)
        with np.load(args.root/'known'/f"trace_{a['seed']}.npz") as first,\
             np.load(args.root/'cyan'/f"trace_{b['seed']}.npz") as second:
            np.testing.assert_allclose(first['joints_and_targets'][0,:7],second['joints_and_targets'][0,:7],atol=1e-5,rtol=0)
    report={'checkpoint':probe['policy_metadata']['checkpoint'],'retrained':False,'novel_color':'cyan',
        'familiar_control_color':'red','shapes':['bowl','bottle','cube','cylinder'],
        'seeds':list(range(42001,42013)),'paired_sampling_seed':170,'summary':summary,
        'results':results,'instruction_probe':probe,'teacher_reference':teacher,
        'limitations':'Cyan is unseen in this robot fine-tune, not necessarily in foundation-model pretraining. '
            'Small fixed-camera simulation experiment. Same familiar procedural geometry. All starts retained; '
            'first-reach agreement and near-object closing attempts do not establish grasp or placement success.'}
    (args.output/'color-transfer-summary.json').write_text(json.dumps(report,indent=2))
    for name in ('instruction-color-probe.json','teacher-reference.json','color-test-scenes.png'):
        shutil.copy2(args.root/name,args.output/name)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    x=np.arange(2)
    axes[0].bar(['Red (familiar)','Cyan (unseen)'],[summary[v]['target_reach_branch_agreement'] for v in ('known','cyan')],
        color=['#bb2222','#149da8'])
    axes[0].set(title='Requested target: initial reach',ylabel='Teacher branch agreement / 12',ylim=(0,13),yticks=range(0,13,2))
    labels=['Correct closing attempt','Grasp and lift','Stable placement'];xx=np.arange(3)
    for shift,variant,color in [(-.18,'known','#bb2222'),(.18,'cyan','#149da8')]:
        values=[summary[variant][k] for k in ('correct_first_attempts','requested_grasps','successful_placements')]
        axes[1].bar(xx+shift,values,.36,label=summary[variant]['color'],color=color)
        for i,value in enumerate(values):axes[1].text(i+shift,value+.15,str(value),ha='center',fontsize=8)
    axes[1].set(title='Paired physical trials',xticks=xx,xticklabels=labels,ylabel='Trials / 12',ylim=(0,13),yticks=range(0,13,2))
    axes[1].legend()
    for ax in axes:ax.grid(axis='y',alpha=.15)
    fig.suptitle('Unchanged π0.5 V6: novel cyan versus familiar red; 4 shapes × 3 positions')
    fig.text(.5,.01,'Color and its instruction word change; paired geometry, physics and sampling noise are fixed. Simulation only.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.04,1,.94]);fig.savefig(args.output/'color-transfer-results.png',dpi=160);plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
