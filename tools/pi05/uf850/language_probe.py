"""Change only the instruction in matched scenes, using fixed diffusion noise."""
import argparse
import json
from pathlib import Path
import numpy as np
from openpi_client import websocket_client_policy


def measure_grounding(infer,source,metadata,*,split='heldout',max_layouts=None):
    manifest=json.loads((source/'layout_manifest.json').read_text())
    groups={}
    for row in manifest[split]:groups.setdefault(row['layout_seed'],[]).append(row)
    if max_layouts is not None:groups=dict(list(groups.items())[:max_layouts])
    probes=[]
    noise=np.random.default_rng(7).standard_normal((10,32)).astype(np.float32)
    for seed,rows in groups.items():
        data=[np.load(source/f"episode_{row['seed']:04d}.npz") for row in rows]
        metas=[json.loads((source/f"attempt_{row['seed']:04d}.json").read_text()) for row in rows]
        at=12
        reference=data[0]
        for other in data[1:]:
            if not (np.allclose(other['state'][at],reference['state'][at],atol=1e-6) and
                    np.array_equal(other['image'][at],reference['image'][at]) and
                    np.array_equal(other['wrist_image'][at],reference['wrist_image'][at])):
                raise ValueError('Language probe requires identical matched observations')
        targets=np.array([demo['actions'][at:at+10,:6] for demo in data])
        predictions=[]
        for meta in metas:
            prediction=infer({'observation/state':reference['state'][at],
                'observation/image':reference['image'][at],'observation/wrist_image':reference['wrist_image'][at],
                'prompt':meta['prompt']},noise=noise)['actions']
            predictions.append(np.asarray(prediction)[:,:6])
        errors=np.mean(np.abs(np.asarray(predictions)[:,None]-targets[None]),axis=(2,3))
        probes.append({'layout_seed':seed,'objects':[meta['object'] for meta in metas],
            'prompts':[meta['prompt'] for meta in metas],'arm_mae_matrix_rad':errors.tolist(),
            'nearest_teacher_branch':np.argmin(errors,axis=1).tolist(),
            'matching_branches':int(np.sum(np.argmin(errors,axis=1)==np.arange(len(metas))))})
    return {'demonstration_split':split,'layouts':len(probes),'instructions':sum(len(row['objects']) for row in probes),
        'nearest_teacher_branch_agreement':sum(row['matching_branches'] for row in probes),
        'policy_metadata':metadata,'probes':probes,
        'note':'Only instruction changes; same RGB/state and diffusion noise. First-reach action proxy, not physical task success.'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--port',type=int,default=8001)
    parser.add_argument('--split',choices=['train','heldout'],default='heldout')
    parser.add_argument('--max-layouts',type=int)
    args=parser.parse_args()
    if args.max_layouts is not None and args.max_layouts<1:parser.error('max-layouts must be positive')
    client=websocket_client_policy.WebsocketClientPolicy('127.0.0.1',args.port)
    metadata=client.get_server_metadata()
    if not metadata.get('supports_sampling_noise'):
        raise ValueError('Restart the updated 850 policy server; this server cannot control diffusion noise')
    def infer(observation,*,noise):
        return client.infer({**observation,'_sampling_noise':noise})
    report=measure_grounding(infer,args.source,metadata,split=args.split,max_layouts=args.max_layouts)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
