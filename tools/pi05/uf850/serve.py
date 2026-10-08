"""Serve a tuned 850 policy, including the training action convention."""
import argparse
from pathlib import Path
import numpy as np
import json
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from training import make_config


class SamplingPolicy:
    """Permit repeatable local instruction probes without changing normal inference."""
    def __init__(self,policy): self.policy=policy
    def infer(self,observation):
        observation=dict(observation)
        noise=observation.pop('_sampling_noise',None)
        if noise is None:return self.policy.infer(observation)
        noise=np.asarray(noise,dtype=np.float32)
        if noise.shape!=(10,32) or not np.isfinite(noise).all():
            raise ValueError('Sampling noise must be finite with shape (10,32)')
        return self.policy.infer(observation,noise=noise)

parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--weights',type=Path,required=True)
parser.add_argument('--checkpoint',type=Path,required=True)
parser.add_argument('--repo-id',default='tactfoundry/uf850_bowl_sim_v1')
parser.add_argument('--port',type=int,default=8001)
args=parser.parse_args()
cfg=make_config(args.root,args.repo_id,args.weights)
policy=policy_config.create_trained_policy(cfg,args.checkpoint)
sampling_path=args.root/'sampling-profile.json'
sampling_metadata={'training_sampling':json.loads(sampling_path.read_text())['name']} if sampling_path.exists() else {}
websocket_policy_server.WebsocketPolicyServer(SamplingPolicy(policy),host='127.0.0.1',port=args.port,
    metadata={**cfg.policy_metadata,**sampling_metadata,'checkpoint':str(args.checkpoint),
              'supports_sampling_noise':True}).serve_forever()
