"""Serve a tuned 850 policy, including the training action convention."""
import argparse
from pathlib import Path
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from training import make_config

parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--weights',type=Path,required=True)
parser.add_argument('--checkpoint',type=Path,required=True)
parser.add_argument('--repo-id',default='tactfoundry/uf850_bowl_sim_v1')
parser.add_argument('--port',type=int,default=8001)
args=parser.parse_args()
cfg=make_config(args.root,args.repo_id,args.weights)
policy=policy_config.create_trained_policy(cfg,args.checkpoint)
websocket_policy_server.WebsocketPolicyServer(policy,host='127.0.0.1',port=args.port,
    metadata=cfg.policy_metadata).serve_forever()
