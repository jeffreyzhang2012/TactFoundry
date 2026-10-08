"""One real policy-controlled LIBERO episode, using upstream observation/action conventions."""
import argparse
import collections
import importlib.util
import json
import os
from pathlib import Path
import time

parser = argparse.ArgumentParser()
parser.add_argument('--openpi-dir', type=Path, default=Path.home() / 'projects/openpi')
parser.add_argument('--suite', default='libero_spatial')
parser.add_argument('--task', type=int, default=0)
parser.add_argument('--episode', type=int, default=0)
parser.add_argument('--port', type=int, default=8000)
parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--output', type=Path, default=Path('data/pi05_demo'))
parser.add_argument('--live-port', type=int, default=0)
parser.add_argument('--step-seconds', type=float, default=0.)
parser.add_argument('--hold-seconds', type=float, default=0.)
args = parser.parse_args()

# Avoid upstream's interactive first-import dataset prompt; no training data is required.
import yaml
root = args.openpi_dir / 'third_party/libero/libero/libero'
configuration = args.output.resolve() / 'libero_config'
configuration.mkdir(parents=True, exist_ok=True)
os.environ['LIBERO_CONFIG_PATH'] = str(configuration)
(configuration / 'config.yaml').write_text(yaml.safe_dump({
    'benchmark_root': str(root), 'bddl_files': str(root / 'bddl_files'),
    'init_states': str(root / 'init_files'), 'assets': str(root / 'assets'),
    'datasets': str(root.parent / 'datasets')}))

import imageio
import numpy as np
from libero.libero import benchmark
from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy

spec = importlib.util.spec_from_file_location('official_libero_example',
                                             args.openpi_dir / 'examples/libero/main.py')
official = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = official
spec.loader.exec_module(official)
np.random.seed(args.seed)
suite = benchmark.get_benchmark_dict()[args.suite]()
task = suite.get_task(args.task)
env, prompt = official._get_libero_env(task, 256, args.seed)
print('Task:', prompt, flush=True)
client = WebsocketClientPolicy('127.0.0.1', args.port)
frames, latencies, action_plan = [], [], collections.deque()
success = False
max_steps = {'libero_spatial': 220, 'libero_object': 280, 'libero_goal': 300,
             'libero_10': 520, 'libero_90': 400}[args.suite]
started = time.monotonic()
live = None
try:
    env.reset()
    obs = env.set_init_state(suite.get_task_init_states(args.task)[args.episode])
    for _ in range(10):
        obs, _, _, _ = env.step(official.LIBERO_DUMMY_ACTION)
    if args.live_port:
        from live_stream import LiveStream
        live = LiveStream(env, args.output, args.live_port)
    for step in range(max_steps):
        image = image_tools.convert_to_uint8(image_tools.resize_with_pad(
            np.ascontiguousarray(obs['agentview_image'][::-1, ::-1]), 224, 224))
        wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(
            np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1, ::-1]), 224, 224))
        frames.append(np.concatenate((image, wrist), axis=1))
        if live:
            live.publish(image, wrist, prompt, step, 'running pi0.5')
        if not action_plan:
            t0 = time.monotonic()
            actions = np.asarray(client.infer({
                'observation/image': image, 'observation/wrist_image': wrist,
                'observation/state': np.concatenate((obs['robot0_eef_pos'],
                    official._quat2axisangle(obs['robot0_eef_quat']), obs['robot0_gripper_qpos'])),
                'prompt': prompt})['actions'])
            if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
                raise ValueError('Invalid LIBERO action chunk')
            latencies.append(time.monotonic() - t0)
            action_plan.extend(actions[:5])
            print(f'Step {step}: inference {latencies[-1]:.3f}s', flush=True)
        obs, _, success, _ = env.step(action_plan.popleft().tolist())
        if success:
            break
        if args.step_seconds:
            time.sleep(args.step_seconds)
    if live:
        image = image_tools.convert_to_uint8(image_tools.resize_with_pad(
            np.ascontiguousarray(obs['agentview_image'][::-1, ::-1]), 224, 224))
        wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(
            np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1, ::-1]), 224, 224))
        deadline = time.monotonic() + args.hold_seconds
        while True:
            live.publish(image, wrist, prompt, step+1, 'SUCCESS' if success else 'episode finished')
            if time.monotonic() >= deadline:
                break
            time.sleep(.1)
finally:
    if live:
        live.close()
    env.close()
args.output.mkdir(parents=True, exist_ok=True)
stem = f'{args.suite}_task{args.task}_episode{args.episode}'
video = args.output / f'{stem}.mp4'
imageio.mimwrite(str(video), frames, fps=10)
result = dict(model='pi05_libero', checkpoint='gs://openpi-assets/checkpoints/pi05_libero',
              environment='LIBERO (benchmark robot, not UFACTORY 850)', prompt=prompt,
              success=bool(success), steps=step+1, seed=args.seed,
              inference_seconds=latencies, elapsed_seconds=time.monotonic()-started,
              video=str(video))
(args.output / f'{stem}.json').write_text(json.dumps(result, indent=2))
print(json.dumps({k:v for k,v in result.items() if k != 'inference_seconds'}, indent=2), flush=True)
