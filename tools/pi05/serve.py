"""Official pi05_libero checkpoint, served locally for the benchmark demo."""
import argparse
import logging
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config
from openpi.shared import download

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=8000)
args = parser.parse_args()
logging.basicConfig(level=logging.INFO, force=True)
checkpoint = download.maybe_download('gs://openpi-assets/checkpoints/pi05_libero', token='anon')
# fsspec retries into an existing partial directory can nest the completed tree.
if (checkpoint / 'pi05_libero' / 'params').is_dir():
    checkpoint = checkpoint / 'pi05_libero'
logging.info('Using checkpoint at %s', checkpoint)
policy = policy_config.create_trained_policy(config.get_config('pi05_libero'), checkpoint)
websocket_policy_server.WebsocketPolicyServer(
    policy=policy, host='127.0.0.1', port=args.port, metadata=policy.metadata).serve_forever()
