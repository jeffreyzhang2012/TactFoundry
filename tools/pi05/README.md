# π0.5 manipulation demo

This runs the official **π0.5-LIBERO** checkpoint in the matching LIBERO MuJoCo
environment. It uses the benchmark robot, external and wrist cameras, task
instruction, and the upstream state/action encoding. It does not control the
UFACTORY 850, AG95, real hardware, or the PyBullet playground.

Tested upstream revision: `215abfb217dbac7d5f1273282331b9b1866c0479` from
[Physical Intelligence OpenPI](https://github.com/Physical-Intelligence/openpi).
The installer keeps separate Python 3.11 model and Python 3.8 simulator environments
in `~/projects/openpi`, leaving ROS Humble's Python environment intact.

In WSL Ubuntu:

```bash
cd ~/projects/TactFoundry
bash tools/pi05/install.sh
```

Start the local model server in one terminal:

```bash
bash tools/pi05/serve.sh
```

The first start downloads `gs://openpi-assets/checkpoints/pi05_libero` to
`~/.cache/openpi/openpi-assets/checkpoints/pi05_libero`. Later starts reuse the
cached weights. The server listens on localhost port 8000 and uses the NVIDIA
GPU via JAX; the first request includes compilation time.

In a second terminal, run one episode:

```bash
bash tools/pi05/demo.sh
# Another benchmark task / initial state:
bash tools/pi05/demo.sh --suite libero_spatial --task 1 --episode 0
```

The demo saves a side-by-side external/wrist camera video and a JSON result
containing the task, checkpoint, success flag, step count and inference timings
under `data/pi05_demo/`. It executes five actions per predicted chunk before
observing again, following the upstream example. Success is measured by the
LIBERO environment, not inferred from appearance. The renderer uses Xvfb and
software GL so it can run without a visible simulator window. Install `xvfb`
and `libgl1-mesa-dri` if absent. `OPENPI_DIR` selects an alternate installation.

## Live RViz view

With the model server already running, open live visualization in WSL:

```bash
bash tools/pi05/live.sh
```

This opens a separate RViz window on ROS domain 43, displaying the actual
LIBERO robot/object poses and external/wrist image topics. Each episode runs
fresh pi0.5 inference, holds its final state for five seconds, then restarts.
Use Ctrl+C to stop. The model server stays available on port 8000.

`/pi05/scene` contains visible MuJoCo geometry exported as mesh/primitive
markers; materials use solid colors (textures and capsule end caps are not
reproduced). The camera images are the actual simulator renders. `/pi05/image/image_raw`
and `/pi05/wrist/image_raw` carry the live RGB feeds. A localhost TCP bridge
separates LIBERO's Python 3.8 environment from ROS Humble's Python 3.10.
Mesh caches and recorded episodes stay under the ignored `data/` directory.

## Verified run

On October 8, 2026, task 0 / episode 0 of `libero_spatial` with seed 7 succeeded:
"pick up the black bowl between the plate and the ramekin and place it on the plate."
LIBERO reported success after 102 actions. On the RTX 5090, first inference took
29.75 seconds including compilation; subsequent chunks took roughly 0.08–0.10 seconds.
This is one successful episode, not a benchmark-wide success-rate measurement.

## Fine-tuning for our robot later

The first simulation adaptation pipeline now lives in
[uf850/README.md](uf850/README.md): verified physical-contact demonstrations,
robot-specific joint actions, separate held-out episodes, LoRA training and
simulation policy rollout. This is separate from the official LIBERO demo.

1. Record synchronized wrist RGB, preferably an external RGB view, six arm joint
   positions, AG95 opening, actual commanded actions, timestamps, and task text
   from successful teleoperated 850 demonstrations. Keep the camera calibration
   and action frame/convention explicit; simulation data alone may not transfer
   to real hardware. Jaw-force recordings can accompany the dataset, but adding
   them to policy inputs requires a matching model/data transform.
2. Convert to a LeRobot dataset and implement OpenPI observation/action transforms
   for the **850 + AG95**, including the gripper convention, normalization statistics,
   control rate, and action chunks. Do not reinterpret Franka joint actions as 850 actions.
3. Initialize a matching training configuration from `pi05_base`, train on those
   demonstrations, then evaluate held-out episodes. The RTX 5090's 32 GB is a
   candidate for LoRA; upstream lists >22.5 GB for LoRA and >70 GB for full tuning.
   Actual memory needs depend on the chosen model configuration and batch size.
4. Connect the resulting policy to this project's simulation first, using its
   trained action convention and fresh observations. Add a real-arm driver and
   measured transforms for physical deployment.

References: [official LIBERO example](https://github.com/Physical-Intelligence/openpi/tree/main/examples/libero),
[OpenPI fine-tuning guide](https://github.com/Physical-Intelligence/openpi#fine-tuning-base-models-on-your-own-data).
