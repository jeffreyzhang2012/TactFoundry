# π0.5 adaptation to the 850 + AG95

This directory provides a simulation demonstration generator, local LeRobot
conversion, π0.5 LoRA configuration, evaluation and a simulation-only rollout.
It extends the pinned OpenPI checkout without modifying its source or
uploading datasets/checkpoints. The original LIBERO policy is a separate demo.

## Robot and data contract

- Robot: UFACTORY 850 with the 90-degree AG95 mount, wrist D435 and external
  camera from the tabletop scene. The xArm6 is not included in this dataset.
- Observations: external and wrist RGB, plus seven measured joint angles
  `[joint1, ..., joint6, ag95_left_outer_knuckle_joint]`, in radians.
- Recorded actions: seven absolute motor targets for the next 50 ms interval.
  The AG95 value increases from 0 (open) toward 0.93 (closed).
- Training converts the first six action dimensions to offsets relative to the
  observed joints at the beginning of each chunk; the AG95 angle stays absolute.
  Inference converts the arm outputs back to absolute targets. No Franka
  Cartesian actions or gripper sign convention are reused.
- 20 Hz recordings and 10-step prediction chunks. Rollout executes four actions
  before observing again by default; `--execute-steps 1` replans each step.
  RGB renders are 224x224 with a 58-degree vertical FOV;
  this pilot's square renders have a narrower horizontal FOV than the normal
  480x360 external RViz stream. Rollout renders match the training views.
- The simulation teacher uses a **3 Nm cap per AG95 motor**. This avoids the
  unstable contact response found at the URDF's 50 Nm motor limit; it is a
  solver setting, not a calibrated physical actuator model. To teleoperate
  the matching physics profile, launch `tabletop.launch.py gripper_effort:=3.0`.

The teacher moves through IK waypoints, closes the jaws, lifts, transports and
releases. It never attaches the object to the gripper or changes the object's
pose during manipulation. Success requires bilateral jaw contact, a measured
lift, and an upright bowl resting near the plate center after settling. Only
successful attempts become training episodes. Initial bowl XY positions vary
by ±15 mm; lighting, object geometry and the other props remain fixed.

## Run in WSL Ubuntu

Install OpenPI using `bash tools/pi05/install.sh` first and build this ROS
workspace. Generation uses ROS Python 3.10; conversion/training use OpenPI's
separate Python 3.11 environment. For the rollout client:

```bash
python3 -m pip install --user numpy==1.24.4 -e ~/projects/openpi/packages/openpi-client
```

From `~/projects/TactFoundry`:

```bash
bash tools/pi05/uf850/run.sh generate --episodes 20
bash tools/pi05/uf850/run.sh convert
bash tools/pi05/uf850/run.sh download
bash tools/pi05/uf850/run.sh stats --steps 1000
# Stop the original benchmark model server to free GPU memory first.
bash tools/pi05/uf850/run.sh train --steps 1000 --exp-name pilot1000
```

Generation refuses to overwrite existing successful episodes, and conversion
refuses to overwrite an existing dataset. Choose a new output path/repo ID for
new recordings. The converter reserves the final 20% of successful episodes
as a distinct held-out dataset; no frames from those episodes enter training
or normalization statistics. No Hugging Face upload is performed. The pilot
uses batch size 4, LoRA in both language/action experts, EMA off and W&B off.
The frozen model and LoRA state use the official JAX training/checkpoint code.

Raw demonstrations are in `data/uf850_sim_v1/`, local LeRobot data under
`~/.cache/huggingface/lerobot/tactfoundry/`, and model assets/checkpoints under
`data/uf850_tuning_v1/`. These large artifacts are ignored by Git. Set
`UF850_RUN_DIR`, `UF850_DATA_DIR`, `UF850_REPO_ID`, `UF850_BASE_WEIGHTS` or
`OPENPI_DIR` to change their locations.
To continue a run, supply the same experiment name and `--resume`.

## Check the policy

After the 1,000-step run, the final checkpoint is named `999` by upstream:

```bash
checkpoint="$HOME/projects/TactFoundry/data/uf850_tuning_v1/checkpoints/pi05_uf850_lora/pilot1000/999"
bash tools/pi05/uf850/run.sh evaluate --checkpoint "$checkpoint"
bash tools/pi05/uf850/run.sh serve --checkpoint "$checkpoint"
# In a second terminal, opens actual policy rollouts in RViz on ROS domain 44:
bash tools/pi05/uf850/run.sh rollout --episodes 3 --seed 101
# Add --headless for evaluation without windows.
```

Evaluation reports held-out action error separately from task success.
Rollout reports actual grasp-and-place results and writes camera videos.
Rollout commands its own PyBullet world, with joint limits and a 0.5 rad/s
motor-target rate cap relative to the previous command, matching the teacher;
it never sends robot commands to a hardware driver.
The PS5 scene on domain 42 remains separate. The server checks/configuration
metadata distinguish the tuned 850 policy from the benchmark policy.

## Second pass: varied starts and deliberate grasps

The first 1,000-update pilot reduced training loss but failed all three
closed-loop trials (seeds 101–103). Its jaws closed before reaching grasp
height. The second dataset adds 20 verified successful demonstrations with
varied empty-gripper starting poses and 20-frame pauses at grasp height.
The original four validation episodes remain excluded from training, along
with four new episodes. The resulting split is 32 training / 8 held-out
episodes. Near-constant joint normalization receives a minimum range so
encoder-scale numerical fluctuations do not dominate discrete state tokens.

For the second pass, select the local dataset and initialize from the first
tuned checkpoint rather than redownloading base weights:

```bash
export UF850_DATA_DIR="$HOME/projects/TactFoundry/data/uf850_sim_v2"
export UF850_RUN_DIR="$HOME/projects/TactFoundry/data/uf850_tuning_v2"
export UF850_REPO_ID=tactfoundry/uf850_bowl_sim_v2
export UF850_BASE_WEIGHTS="$HOME/projects/TactFoundry/data/uf850_tuning_v1/checkpoints/pi05_uf850_lora/pilot1000/999"
# Create a fresh raw directory and copy the existing episode_*.npz and
# attempt_*.json files from uf850_sim_v1 before generating the new episodes.
bash tools/pi05/uf850/run.sh generate --episodes 20 --seed 1001 --varied-start --grasp-dwell 20
# These new held-out seeds succeeded in the verified run; choose existing
# successful episode seeds if your solver produces a different set.
bash tools/pi05/uf850/run.sh convert --heldout-seeds 17,18,19,20,1019,1020,1021,1022 \
  --reuse-repo-id tactfoundry/uf850_bowl_sim_v1 --prior-split data/uf850_sim_v1/split.json
bash tools/pi05/uf850/run.sh stats --steps 500 --state-min-range .02 --action-min-range .002
bash tools/pi05/uf850/run.sh train --steps 500 --exp-name refine500
checkpoint="$UF850_RUN_DIR/checkpoints/pi05_uf850_lora/refine500/499"
bash tools/pi05/uf850/run.sh evaluate --checkpoint "$checkpoint"
bash tools/pi05/uf850/run.sh serve --checkpoint "$checkpoint"
# In a second terminal with the same UF850 environment settings:
bash tools/pi05/uf850/run.sh rollout --episodes 3 --seed 201 --execute-steps 4
```

This pass performs another 500 gradient updates from the first checkpoint.
Reusing local LeRobot datasets copies their existing episodes and appends
new ones; it refuses a split that would move a previous validation episode
into training. Rollout can keep its final scene open with `--hold`.

## Measured pilot results

The completed run used 40 successful physical simulation demonstrations
(32 training / 8 held out) and 1,500 gradient updates: 1,000 from the base
model followed by 500 from the first tuned checkpoint. The final checkpoint
is `data/uf850_tuning_v2/checkpoints/pi05_uf850_lora/refine500/499`.

Held-out mean absolute action error was 0.00446 rad for the arm and 0.00777 rad
for the gripper, versus 0.01331 / 0.06039 rad for holding the measured position.
These action errors do not establish grasp success. Closed-loop results were:

| Policy / execution | Trial seeds | Successful placements |
| --- | --- | --- |
| First 1,000-update pilot, four actions per observation | 101–103 | 0/3 |
| Refined policy, four actions per observation | 101–103 (retested) | 1/3 |
| Refined policy, one action per observation | 201–203 (fresh) | 0/3 |

The refined policy physically grasped, lifted, placed and released the bowl
in trial 102, without attachment constraints or a scripted grasp assistant.
Failures commonly closed the gripper above or beside the bowl and transported
an empty hand. More frequent replanning did not fix this. Keep the default
four-action prefix; the policy remains experimental and unreliable.
The three-case samples and different execution settings are not a benchmark
of generalization. Videos, traces and JSON results are saved locally under
`data/uf850_tuning_v2/rollouts*`; checkpoints and datasets stay outside Git.

With the refined server running, replay the previously successful initial
condition in RViz (a new stochastic rollout can still fail):

```bash
UF850_RUN_DIR="$HOME/projects/TactFoundry/data/uf850_tuning_v2" \
  bash tools/pi05/uf850/run.sh rollout --episodes 1 --seed 102 --execute-steps 4 \
  --output "data/uf850_live_$(date +%Y%m%d_%H%M%S)" --hold
```

A small, fixed-scene pilot can overfit and does not establish broad grasping
ability. Real deployment needs physical demonstrations, measured camera/tool
transforms, matching image preprocessing and action timing, and a real-arm
control interface. The simulation motor effort is not a physical calibration.

Reference: [OpenPI's official fine-tuning guide](https://github.com/Physical-Intelligence/openpi#fine-tuning-base-models-on-your-own-data).
