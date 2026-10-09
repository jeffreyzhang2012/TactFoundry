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

## Other objects and automatic success evaluation

Use the same checkpoint without retraining to test language/object transfer.
`bottle` is the green bottle; `distractor_block` is the red block. The target
stays at its original scene position, with the same ±15 mm XY variation;
it is not moved into the demonstrated bowl location. Other props remain in
the scene. Each trial receives the corresponding object-specific prompt.

```bash
bash tools/pi05/uf850/run.sh rollout --object bottle --episodes 3 --seed 301 \
  --max-steps 320 --headless --ros-domain-id 45 --output data/transfer_green
bash tools/pi05/uf850/run.sh rollout --object distractor_block --episodes 3 --seed 301 \
  --max-steps 320 --headless --ros-domain-id 45 --output data/transfer_red
```

Each output directory contains side-by-side external/wrist camera MP4s,
joint traces and `results.json`. These are policy attempts, including failures.
Playback is at the simulated 20 Hz, independently of inference wall-clock time.
There is no new training, waypoint assist or object attachment in these runs.

The completed transfer test of checkpoint `refine500/499`, seeds 301–303,
with four-action prefixes and 320-step limits scored **0/3 for the green
bottle and 0/3 for the red block**. Neither requested object was grasped or
lifted. The arm repeatedly followed the demonstrated bowl-to-plate route
despite the changed prompts. These six trials do not establish general
object or language-conditioned manipulation. Local combined videos and
the complete JSON summary are in `data/uf850_tuning/v2/transfer/`.

The evaluator now handles different object heights. It checks the **requested
body**, not whichever object the robot happens to touch:

- Bilateral jaw contact while its center rises at least 4 cm from its initial
  settled height.
- Its center is within 25 mm of the plate center, with actual plate contact.
  Bowl and bottle must be upright (local vertical dot world vertical > 0.8);
  the cubic block may rest on any face.
- The commanded gripper is open (<0.1 rad), with no load-bearing robot contact.
- Linear speed <0.01 m/s and angular speed <0.1 rad/s continuously for at
  least the last second of a two-second settling period.

This is stricter than the original bowl-only scoring and uses simulator
ground truth. JSON records each failed condition and the final distance;
new traces include the target position, distance and grasp/lift history.
Report success count / trial count over a fixed set of randomized starts,
alongside videos and failure reasons. Low training loss is not this metric.

For real hardware, simulator body poses and contact flags are unavailable.
Use a calibrated external RGB-D camera to track the object and plate in a
shared frame (tags can simplify an initial evaluation setup), plus jaw force
and gripper state for grasp/release. Check a measured lift, object footprint
inside the plate, clearance after release, and stable pose for a fixed dwell.
Validate the evaluator against human-labeled successes/failures and report
uncertain detections separately. An unvalidated learned success classifier
should not be the only measure of physical task completion.

The current simulation-only checkpoint is not ready to command the real arm.
Collect physical 850 + AG95 demonstrations, calibrate camera/tool frames,
match joint/action conventions and image preprocessing, and validate timing
and the robot control interface before supervised low-speed trials. A camera
extrinsic calibration update alone does not adapt this raw-RGB policy to a
new viewpoint. Small camera shifts may work and should be tested; substantial
changes call for demonstrations from the new view and fine-tuning. Training
across camera poses reduces reliance on one fixed placement.

## Balanced tasks and excluded layouts

`multitask.py` records matched demonstrations for black bowl, green bottle
and red block. Each accepted layout has **three episodes with identical
initial pixels and state, but different target instructions and actions**.
Object identities are permuted among three workspace areas near the original
bowl/bottle/block positions. XY positions and the plate position receive
random jitter. All three objects visit all three slots during training.
Four complete permutations are used for training and two different complete
permutations are reserved for validation and unseen-layout rollouts.
All instructions from one layout stay in the same split. Camera poses,
geometry and lighting remain fixed; this is not a camera-robustness dataset.

The teacher accounts for each object's settled height and jaw closure.
It grasps with physical contact, lifts, places and retreats, with no object
attachment or manipulation-time object reset. Only layouts where **every
target** passes the physical scoring checks are recorded. This selects
teacher-reachable cases, not every possible pose. Each object has three
instruction variants. Balanced sampling helps teach instruction grounding;
it does not guarantee semantic or physical generalization.

The multi-object wrapper selects new directories and initializes from the
previous `refine500/499` checkpoint; it preserves the previous model/data:

```bash
bash tools/pi05/uf850/multi.sh generate-multi --train-layouts 24 --heldout-layouts 6
bash tools/pi05/uf850/multi.sh convert
bash tools/pi05/uf850/multi.sh stats --steps 2000 --state-min-range .02 --action-min-range .002
# Stop the earlier policy server to free GPU memory first.
bash tools/pi05/uf850/multi.sh train --steps 2000 --exp-name multi2000
checkpoint="$HOME/projects/TactFoundry/data/uf850_tuning_multi_v3/checkpoints/pi05_uf850_lora/multi2000/1999"
bash tools/pi05/uf850/multi.sh evaluate --checkpoint "$checkpoint" --samples 96
bash tools/pi05/uf850/multi.sh serve --checkpoint "$checkpoint"
# Second terminal: these seeds are excluded from generation and training.
bash tools/pi05/uf850/multi.sh rollout --object distractor_block --layout heldout \
  --seed 5001 --episodes 3 --max-steps 400 --headless --output data/multi_red_tests
bash tools/pi05/uf850/multi.sh language-probe
```

The completed split has 72 training / 18 validation episodes (24 / 6 complete
layouts), totaling 24,255 / 6,060 frames. The run added 2,000 gradient updates, with batch size 4 and unchanged
LoRA/action conventions. Raw data live in `data/uf850_multi_sim_v3_wide/`,
the LeRobot repo is `tactfoundry/uf850_multi_sim_v3`, and checkpoints/results
live in `data/uf850_tuning_multi_v3/`. None are uploaded to GitHub.

`language-probe` fixes images, state and diffusion noise while changing only
the instruction, then compares the predicted initial reach with the three
teacher branches. Its branch-agreement score is an imitation diagnostic,
not a grasp-success metric. Actual rollouts report which objects were
grasped/lifted, task success, failures, videos and traces. Novel layouts test
spatial transfer with familiar objects. New object geometry, colors, cameras,
backgrounds, tasks and real hardware require separate evaluation and often
additional demonstrations; broad language knowledge alone is not proof of
correct robot control.

## Emphasizing the initial target choice

The first multi-object pass achieved arm MAE 0.00534 rad and gripper MAE
0.01369 rad on 96 sampled validation chunks, but its controlled first-reach
diagnostic matched only **6/18 requested teacher branches**. All three
instructions produced the same nearest branch in each validation scene.
Distinct target prompts were verified at the model's tokenized input; this
was not a missing-text transport bug. Uniform full-trajectory training gave
little weight to the initial instruction-dependent decision.

Completed physical tests of `multi2000/1999` scored **0/3 per target** on
excluded layouts (seeds 5001–5003), and **0/1 per target** in the original
scene (seed 601). The previous bowl-only policy also scored 0/9 on the same
excluded-layout starts. These outcomes are retained, including failure
videos; the multi-object pass has not established task generalization.

`focus_sampling.py` therefore resamples only existing training frames for
an additional pass. It excludes the 12 pre-task idle frames, repeats frame
12 with weight 100 and frames 13–41 with weight 8, and balances those weights
by object/slot frequency. Later frames retain weight 1. The resulting
45,227 virtual sample indices reference the original 24,255 training frames;
action chunks retain their original time context. No demonstrations or
synthetic actions are added. Validation layouts remain excluded.

The completed focused pass has arm MAE **0.00679 rad**, gripper MAE
**0.01347 rad**, and still matches only **6/18** first-reach branches on the
same controlled validation diagnostic. Resampling alone did not establish
instruction grounding. Its loss reached 0.0036 at the last logged update;
that loss is measured under a different sampling distribution from the
uniform pass and should not be treated as a direct comparison.

The focused checkpoint `grounding1000/999` completed the following tests:

| Requested object | Fresh excluded layouts (6001–6003) | Original scene (601) | IK teacher on fresh layouts |
| --- | --- | --- | --- |
| Black bowl | 0/3 | 0/1 | 3/3 |
| Green bottle | 0/3 | 0/1 | 2/3 |
| Red block | 0/3 | 0/1 | 2/3 |

No requested object met the grasp-and-lift criterion in these 12 policy trials.
The IK reference uses ground-truth poses, not policy actions; it is a separate
physical reference and not a success claim for π0.5. All nine predetermined
fresh starts were retained, including the two teacher failures.

A second controlled probe on six **training** layouts also matched only
6/18 instruction branches. Each scene selected the same branch for all
three prompts. Target selection was therefore not learned reliably even
on those training observations; this is not just a novel-layout problem.
The checkpoint remains experimental. More updates alone are not an
established remedy: clearer target views and explicit target-selection
supervision are candidate next investigations, followed by fresh physical
tests and demonstrations with recovery states.

Local Windows reports, JSON/NPZ traces and videos are under
`data/uf850_tuning/multi_v3/` and `data/uf850_tuning/multi_v4/`. The latter
contains `training-results.png`, `training-summary.json`,
`training-language-probe.json`, and combined `unseen_<object>.mp4` files.
The V4 trials used the focused checkpoint server on localhost:8001.
Fourteen relevant tests passed, including physical teacher replay, target
scoring, near-zero-joint IK and split/resampling checks; Python compilation
and shell syntax checks also passed. Weights and raw data remain local.

The focused pass initializes from `multi2000/1999`, reuses its normalization,
and saves independently under `data/uf850_tuning_multi_v4/`. It adds 1,000
updates (4,500 total in checkpoint ancestry). Stop the running policy server
before training or evaluating to free GPU memory:

```bash
bash tools/pi05/uf850/focus.sh train
bash tools/pi05/uf850/focus.sh evaluate
bash tools/pi05/uf850/focus.sh serve
# Another terminal: fresh seeds excluded from demonstrations and earlier rollouts.
bash tools/pi05/uf850/focus.sh rollout --object distractor_block --layout heldout \
  --seed 6001 --episodes 3 --max-steps 400 --headless --output data/focus_red_tests
```

Headless rollouts render the same 224-pixel policy observations and recorded
videos, but skip the extra full-resolution ROS camera streams. This saves
rendering time without changing physics steps or action timing. Add
`--publish-ros-images` if a remote ROS viewer needs those streams. Interactive
RViz rollouts publish them automatically.

`report.py` creates a JSON summary, loss/action-error/physical-success plot,
and one combined video per target from completed evaluations. It expects
`unseen_layouts/<object>/results.json` for all three targets:

```bash
$HOME/projects/openpi/.venv/bin/python tools/pi05/uf850/report.py \
  --root data/uf850_tuning_multi_v4 --source data/uf850_multi_sim_v3_wide \
  --train-log log/uf850-focus-train.log --output data/uf850_tuning/multi_v4 \
  --updates 1000 --ancestry-updates 4500
```

Language pretraining provides semantic priors, but the new camera views,
UF850 joint representation and AG95 grasp mechanics still need grounded
robot demonstrations. The [π0.5 paper](https://arxiv.org/abs/2504.16054)
attributes broad generalization to co-training across robot data, semantic
tasks and web data. This small local fine-tune does not reproduce that
coverage. Its excluded-layout tests measure spatial transfer for three
familiar objects with fixed cameras; they do not establish novel-object,
viewpoint, task or real-hardware generalization.

## Fresh-base diverse run (V6)

V6 restarts from the downloaded pretrained `pi05_base`, with a fresh optimizer.
It does not inherit V1–V4 or the one-scene diagnostic's weights. This is robot
fine-tuning from the pretrained model, rather than random initialization of
the entire foundation model.

The data recipe records 96 training layouts and 24 held-out layouts, with three
physically successful target-specific trajectories per layout: 288 training
and 72 validation demonstrations. Targets span eight colors and four shapes
(bowl, bottle, block and cylinder). Same-color and same-shape scenes require
using both parts of the instruction. Layouts vary object XY by up to 6/8 cm,
plate XY by 3.5 cm, initial arm position, yaw, size, mass, friction and RGB.
Four color/shape combinations are reserved for validation: yellow bottle,
purple block, blue bowl and green cylinder. Camera extrinsics and lighting
remain fixed. A workspace crop increases object visibility; its projection
is saved with the model and reused during inference.

The vision encoder is frozen. Language/action LoRA and action projections are
trained with seven real action axes, excluding padding. Half of training
examples use pure diffusion noise, which carries no action-target hints;
the remainder retain ordinary flow matching. Initial reaches are resampled
and balanced by color, shape and spatial slot. Loss values are consequently
not directly comparable to the older 32-axis objective.

```bash
# From the WSL repository, after building the ROS packages.
bash tools/pi05/uf850/diverse.sh generate-multi --diverse \
  --train-layouts 96 --heldout-layouts 24 --train-seed 10001 --heldout-seed 20001
bash tools/pi05/uf850/diverse.sh convert
mkdir -p data/uf850_tuning_diverse_v6
cp data/uf850_diverse_sim_v6/dataset-profile.json data/uf850_tuning_diverse_v6/
bash tools/pi05/uf850/diverse.sh stats --state-min-range .02 --action-min-range .002 \
  --pure-noise-probability .5 --freeze-vision
bash tools/pi05/uf850/diverse.sh train --steps 4000 --batch-size 4 --exp-name restart4000 \
  --pure-noise-probability .5 --freeze-vision --diverse-sampling \
  --selection-manifest data/uf850_diverse_sim_v6/split.json
```

Weights save under
`data/uf850_tuning_diverse_v6/checkpoints/pi05_uf850_lora/restart4000/3999/`.
Generation, conversion and training do not publish ROS topics. Physical
evaluation uses a separate domain and executes raw policy actions with the
documented joint rate limit; IK is used only as a separate teacher reference.
The held-out instruction probe changes text while keeping images, state and
sampling noise identical. Its branch agreement measures initial reach choice,
not successful manipulation. Independent physical trials score the requested
object's grasp, lift, stable plate support and release, including wrong-object
activity and the first near-object closing attempt.

The completed V6 experiment ran 4,000 updates with batch size 4. It used
102,941 training frames and 25,753 validation frames. Logged training loss
fell from 0.4432 to 0.0072. On 96 validation samples, arm MAE was 0.005984 rad
and gripper MAE 0.017570 rad; holding the measured position gave 0.015015 rad
and 0.063079 rad respectively. The controlled instruction probe agreed with
the requested first-reach teacher branch on **71/72 validation instructions**
and **18/18 instructions from six training layouts**. This is a reach-choice
proxy, not a grasp or placement success score.

| Physical evaluation | Correct first near-object closing attempt | Requested object grasped and lifted | Stable successful placement |
| --- | --- | --- | --- |
| Fresh layouts 30001–30003, three targets each | 9/9 | 5/9 | 2/9 |
| Original scene 601, bowl/bottle/block | 3/3 | 2/3 | 1/3 |

The fresh successes were the purple block and black block. The separate IK
reference succeeded on 7/9 of the same predetermined starts; its two failures
were retained. In the original scene, the black bowl was correctly picked and
placed (5.3 mm from the plate center), addressing the earlier seed-601 wrong
target behavior in this trial. The green bottle was approached correctly but
not grasped. The red block was grasped, lifted and released on the plate,
but its center remained 28.6 mm from the plate center, outside the 25 mm limit.
No waypoint controller or ground-truth target pose modified policy actions.

Target selection is substantially stronger in these tests, while manipulation
accuracy and recovery remain limited. These are small stochastic simulation
tests with fixed cameras, not evidence of real-hardware readiness or arbitrary
camera-placement generalization. Positive teacher demonstrations do not cover
missed grasps, dropped objects or correction after an inaccurate deposit.

All 12 MP4 videos decode successfully. Windows reports, videos and JSON traces
are under `data/uf850_tuning/diverse_v6/`, including `training-results.png` and
`training-summary.json`. The model weights and raw recordings remain in WSL.
Thirteen relevant tests passed, and the full training-loss graph was checked
before the run.

```bash
checkpoint="$PWD/data/uf850_tuning_diverse_v6/checkpoints/pi05_uf850_lora/restart4000/3999"
bash tools/pi05/uf850/diverse.sh evaluate --checkpoint "$checkpoint" --samples 96 \
  --language-probe-source data/uf850_diverse_sim_v6 --training-probe-layouts 6
bash tools/pi05/uf850/diverse.sh serve --checkpoint "$checkpoint"
# Another terminal: live simulated bowl trial on a separate ROS domain.
bash tools/pi05/uf850/diverse.sh rollout --object black_bowl --seed 601 --episodes 1 \
  --max-steps 400 --ros-domain-id 51 --output data/v6_bowl_live
```

Stop the existing policy server before loading another copy for evaluation
or training. The live rollout opens its own RViz on domain 51; the PS5 scene
on domain 42 can remain separate.
