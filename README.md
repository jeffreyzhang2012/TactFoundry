# Tact Foundry

ROS 2 tactile sensing workspace for NVIDIA Jetson Orin and real hardware.

## Choose the arm

The default arm is **UFACTORY 850** (`uf850`). Select either arm at launch with
`robot_model:=uf850` or `robot_model:=xarm6`. Both use the AG95 gripper rotated
90 degrees, the wrist D435 kit, simulated camera/objects, and PS5 controls.

```bash
# Default: UFACTORY 850
ros2 launch tactile_simulation playground.launch.py gamepad:=true
# xArm6
ros2 launch tactile_simulation playground.launch.py robot_model:=xarm6 gamepad:=true
# RViz model display or real-camera view also accept the selector
ros2 launch tactile_robot_description display.launch.py robot_model:=uf850
ros2 launch tactile_simulation camera_view.launch.py robot_model:=xarm6
```

Restart the launch to change arms. The filename `xarm6_ag95.urdf.xacro` is retained
for compatibility; its `robot_model` argument now defaults to `uf850`. Arm models
come from the pinned official UFACTORY descriptions. Camera and gripper mount
offsets remain nominal and need verification on the physical 850 assembly.

## Camera and grasping playground

The WSL playground provides a CPU-rendered wrist camera, a low tabletop,
gravity/contact physics, and an object-control window. The joint sliders
command simulation motors; RViz receives the simulated joint positions.
This mode uses PyBullet 3.2.7 and runs without a dedicated rendering GPU.

```bash
source /opt/ros/humble/setup.bash
cd ~/projects/TactFoundry
python3 -m pip install --user -r src/tactile_simulation/requirements.txt
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
ros2 launch tactile_simulation playground.launch.py
```

The **Grasp objects** window lets you choose cubes, boxes, spheres, cylinders,
capsules, bottles, mugs, bowls, or a mixed batch. Select a quantity and seed,
then add objects, clear them, generate a new mixed scene, or reset the arm.
The scene supports up to 48 objects at a time. Mugs and bowls use hollow
compound collision shapes; these are procedural approximations, not scanned
household models. Move the arm with its six sliders, and close the AG-95
with its gripper slider to explore object contact. Successful holding depends
on alignment and simulated contact forces; there is no automatic grasp planner.

Examples:

```bash
ros2 launch tactile_simulation playground.launch.py object_kind:=mixed object_count:=24 seed:=7
ros2 launch tactile_simulation playground.launch.py object_kind:=mug object_count:=6
ros2 launch tactile_simulation playground.launch.py object_count:=0
```

Simulation uses ROS domain **42** by default to isolate it from real hardware
and display-only nodes. For CLI tools in another terminal:

```bash
export ROS_DOMAIN_ID=42
ros2 topic hz /camera/color/image_raw
ros2 service call /scene/spawn_objects tactile_interfaces/srv/SpawnObjects \
  "{kind: 'bottle', count: 5, seed: 10}"
ros2 service call /scene/clear_objects std_srvs/srv/Trigger '{}'
```

Use `domain_id:=N` to change the simulation domain, `gui:=false rviz:=false`
for headless mode, or `camera_source:=none` to disable camera rendering.
RViz has color and depth image panels. Simulation publishes 320x240 RGB
(`rgb8`), idealized depth (`32FC1`, meters), and matching CameraInfo at a
target 10 Hz; achieved rate depends on CPU load. Camera pose comes from the
moving wrist's optical frame. This models geometry and occlusion, not D435
stereo noise, exposure, distortion, or a calibrated depth sensor.

### Real D435 view

On a machine with a connected D435 and ROS USB access:

```bash
sudo apt install ros-humble-realsense2-camera
ros2 launch tactile_simulation camera_view.launch.py
```

This opens the existing robot display and starts the RealSense driver. It
uses the same `/camera/color/image_raw`, `/camera/depth/image_raw`, and
`/camera/{color,depth}/camera_info` interface as simulation. Real depth images
may use a different encoding/unit than simulated depth; inspect the message
and CameraInfo before processing them. Driver TF publishing is disabled
because the robot description owns the nominal camera frames. Calibrate the
mount/extrinsics before using them for real robot motion. The real camera
mode does not command the physical arm or gripper. Live USB streaming must
be tested with the actual camera; it was not connected during development.

WSL USB devices require explicit USB passthrough; the physical D435 is
usually easier to run directly on the Jetson. `camera_source:=real` can also
replace the simulated camera feed inside the playground, but the physical
camera pose then does not follow the simulated robot.

### Simulation environment choice

- **PyBullet**: included here for fast WSL prototypes, contact experiments,
  and CPU camera rendering. The AG-95's mimic joints use individual simulated
  motors; the physical closed-loop linkage, calibrated actuators, self-collision
  checks, tactile deformation, and force sensing are not implemented.
- **Gazebo Fortress**: the official pairing for the project's ROS 2 Humble /
  Ubuntu 22.04 stack, suitable for a future ros2_control/MoveIt integration.
  For a new Ubuntu 24.04 / ROS 2 Jazzy project, use Gazebo Harmonic instead.
  See the [official ROS/Gazebo compatibility guide](https://gazebosim.org/docs/fortress/ros_installation/).
- **Isaac Sim**: consider it for richer rendering and GPU training on a
  compatible workstation. Verify its [GPU and system requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)
  before choosing it. Run heavy simulation on the workstation and use the
  Jetson for the real robot stack.

Gazebo and Isaac Sim integrations are not included in this playground.

## xArm6 + AG-95 in RViz

The repo includes pinned xArm6 and DH AG-95 description macros and STL meshes.
The AG-95 is mounted at the xArm tool flange (`link_eef`). Six arm sliders and
one gripper slider move the combined model; the other finger joints follow
the gripper's URDF mimic relationships.

A RealSense D435 and UFACTORY camera mounting kit are attached to the wrist
by default. The combined camera/stand mesh and nominal sensor frames come
from the same pinned UFACTORY source as the arm. The assembly moves with
the wrist; the gripper retains its independent open/close control.

In WSL Ubuntu 22.04 or on the Jetson:

```bash
source /opt/ros/humble/setup.bash
cd ~/projects/TactFoundry
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
ros2 launch tactile_robot_description display.launch.py
```

Move `joint1` through `joint6` in the Joint State Publisher window. Move
`ag95_left_outer_knuckle_joint` from 0 (open) toward 0.93 rad (closed) to move
both fingers. RViz displays their movement together on the arm. The sliders
publish visualization joint states; connecting a real controller is a separate
integration step. This launch uses the nominal upstream xArm6 geometry and
kinematics, without per-unit calibration or an actuator driver.

The default gripper mount has zero translation and a +90-degree rotation
around the flange Z axis. Set the measured adapter
translation in meters and rotation in radians if your physical mount differs:

```bash
ros2 launch tactile_robot_description display.launch.py \
  mount_xyz:="0 0 0.02" mount_rpy:="0 0 1.5708"
```

Those values are an example, not measurements of your hardware. To run the
joint-state/TF publishers without windows, use `gui:=false rviz:=false`.
The camera kit has its own `camera_mount_xyz` and `camera_mount_rpy` arguments
(meters/radians relative to `link_eef`); use `add_camera:=false` to hide it.
The frame chain is `link_eef -> camera_mount_link -> d435_link_eef ->
d435_camera_link`, with depth/color/left/right infrared optical frames below
it. The nominal camera offsets should be calibrated to your actual mount
before using camera data for robot motion. This display launch adds the
camera model and TF frames; it does not start image acquisition. The upstream
camera macro's legacy name contains `d435i`, but this assembly uses the
`d435_with_cam_stand` mesh and does not add IMU frames.
Windows graphics use WSLg; if RViz has graphics-driver problems, try
`LIBGL_ALWAYS_SOFTWARE=1 ros2 launch tactile_robot_description display.launch.py`.

Model provenance and retained licenses are in
`src/vendor/xarm_description/UPSTREAM.md` (UFACTORY, BSD-3-Clause) and
`src/vendor/dh_ag95_description/UPSTREAM.md` (AG-95 ROS 2 model, Apache-2.0).

## Platform

Target: JetPack 6, Ubuntu 22.04 ARM64, ROS 2 Humble, Python 3.10.
JetPack 5 / Ubuntu 20.04 requires an OS upgrade or a separately validated container setup.
Compatibility references: [NVIDIA JetPack](https://docs.nvidia.com/jetson/jetpack/6.0/introduction/index.html), [ROS 2 Humble installation](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html).

## Install and run on the Jetson

Install ROS 2 Humble using the official instructions above, then:

```bash
sudo apt install python3-colcon-common-extensions python3-rosdep python3-serial
# Run sudo rosdep init once if rosdep has not been initialized.
rosdep update
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
sudo usermod -aG dialout "$USER"
# Log out and back in for the group change to take effect.
ros2 launch tactile_driver hardware.launch.py port:=/dev/serial/by-id/YOUR_SENSOR
```

Use a persistent /dev/serial/by-id path for deployed hardware. Adjust
`src/tactile_driver/config/hardware.yaml` for baud rate, grid dimensions,
sensor ID, frame, units, and stale-data timeout.

## Hardware contract

The initial adapter reads USB/UART serial at 115200 baud. This is an explicit
integration contract, not a claim of compatibility with a specific sensor.
Firmware must send one UTF-8 JSON object per newline, for example:

```json
{"values": [1.0, 2.0, 3.0, 4.0]}
```

Values are a row-major grid of calibrated taxels. The default grid is 2 x 2
and units are `raw`; change both to match your device. Samples must contain
exactly rows * columns finite numeric values. Invalid or oversized frames
are rejected. No synthetic readings are published on hardware failure.
The host reception timestamp is used; sensor clock synchronization is not
implemented. This is sensing software, not an actuator controller or a
hard real-time safety system.

Topics:

- `/tactile/samples`: `tactile_interfaces/msg/TactileFrame`, sensor-data QoS.
- `/diagnostics`: device connection, stale-data and rejected-frame status.

The driver reconnects after serial failures and reports stale input. A
different serial protocol, CAN device, SPI/I2C sensor, or camera sensor needs
its own adapter once the hardware model and protocol are known.

## Verify

```bash
colcon test
colcon test-result --verbose
ros2 topic echo /diagnostics
ros2 topic echo /tactile/samples --qos-reliability best_effort
ros2 topic hz /tactile/samples
ros2 bag record /tactile/samples /diagnostics
```

Before deployment, verify device identity, calibration, taxel ordering,
sample rate, unplug/reconnect behavior, and timestamps on the actual Jetson.
CI builds and tests the workspace on Ubuntu; physical hardware verification
must be performed separately.
## PS5 end-effector control

Connect a DualSense controller to Ubuntu, then launch:

```bash
ros2 launch tactile_simulation playground.launch.py gamepad:=true
```

Hold **L1** while moving. Left stick left/right moves along world X;
left stick forward/back moves along world Y (forward is positive Y). D-pad
up/down moves along world Z (height). Right stick up/down controls pitch,
left/right controls yaw; R2 rolls positive and L2 rolls negative. These rotation
rates use fixed world axes. Cross closes the gripper; Circle opens it.
This is the default `stick_plane:=xy` mapping. For the earlier Y/Z stick mapping,
add `stick_plane:=yz`. Select another controller with `device_id:=1`.

Controller mode disables slider commands so they cannot fight the gamepad; sliders
show commanded positions. Motion uses damped differential inverse kinematics at a
nominal gripper center 150 mm from its base, with joint limits, a 0.5 rad/s joint
speed cap, 0.35 m/s translation and 0.4 rad/s rotation. Releasing L1 or receiving
no valid input for 250 ms holds the current target. Reset clears pending input.
This controls the physics playground; a real-arm Cartesian driver is not connected.

On Windows/WSL, attach the USB controller to WSL using USB/IP (Bluetooth pairing
to Windows alone does not expose it to Linux). From an administrator PowerShell
after installing usbipd-win, run `usbipd list`, `usbipd bind --busid <BUSID>`, then
`usbipd attach --wsl --busid <BUSID>`. Confirm `/dev/input` exists in Ubuntu and
use `ros2 run joy joy_enumerate_devices` to check detection. Ubuntu/Jetson can use
USB directly or Bluetooth paired in Ubuntu. ROS dependencies include `ros-humble-joy`.
