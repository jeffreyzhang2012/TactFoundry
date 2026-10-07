# Tact Foundry

ROS 2 tactile sensing workspace for NVIDIA Jetson Orin and real hardware.

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
