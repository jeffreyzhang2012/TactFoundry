# Tact Foundry

ROS 2 tactile sensing workspace for NVIDIA Jetson Orin and real hardware.

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
