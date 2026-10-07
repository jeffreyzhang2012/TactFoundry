# Upstream model

Source: https://github.com/xArm-Developer/xarm_ros2.git
Commit: `62936f7ea1846a85f7350de2c4c18f39e6d19715`
License: BSD-3-Clause; original license retained in LICENSE.

Only description macros, required configuration, and mesh assets are retained.
Macros and meshes are unchanged. Package metadata and CMake installation are
adapted for this standalone model subset. Upstream controllers, standalone
launchers, and unrelated robot models are excluded. The xArm subset uses
the nominal legacy xArm6 STL geometry and default kinematics.

Also includes `urdf/camera/realsense_d435i.urdf.xacro` and the visual/collision
`d435_with_cam_stand.stl` meshes from the same pinned commit. This is a combined
D435 housing and UFACTORY camera mounting stand. The upstream macro's legacy
name is d435i, but its frames describe depth, color, and infrared sensors and
do not include an IMU. These are nominal mount/frame offsets, not a measured
hand-eye calibration or a live RealSense driver configuration.
