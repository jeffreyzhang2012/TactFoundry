# Upstream model

Source: https://github.com/xArm-Developer/xarm_ros2.git
Commit: `62936f7ea1846a85f7350de2c4c18f39e6d19715`
License: BSD-3-Clause; original license retained in LICENSE.

Only description macros, required configuration, and mesh assets are retained.
Macros and meshes are unchanged. Package metadata and CMake installation are
adapted for this standalone model subset. Upstream controllers, standalone
launchers, and unrelated robot models are excluded. The xArm subset uses
the nominal legacy xArm6 STL geometry and default kinematics.
