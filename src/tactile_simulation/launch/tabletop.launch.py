"""Open the robot-adapted tabletop task with two camera views."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    return LaunchDescription([IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('tactile_simulation'), 'launch', 'playground.launch.py')),
        launch_arguments={'scene': 'tabletop', 'object_count': '0'}.items())])
