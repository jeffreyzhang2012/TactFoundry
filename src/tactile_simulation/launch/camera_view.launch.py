import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from tactile_simulation.camera_launch import real_camera


def generate_launch_description():
    display = os.path.join(get_package_share_directory('tactile_robot_description'),
                           'launch', 'display.launch.py')
    return LaunchDescription([
        IncludeLaunchDescription(PythonLaunchDescriptionSource(display)),
        real_camera(),
    ])
