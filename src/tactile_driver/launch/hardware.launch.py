import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(get_package_share_directory('tactile_driver'),
                          'config', 'hardware.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('port', default_value='/dev/serial/by-id/YOUR_SENSOR'),
        Node(package='tactile_driver', executable='serial_driver',
             name='tactile_driver', output='screen',
             parameters=[config, {'port': LaunchConfiguration('port')}]),
    ])
