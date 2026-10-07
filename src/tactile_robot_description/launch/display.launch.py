import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_nodes(context):
    share = get_package_share_directory('tactile_robot_description')
    description = xacro.process_file(
        os.path.join(share, 'urdf', 'xarm6_ag95.urdf.xacro'),
        mappings={name: LaunchConfiguration(name).perform(context)
                  for name in ('mount_xyz', 'mount_rpy', 'add_camera',
                               'camera_mount_xyz', 'camera_mount_rpy')},
    ).toxml()
    gui = LaunchConfiguration('gui').perform(context).lower() == 'true'
    return [
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': description}], output='screen'),
        Node(package='joint_state_publisher_gui' if gui else 'joint_state_publisher',
             executable='joint_state_publisher_gui' if gui else 'joint_state_publisher',
             parameters=[os.path.join(share, 'config', 'joints.yaml'),
                         {'robot_description': description}], output='screen'),
        Node(package='rviz2', executable='rviz2', output='screen',
             arguments=['-d', os.path.join(share, 'rviz', 'xarm6_ag95.rviz')],
             condition=IfCondition(LaunchConfiguration('rviz'))),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('rviz', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('mount_xyz', default_value='0 0 0',
                              description='Flange-to-gripper translation in meters'),
        DeclareLaunchArgument('mount_rpy', default_value='0 0 0',
                              description='Flange-to-gripper rotation in radians'),
        DeclareLaunchArgument('add_camera', default_value='true', choices=['true', 'false'],
                              description='Attach the D435 camera and mounting kit'),
        DeclareLaunchArgument('camera_mount_xyz', default_value='0 0 0',
                              description='Flange-to-camera-kit translation in meters'),
        DeclareLaunchArgument('camera_mount_rpy', default_value='0 0 0',
                              description='Flange-to-camera-kit rotation in radians'),
        OpaqueFunction(function=launch_nodes),
    ])
