import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from tactile_simulation.camera_launch import real_camera
from tactile_simulation.tabletop import HOMES


def nodes(context):
    share = get_package_share_directory('tactile_robot_description')
    settings = {name: LaunchConfiguration(name).perform(context)
                for name in ('robot_model', 'mount_xyz', 'mount_rpy', 'camera_mount_xyz', 'camera_mount_rpy')}
    scene = LaunchConfiguration('scene').perform(context)
    settings['base_xyz'] = '0 0 .75' if scene == 'tabletop' else '0 0 0'
    joint_defaults = {}
    if scene == 'tabletop':
        joint_defaults = {'zeros.' + name: value for name, value in zip(
            [f'joint{i}' for i in range(1, 7)] + ['ag95_left_outer_knuckle_joint'],
            HOMES[settings['robot_model']])}
    description = xacro.process_file(os.path.join(share, 'urdf', 'xarm6_ag95.urdf.xacro'),
                                     mappings=settings).toxml()
    source = LaunchConfiguration('camera_source').perform(context)
    actions = [
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': description}], output='screen'),
        Node(package='joint_state_publisher_gui', executable='joint_state_publisher_gui',
             parameters=[os.path.join(share, 'config', 'joints.yaml'),
                         {'source_list': ['simulation/target_feedback']}, joint_defaults],
             remappings=[('joint_states', 'simulation/joint_commands')],
             condition=IfCondition(LaunchConfiguration('gui'))),
        Node(package='tactile_simulation', executable='playground', output='screen',
             parameters=[{'robot_description': description, 'camera_source': source,
                          'scene': scene, 'robot_model': settings['robot_model'],
                          'gripper_effort': float(LaunchConfiguration('gripper_effort').perform(context)),
                          'object_kind': LaunchConfiguration('object_kind').perform(context),
                          'object_count': int(LaunchConfiguration('object_count').perform(context)),
                          'gamepad': LaunchConfiguration('gamepad').perform(context) == 'true',
                          'stick_plane': LaunchConfiguration('stick_plane').perform(context),
                          'force_arrow_scale': float(LaunchConfiguration('force_arrow_scale').perform(context)),
                          'seed': int(LaunchConfiguration('seed').perform(context))}]),
        Node(package='tactile_simulation', executable='scene_controls',
             condition=IfCondition(LaunchConfiguration('gui'))),
        Node(package='rviz2', executable='rviz2',
             arguments=['-d', os.path.join(share, 'rviz', 'tabletop.rviz' if scene == 'tabletop' else 'xarm6_ag95.rviz')],
             condition=IfCondition(LaunchConfiguration('rviz'))),
    ]
    if LaunchConfiguration('gamepad').perform(context) == 'true':
        actions.append(Node(package='joy', executable='game_controller_node',
                            parameters=[{'device_id': int(LaunchConfiguration('device_id').perform(context)),
                                         'deadzone': .08, 'autorepeat_rate': 50.}], output='screen'))
    if source == 'real':
        actions.append(real_camera())
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('scene', default_value='playground', choices=['playground', 'tabletop']),
        DeclareLaunchArgument('robot_model', default_value='uf850', choices=['uf850', 'xarm6']),
        DeclareLaunchArgument('domain_id', default_value='42',
                              description='Isolate simulation from hardware and display-only nodes'),
        SetEnvironmentVariable('ROS_DOMAIN_ID', LaunchConfiguration('domain_id')),
        SetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE', os.path.join(
            get_package_share_directory('tactile_simulation'), 'config', 'fastdds_udp.xml')),
        DeclareLaunchArgument('camera_source', default_value='sim', choices=['sim', 'real', 'none']),
        DeclareLaunchArgument('object_kind', default_value='mixed'),
        DeclareLaunchArgument('object_count', default_value='8'),
        DeclareLaunchArgument('seed', default_value='42'),
        DeclareLaunchArgument('gui', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('gamepad', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument('device_id', default_value='0'),
        DeclareLaunchArgument('force_arrow_scale', default_value='0.01',
                              description='Force arrow length in meters per newton, capped at 0.25 m'),
        DeclareLaunchArgument('gripper_effort', default_value='50.0',
                              description='Simulation AG95 motor effort cap in Nm; 3.0 matches the VLA teacher profile'),
        DeclareLaunchArgument('stick_plane', default_value='xz', choices=['xz', 'yz', 'xy']),
        DeclareLaunchArgument('rviz', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('mount_xyz', default_value='0 0 0'),
        DeclareLaunchArgument('mount_rpy', default_value='0 0 1.5707963267948966'),
        DeclareLaunchArgument('camera_mount_xyz', default_value='0 0 0'),
        DeclareLaunchArgument('camera_mount_rpy', default_value='0 0 0'),
        OpaqueFunction(function=nodes),
    ])
