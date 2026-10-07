"""Optional hardware camera node with a consistent RViz topic interface."""
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node


def real_camera():
    # Fail clearly only when hardware mode is requested, not for simulation.
    get_package_share_directory('realsense2_camera')
    remaps = []
    for sensor, image in [('color', 'image_raw'), ('depth', 'image_rect_raw')]:
        for original, target in [(image, 'image_raw'), ('camera_info', 'camera_info')]:
            remaps.append((f'/camera/d435_camera/{sensor}/{original}',
                           f'/camera/{sensor}/{target}'))
    return Node(package='realsense2_camera', executable='realsense2_camera_node',
                namespace='camera', name='d435_camera', output='screen',
                parameters=[{'camera_name': 'd435_camera', 'publish_tf': False,
                             'enable_color': True, 'enable_depth': True,
                             'rgb_camera.color_profile': '640x480x30',
                             'depth_module.depth_profile': '640x480x30'}],
                remappings=remaps)
