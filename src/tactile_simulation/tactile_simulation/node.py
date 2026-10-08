import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, JointState, Joy
from std_srvs.srv import Trigger
from std_msgs.msg import Float32
from geometry_msgs.msg import WrenchStamped, TransformStamped, Point
from tf2_ros import StaticTransformBroadcaster
import pybullet as p
from . import tabletop
from tactile_interfaces.srv import SpawnObjects
from visualization_msgs.msg import Marker, MarkerArray

from .engine import World
from .gamepad import Gamepad
from .force_markers import force_markers


class Playground(Node):
    def __init__(self):
        super().__init__('tactile_playground')
        for name, value in [('robot_description', ''), ('camera_source', 'sim'),
                            ('object_kind', 'mixed'), ('object_count', 8), ('seed', 42),
                            ('scene', 'playground'), ('robot_model', 'uf850'),
                            ('gamepad', False), ('stick_plane', 'xz'), ('force_arrow_scale', .01)]:
            self.declare_parameter(name, value)
        self.world = World(self.get_parameter('robot_description').value,
                           self.get_parameter('scene').value,
                           self.get_parameter('robot_model').value)
        self.external = self.world.scene_kind == 'tabletop'
        if self.external:
            self.camera_tf = StaticTransformBroadcaster(self)
            transform = TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = 'world'
            transform.child_frame_id = 'external_camera_optical_frame'
            xyz, rpy = tabletop.optical_pose()
            transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = xyz
            q = p.getQuaternionFromEuler(rpy)
            transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w = q
            self.camera_tf.sendTransform(transform)
        count = self.get_parameter('object_count').value
        if count:
            self.world.spawn(self.get_parameter('object_kind').value, count,
                             self.get_parameter('seed').value)
        self.states = self.create_publisher(JointState, 'joint_states', 10)
        self.targets = self.create_publisher(JointState, 'simulation/target_feedback', 10)
        self.create_subscription(JointState, 'simulation/joint_commands', self.command, 10)
        self.pad = Gamepad(self.get_parameter('stick_plane').value)
        if self.get_parameter('gamepad').value:
            self.create_subscription(Joy, 'joy', self.joy, qos_profile_sensor_data)
        self.markers = self.create_publisher(MarkerArray, 'scene/objects',
                        QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.force_arrows = self.create_publisher(MarkerArray, 'simulation/jaw_force_markers', 10)
        self.wrenches = {side: self.create_publisher(WrenchStamped,
            f'simulation/jaw_forces/{side}', 10) for side in ('left', 'right')}
        self.normal_loads = {side: self.create_publisher(Float32,
            f'simulation/jaw_forces/{side}/normal_load', 10) for side in ('left', 'right')}
        self.images, self.infos = {}, {}
        for sensor in ('color', 'depth', 'external', 'external_depth'):
            self.images[sensor] = self.create_publisher(Image, f'camera/{sensor}/image_raw',
                                                       qos_profile_sensor_data)
            self.infos[sensor] = self.create_publisher(CameraInfo, f'camera/{sensor}/camera_info',
                                                      qos_profile_sensor_data)
        self.create_service(SpawnObjects, 'scene/spawn_objects', self.spawn)
        self.create_service(Trigger, 'scene/clear_objects', self.clear)
        self.create_service(Trigger, 'scene/reset_arm', self.reset)
        self.create_service(Trigger, 'scene/reset_layout', self.reset_layout)
        self.create_timer(1/60, self.physics)
        self.create_timer(0.2, self.scene)
        if self.get_parameter('camera_source').value == 'sim' or self.external:
            self.create_timer(0.1, self.camera)
        self.get_logger().info('Playground ready: scene services and physical joint-state feedback')

    def command(self, msg):
        if not self.get_parameter('gamepad').value and len(msg.name) == len(msg.position):
            self.world.command(msg.name, msg.position)

    def joy(self, msg):
        self.pad.update(msg.axes, msg.buttons, time.monotonic())

    def physics(self):
        if self.get_parameter('gamepad').value:
            twist, grip = self.pad.command(time.monotonic())
            if any(twist) or grip:
                self.world.cartesian_command(twist, grip, 1/60,
                                             frame='d435_camera_color_optical_frame')
            target = JointState()
            target.name, target.position = list(self.world.targets), list(self.world.targets.values())
            self.targets.publish(target)
        self.world.step()
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name, msg.position = self.world.joint_states()
        self.states.publish(msg)
        self.publish_forces(msg.header.stamp)

    def publish_forces(self, stamp):
        readings = self.world.jaw_forces()
        for side, reading in readings.items():
            msg = WrenchStamped()
            msg.header.frame_id, msg.header.stamp = reading['frame'], stamp
            msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z = map(float, reading['local_force'])
            msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z = map(float, reading['local_torque'])
            self.wrenches[side].publish(msg)
            self.normal_loads[side].publish(Float32(data=reading['normal_load']))
        self.force_arrows.publish(force_markers(readings, stamp,
            self.get_parameter('force_arrow_scale').value))

    def spawn(self, req, res):
        try:
            self.world.spawn(req.kind, req.count, req.seed)
            res.success, res.message = True, f'Added {req.count} {req.kind} objects'
        except ValueError as exc:
            res.success, res.message = False, str(exc)
        self.scene()
        return res

    def clear(self, req, res):
        self.world.clear()
        self.scene()
        res.success, res.message = True, 'Objects cleared'
        return res

    def reset_layout(self, req, res):
        self.world.reset_scene()
        count = self.get_parameter('object_count').value
        if self.world.scene_kind == 'playground' and count:
            self.world.spawn(self.get_parameter('object_kind').value, count,
                             self.get_parameter('seed').value)
        self.scene()
        res.success, res.message = True, 'Original scene layout restored'
        return res

    def reset(self, req, res):
        self.pad.received = -float('inf')
        self.world.reset_arm()
        targets = JointState()
        targets.name = list(self.world.targets)
        targets.position = list(self.world.targets.values())
        self.targets.publish(targets)
        res.success, res.message = True, 'Arm returned to the home pose'
        return res

    def marker(self, mid, shape, size, xyz, quat, color):
        marker = Marker()
        marker.header.frame_id = 'world'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns, marker.id = 'playground', mid
        marker.type = {'box': Marker.CUBE, 'sphere': Marker.SPHERE,
                       'cylinder': Marker.CYLINDER}[shape]
        marker.action = Marker.ADD
        marker.scale.x, marker.scale.y, marker.scale.z = size
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = map(float, xyz)
        marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z, marker.pose.orientation.w = map(float, quat)
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        return marker

    def scene(self):
        message = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        message.markers = [clear]
        for mid, (size, xyz, color) in enumerate(self.world.fixtures):
            message.markers.append(self.marker(-mid-1, 'box', size, xyz,
                                                (0., 0., 0., 1.), color))
        if self.external:
            xyz, rpy = tabletop.optical_pose()
            message.markers.append(self.marker(-100, 'box', (.08, .045, .035), xyz,
                p.getQuaternionFromEuler(rpy), (.12, .16, .23, 1.)))
            frustum = Marker()
            frustum.header.frame_id = 'external_camera_optical_frame'
            frustum.header.stamp = self.get_clock().now().to_msg()
            frustum.ns, frustum.id = 'external_camera', 0
            frustum.type, frustum.action = Marker.LINE_LIST, Marker.ADD
            frustum.pose.orientation.w = 1.
            frustum.scale.x = .002
            frustum.color.r, frustum.color.g, frustum.color.b, frustum.color.a = (0., .75, 1., .8)
            corners = [Point(x=x, y=y, z=.25) for x, y in
                       [(-.185, -.139), (.185, -.139), (.185, .139), (-.185, .139)]]
            for i, corner in enumerate(corners):
                frustum.points.extend([Point(), corner, corner, corners[(i+1) % 4]])
            message.markers.append(frustum)
        for mid, part, color, (xyz, quat) in self.world.part_poses():
            message.markers.append(self.marker(mid, part['shape'], part['size'], xyz, quat, color))
        self.markers.publish(message)

    def camera(self):
        stamp = self.get_clock().now().to_msg()
        sensors = ['color', 'depth'] if self.get_parameter('camera_source').value == 'sim' else []
        if self.external:
            sensors += ['external', 'external_depth']
        external_pixels = self.world.render_external() if self.external else None
        for sensor in sensors:
            if sensor.startswith('external'):
                frame = 'external_camera_optical_frame'
                rgb, depth, focal = external_pixels
            else:
                frame = f'd435_camera_{sensor}_optical_frame'
                rgb, depth, focal = self.world.render(frame)
            is_color = sensor in ('color', 'external')
            pixels = rgb if is_color else depth
            msg = Image()
            msg.header.stamp, msg.header.frame_id = stamp, frame
            msg.height, msg.width = pixels.shape[:2]
            msg.encoding = 'rgb8' if is_color else '32FC1'
            msg.is_bigendian = 0
            msg.step = msg.width * (3 if is_color else 4)
            msg.data = pixels.tobytes()
            info = CameraInfo()
            info.header = msg.header
            info.height, info.width = msg.height, msg.width
            info.distortion_model = 'plumb_bob'
            info.d = [0.]*5
            cx, cy = msg.width/2, msg.height/2
            info.k = [focal, 0., cx, 0., focal, cy, 0., 0., 1.]
            info.r = [1., 0., 0., 0., 1., 0., 0., 0., 1.]
            info.p = [focal, 0., cx, 0., 0., focal, cy, 0., 0., 0., 1., 0.]
            self.images[sensor].publish(msg)
            self.infos[sensor].publish(info)

    def destroy_node(self):
        self.world.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = Playground()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
