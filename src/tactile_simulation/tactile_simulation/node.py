import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_srvs.srv import Trigger
from tactile_interfaces.srv import SpawnObjects
from visualization_msgs.msg import Marker, MarkerArray

from .engine import World


class Playground(Node):
    def __init__(self):
        super().__init__('tactile_playground')
        for name, value in [('robot_description', ''), ('camera_source', 'sim'),
                            ('object_kind', 'mixed'), ('object_count', 8), ('seed', 42)]:
            self.declare_parameter(name, value)
        self.world = World(self.get_parameter('robot_description').value)
        count = self.get_parameter('object_count').value
        if count:
            self.world.spawn(self.get_parameter('object_kind').value, count,
                             self.get_parameter('seed').value)
        self.states = self.create_publisher(JointState, 'joint_states', 10)
        self.targets = self.create_publisher(JointState, 'simulation/target_feedback', 10)
        self.create_subscription(JointState, 'simulation/joint_commands', self.command, 10)
        self.markers = self.create_publisher(MarkerArray, 'scene/objects',
                        QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.images, self.infos = {}, {}
        for sensor in ('color', 'depth'):
            self.images[sensor] = self.create_publisher(Image, f'camera/{sensor}/image_raw',
                                                       qos_profile_sensor_data)
            self.infos[sensor] = self.create_publisher(CameraInfo, f'camera/{sensor}/camera_info',
                                                      qos_profile_sensor_data)
        self.create_service(SpawnObjects, 'scene/spawn_objects', self.spawn)
        self.create_service(Trigger, 'scene/clear_objects', self.clear)
        self.create_service(Trigger, 'scene/reset_arm', self.reset)
        self.create_timer(1/60, self.physics)
        self.create_timer(0.2, self.scene)
        if self.get_parameter('camera_source').value == 'sim':
            self.create_timer(0.1, self.camera)
        self.get_logger().info('Playground ready: scene services and physical joint-state feedback')

    def command(self, msg):
        if len(msg.name) == len(msg.position):
            self.world.command(msg.name, msg.position)

    def physics(self):
        self.world.step()
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name, msg.position = self.world.joint_states()
        self.states.publish(msg)

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

    def reset(self, req, res):
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
        message.markers = [clear,
            self.marker(0, 'box', (0.64, 0.5, 0.02), (0.415, 0., 0.05),
                        (0., 0., 0., 1.), (0.55, 0.42, 0.28, 1.))]
        for mid, part, color, (xyz, quat) in self.world.part_poses():
            message.markers.append(self.marker(mid, part['shape'], part['size'], xyz, quat, color))
        self.markers.publish(message)

    def camera(self):
        stamp = self.get_clock().now().to_msg()
        for sensor in ('color', 'depth'):
            frame = f'd435_camera_{sensor}_optical_frame'
            rgb, depth, focal = self.world.render(frame)
            pixels = rgb if sensor == 'color' else depth
            msg = Image()
            msg.header.stamp, msg.header.frame_id = stamp, frame
            msg.height, msg.width = pixels.shape[:2]
            msg.encoding = 'rgb8' if sensor == 'color' else '32FC1'
            msg.is_bigendian = 0
            msg.step = msg.width * (3 if sensor == 'color' else 4)
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
