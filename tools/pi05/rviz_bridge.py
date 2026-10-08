"""ROS 2 receiver for the live LIBERO state; no model or robot control in this process."""
import base64
import json
import socket
import threading
import numpy as np
from scipy.spatial.transform import Rotation
from PyQt5.QtGui import QImage
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster


class Bridge(Node):
    def __init__(self):
        super().__init__('pi05_libero_live')
        self.declare_parameter('port', 8766)
        self.latest = None
        self.geom_ids = set()
        self.tf = StaticTransformBroadcaster(self)
        transform = TransformStamped()
        transform.header.frame_id, transform.child_frame_id = 'world', 'libero_scene_origin'
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.transform.rotation.w = 1.
        self.tf.sendTransform(transform)
        self.scene = self.create_publisher(MarkerArray, 'pi05/scene',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.cameras = {name: self.create_publisher(Image, f'pi05/{name}/image_raw', 1)
                        for name in ('image', 'wrist')}
        self.server = socket.socket()
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(('127.0.0.1', self.get_parameter('port').value))
        self.server.listen(1)
        threading.Thread(target=self.receive, daemon=True).start()
        self.create_timer(.05, self.publish)

    def receive(self):
        while rclpy.ok():
            connection, _ = self.server.accept()
            with connection, connection.makefile('rb') as stream:
                for line in stream:
                    self.latest = json.loads(line)

    def publish(self):
        frame = self.latest
        if frame is None:
            return
        stamp = self.get_clock().now().to_msg()
        message = MarkerArray()
        current_ids = {geom['id'] for geom in frame['geometries']}
        for old_id in self.geom_ids - current_ids:
            deleted = Marker()
            deleted.ns, deleted.id, deleted.action = 'libero', old_id, Marker.DELETE
            message.markers.append(deleted)
        self.geom_ids = current_ids
        for geom in frame['geometries']:
            marker = Marker()
            marker.header.frame_id, marker.header.stamp = 'world', stamp
            marker.ns, marker.id = 'libero', geom['id']
            marker.action = Marker.ADD
            marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = geom['xyz']
            quat = Rotation.from_matrix(geom['rotation']).as_quat()
            marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z, marker.pose.orientation.w = map(float, quat)
            kind, size = geom['kind'], geom['size']
            marker.type = {0: Marker.CUBE, 2: Marker.SPHERE, 3: Marker.CYLINDER,
                           4: Marker.SPHERE, 5: Marker.CYLINDER, 6: Marker.CUBE,
                           7: Marker.MESH_RESOURCE}[kind]
            scale = {0: [4., 4., .005], 2: [2*size[0]]*3,
                     3: [2*size[0], 2*size[0], 2*(size[1]+size[0])],
                     4: [2*v for v in size], 5: [2*size[0], 2*size[0], 2*size[1]],
                     6: [2*v for v in size], 7: [1., 1., 1.]}[kind]
            marker.scale.x, marker.scale.y, marker.scale.z = scale
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = geom['color']
            if kind == 7:
                marker.mesh_resource = geom['mesh']
            message.markers.append(marker)
        label = Marker()
        label.header.frame_id, label.header.stamp = 'world', stamp
        label.ns, label.id, label.type = 'policy_status', 0, Marker.TEXT_VIEW_FACING
        label.pose.position.z, label.pose.orientation.w = 1.7, 1.
        label.scale.z = .035
        label.color.r = label.color.g = label.color.b = label.color.a = 1.
        status = 'SUCCESS' if frame['status'] == 'SUCCESS' else 'LIVE'
        label.text = f'pi0.5 {status}\nstep {frame["step"]}'
        message.markers.append(label)
        self.scene.publish(message)
        for name, publisher in self.cameras.items():
            decoded = QImage.fromData(base64.b64decode(frame[name])).convertToFormat(QImage.Format_RGB888)
            raw = decoded.bits()
            raw.setsize(decoded.byteCount())
            msg = Image()
            msg.header.frame_id, msg.header.stamp = 'world', stamp
            msg.width, msg.height = decoded.width(), decoded.height()
            msg.encoding, msg.step, msg.data = 'rgb8', decoded.bytesPerLine(), bytes(raw)
            publisher.publish(msg)


rclpy.init()
node = Bridge()
try:
    rclpy.spin(node)
except KeyboardInterrupt:
    pass
finally:
    node.server.close()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
