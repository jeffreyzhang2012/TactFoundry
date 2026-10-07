import time

import rclpy
import serial
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from tactile_interfaces.msg import TactileFrame

from .protocol import parse_frame


class SerialDriver(Node):
    def __init__(self):
        super().__init__('tactile_driver')
        defaults = {'port': '/dev/serial/by-id/YOUR_SENSOR', 'baud_rate': 115200,
                    'rows': 2, 'columns': 2, 'sensor_id': 'tactile_0',
                    'frame_id': 'tactile_link', 'units': 'raw', 'stale_timeout': 1.0}
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.settings = {name: self.get_parameter(name).value for name in defaults}
        if (self.settings['rows'] <= 0 or self.settings['columns'] <= 0
                or self.settings['baud_rate'] <= 0
                or self.settings['stale_timeout'] <= 0):
            raise ValueError('dimensions, baud rate and stale timeout must be positive')
        self.samples = self.create_publisher(TactileFrame, 'tactile/samples',
                                             qos_profile_sensor_data)
        self.health = self.create_publisher(DiagnosticArray, 'diagnostics', 10)
        self.device = None
        self.buffer = bytearray()
        self.discarding = False
        self.last_sample = None
        self.next_connect = 0.0
        self.rejected = 0
        self.error = 'waiting for device'
        self.create_timer(0.005, self.poll)
        self.create_timer(1.0, self.diagnostics)

    def poll(self):
        now = time.monotonic()
        try:
            if self.device is None:
                if now < self.next_connect:
                    return
                self.next_connect = now + 2.0
                self.device = serial.Serial(self.settings['port'],
                                            self.settings['baud_rate'], timeout=0,
                                            exclusive=True)
                self.buffer.clear()
                self.discarding = False
                self.last_sample = None
                self.error = 'waiting for first sample'
                self.get_logger().info('Serial device connected')
            # Bound work per callback so diagnostics and shutdown remain responsive.
            for byte in self.device.read(min(self.device.in_waiting, 16384)):
                if byte == 10:
                    if not self.discarding:
                        self.publish_frame(bytes(self.buffer))
                    self.buffer.clear()
                    self.discarding = False
                elif not self.discarding:
                    self.buffer.append(byte)
                    if len(self.buffer) > 8192:
                        self.buffer.clear()
                        self.discarding = True
                        self.rejected += 1
        except (serial.SerialException, OSError) as exc:
            self.error = str(exc)
            if self.device is not None:
                self.device.close()
            self.device = None
            self.last_sample = None
            self.next_connect = now + 2.0

    def publish_frame(self, line):
        try:
            values = parse_frame(line, self.settings['rows'] * self.settings['columns'])
        except (ValueError, UnicodeError, OverflowError):
            self.rejected += 1
            return
        frame = TactileFrame()
        frame.header.stamp = self.get_clock().now().to_msg()
        frame.header.frame_id = self.settings['frame_id']
        frame.sensor_id = self.settings['sensor_id']
        frame.rows = self.settings['rows']
        frame.columns = self.settings['columns']
        frame.units = self.settings['units']
        frame.values = values
        self.samples.publish(frame)
        self.last_sample = time.monotonic()

    def diagnostics(self):
        status = DiagnosticStatus()
        status.name = 'tactile/serial'
        status.hardware_id = self.settings['sensor_id']
        age = None if self.last_sample is None else time.monotonic() - self.last_sample
        if self.device is None:
            status.level, status.message = DiagnosticStatus.ERROR, self.error
        elif age is None or age > self.settings['stale_timeout']:
            status.level, status.message = DiagnosticStatus.WARN, 'no fresh tactile sample'
        else:
            status.level, status.message = DiagnosticStatus.OK, 'receiving tactile samples'
        status.values = [KeyValue(key='port', value=self.settings['port']),
                         KeyValue(key='rejected_frames', value=str(self.rejected)),
                         KeyValue(key='sample_age_seconds', value=str(age))]
        report = DiagnosticArray()
        report.header.stamp = self.get_clock().now().to_msg()
        report.status = [status]
        self.health.publish(report)

    def destroy_node(self):
        if self.device is not None:
            self.device.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SerialDriver()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
