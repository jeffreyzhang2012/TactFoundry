import numpy as np
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray


def force_markers(readings, stamp, scale=.01):
    message = MarkerArray()
    for side, reading in readings.items():
        arrow = Marker()
        arrow.header.frame_id, arrow.header.stamp = 'world', stamp
        arrow.ns, arrow.id = f'jaw_force_{side}', 0
        arrow.type = Marker.ARROW
        arrow.pose.orientation.w = 1.
        arrow.scale.x, arrow.scale.y, arrow.scale.z = .004, .009, .015
        arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = (
            (0., .9, 1., 1.) if side == 'left' else (1., .45, .1, 1.))
        arrow.lifetime.nanosec = 300000000
        magnitude = float(np.linalg.norm(reading['force']))
        arrow.action = Marker.ADD if magnitude >= .01 else Marker.DELETE
        start = reading['point']
        # Saturate only the drawing length; readings and labels stay in newtons.
        end = start + reading['force'] * min(scale, .25 / max(magnitude, .01))
        arrow.points = [Point(x=float(v[0]), y=float(v[1]), z=float(v[2])) for v in (start, end)]
        label = Marker()
        label.header = arrow.header
        label.ns, label.id = arrow.ns, 1
        label.type, label.action = Marker.TEXT_VIEW_FACING, Marker.ADD
        label.pose.orientation.w = 1.
        label.pose.position = Point(x=float(start[0]), y=float(start[1]), z=float(start[2] + .04))
        label.scale.z = .018
        label.color = arrow.color
        label.lifetime = arrow.lifetime
        label.text = f'{side}: {magnitude:.2f} N'
        message.markers.extend([arrow, label])
    return message
