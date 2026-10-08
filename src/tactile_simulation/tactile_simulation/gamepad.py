"""SDL game-controller mapping and stale-input protection, independent of ROS."""
import math


class Gamepad:
    def __init__(self, plane='xz'):
        if plane not in ('xz', 'yz', 'xy'):
            raise ValueError('stick_plane must be xz, yz or xy')
        self.plane = plane
        self.axes = [0.] * 6
        self.buttons = [0] * 15
        self.received = -math.inf

    def update(self, axes, buttons, now):
        if len(axes) < 6 or len(buttons) < 15 or not all(math.isfinite(v) for v in axes):
            self.received = -math.inf
            return
        self.axes = [max(-1., min(1., v)) for v in axes[:6]]
        self.buttons = list(buttons)
        self.received = now

    def command(self, now):
        # Hold L1 to move. Disconnect, malformed packets, and release stop motion.
        if now - self.received > 0.25 or not self.buttons[9]:
            return [0.] * 6, 0.
        a = [0. if abs(v) < .08 else v for v in self.axes]
        b = self.buttons
        other = float(b[11] - b[12])  # D-pad up/down
        # Optical camera coordinates: X right, Y down, Z forward.
        linear = {'xz': [-a[0], -other, a[1]],
                  'xy': [-a[0], -a[1], other],
                  'yz': [other, -a[0], a[1]]}[self.plane]
        # ROS joy's SDL axes are inverted: triggers rest at 0, pressed at -1.
        angular = [a[3], -a[2], max(0., -a[5]) - max(0., -a[4])]
        return [v * .5 for v in linear] + [v * .4 for v in angular], float(b[0] - b[1]) * .5
