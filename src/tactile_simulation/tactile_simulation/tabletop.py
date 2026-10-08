"""Robot-sized interpretation of the LIBERO bowl/plate tabletop task.

These are procedural props, not LIBERO assets or a checkpoint-compatible env.
"""
import math
import numpy as np

TABLE_HEIGHT = .75
EXTERNAL_EYE = (.98, -.95, 1.55)
EXTERNAL_TARGET = (.27, 0., 1.02)
# Separately solved poses for each robot, with the gripper flange at
# (.39, 0, 1.08), tool +Z down and the existing +90-degree gripper mounting.
# Keeping these separate avoids copying Franka or xArm joint angles to the 850.
HOMES = {
    'uf850': (0., -.200403, -.625593, -3.141554, .425190, -.000052, 0.),
    'xarm6': (0., -.229144, -.742299, 0., .971442, 3.141586, 0.),
}


def optical_pose():
    forward = np.asarray(EXTERNAL_TARGET) - EXTERNAL_EYE
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.column_stack((right, down, forward))
    rpy = (math.atan2(rotation[2, 1], rotation[2, 2]),
           math.asin(-rotation[2, 0]), math.atan2(rotation[1, 0], rotation[0, 0]))
    return EXTERNAL_EYE, rpy


def fixtures():
    # Size, center, RGBA. Top coincides with the robot base plane.
    yield (1.0, .8, .06), (.30, 0., .72), (.57, .43, .29, 1.)
    for x in (-.12, .72):
        for y in (-.32, .32):
            yield (.045, .045, .69), (x, y, .345), (.18, .19, .21, 1.)
    # Rear cabinet and wall are outside the manipulation region.
    yield (.35, .55, .72), (-.44, .55, .36), (.32, .36, .4, 1.)
    yield (1.7, .04, 1.5), (.15, .94, .75), (.78, .79, .76, 1.)


def props():
    # name, catalogue geometry, scale, position XY, color, mass kg
    yield 'target_plate', 'plate', 1., (.43, -.17), (.88, .89, .9, 1.), .12
    yield 'black_bowl', 'bowl', 1.35, (.43, 0.), (.08, .09, .10, 1.), .06
    yield 'ramekin', 'bowl', 1.05, (.43, .15), (.92, .89, .78, 1.), .05
    yield 'bottle', 'bottle', 1., (.62, .23), (.22, .48, .3, 1.), .05
    yield 'distractor_block', 'cube', 1., (.61, -.26), (.7, .25, .17, 1.), .05
