"""Procedural grasp objects; dimensions are meters, not scanned assets."""
import math

KINDS = ('cube', 'box', 'sphere', 'cylinder', 'capsule', 'bottle', 'mug', 'bowl')
PALETTE = ((0.9, 0.3, 0.2, 1.), (0.2, 0.65, 0.95, 1.),
           (0.3, 0.8, 0.45, 1.), (0.95, 0.75, 0.2, 1.), (0.65, 0.4, 0.85, 1.))


def part(shape, size, xyz=(0., 0., 0.), yaw=0.):
    return {'shape': shape, 'size': size, 'xyz': xyz, 'yaw': yaw}


def geometry(kind):
    if kind == 'plate':
        return [part('cylinder', (.14, .14, .008))], .008
    if kind == 'cube':
        return [part('box', (0.035, 0.035, 0.035))], 0.035
    if kind == 'box':
        return [part('box', (0.025, 0.045, 0.025))], 0.025
    if kind == 'sphere':
        return [part('sphere', (0.04, 0.04, 0.04))], 0.04
    if kind == 'cylinder':
        return [part('cylinder', (0.035, 0.035, 0.055))], 0.055
    if kind == 'capsule':
        # A cylindrical body with rounded end caps as three convex parts.
        return [part('cylinder', (0.024, 0.024, 0.04)),
                part('sphere', (0.024,)*3, (0., 0., -0.02)),
                part('sphere', (0.024,)*3, (0., 0., 0.02))], 0.064
    if kind == 'bottle':
        return [part('cylinder', (0.032, 0.032, 0.065)),
                part('cylinder', (0.016, 0.016, 0.025), (0., 0., 0.045))], 0.115
    if kind in ('mug', 'bowl'):
        radius, height = (0.022, 0.045) if kind == 'mug' else (0.024, 0.022)
        parts = [part('cylinder', (radius*2, radius*2, 0.004),
                      (0., 0., -height/2 + 0.002))]
        for i in range(12):
            angle = i * 2*math.pi/12
            parts.append(part('box', (0.004, 2*radius*math.tan(math.pi/12), height),
                              (radius*math.cos(angle), radius*math.sin(angle), 0.), angle))
        if kind == 'mug':
            parts.extend([
                part('box', (0.014, 0.006, 0.005), (0.029, 0., 0.014)),
                part('box', (0.014, 0.006, 0.005), (0.029, 0., -0.014)),
                part('box', (0.005, 0.006, 0.033), (0.036, 0., 0.)),
            ])
        return parts, height
    raise ValueError(f'unknown object type: {kind}')


def slot_position(slot):
    """48 spaced slots on a low tabletop in front of the robot."""
    if not 0 <= slot < 48:
        raise ValueError('slot must be between 0 and 47')
    return 0.17 + 0.07*(slot % 8), -0.175 + 0.07*(slot // 8)
