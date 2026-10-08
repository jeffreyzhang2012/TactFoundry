"""PyBullet backend with actual contacts and a CPU-rendered wrist camera."""
import math
from pathlib import Path
import random
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import pybullet as p
from ament_index_python.packages import get_package_share_directory

from .objects import KINDS, PALETTE, geometry, slot_position

HOME = (0., -0.6, -0.8, 0., 1.4, 0., 0.)


class World:
    def __init__(self, description):
        self.client = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(1/240, physicsClientId=self.client)
        p.setPhysicsEngineParameter(numSolverIterations=80, physicsClientId=self.client)
        robot = ET.fromstring(description)
        # SDF-only closed-loop fragments are not part of Bullet's URDF tree.
        for extension in robot.findall('gazebo'):
            robot.remove(extension)
        for mesh in robot.findall('.//mesh'):
            uri = mesh.attrib['filename']
            if uri.startswith('package://'):
                package, relative = uri[10:].split('/', 1)
                mesh.set('filename', str(Path(get_package_share_directory(package)) / relative))
        # Bullet otherwise assigns one kilogram to massless URDF frame links.
        for link in robot.findall('link'):
            if link.find('inertial') is None:
                inertial = ET.SubElement(link, 'inertial')
                ET.SubElement(inertial, 'mass', value='0.000001')
                ET.SubElement(inertial, 'inertia', ixx='0.000000001', iyy='0.000000001',
                              izz='0.000000001', ixy='0', ixz='0', iyz='0')
        ET.indent(robot)
        with tempfile.NamedTemporaryFile(suffix='.urdf') as source:
            source.write(ET.tostring(robot))
            source.flush()
            self.robot = p.loadURDF(source.name, useFixedBase=True,
                                    flags=p.URDF_USE_INERTIA_FROM_FILE,
                                    physicsClientId=self.client)
        self.joints, self.links, self.mimics, self.limits = {}, {}, {}, {}
        for i in range(p.getNumJoints(self.robot, physicsClientId=self.client)):
            info = p.getJointInfo(self.robot, i, physicsClientId=self.client)
            self.links[info[12].decode()] = i
            if info[2] != p.JOINT_FIXED:
                name = info[1].decode()
                self.joints[name] = i
                self.limits[name] = (info[8], info[9], max(info[10], 1.))
            p.changeDynamics(self.robot, i, lateralFriction=1.2,
                             physicsClientId=self.client)
        for joint in robot.findall('joint'):
            mimic = joint.find('mimic')
            if mimic is not None:
                self.mimics[joint.attrib['name']] = (
                    mimic.attrib['joint'], float(mimic.get('multiplier', 1)),
                    float(mimic.get('offset', 0)))
        self.targets = dict(zip([f'joint{i}' for i in range(1, 7)] +
                                ['ag95_left_outer_knuckle_joint'], HOME))
        self.objects = {}
        self.next_id = 1
        self.create_static_box((2., 2., 0.01), (0., 0., -0.005), (0.25,)*3 + (1.,))
        self.create_static_box((0.64, 0.5, 0.02), (0.415, 0., 0.05), (0.55, 0.42, 0.28, 1.))
        self.reset_arm()

    def create_static_box(self, size, xyz, color):
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[v/2 for v in size],
                                           physicsClientId=self.client)
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[v/2 for v in size],
                                    rgbaColor=color, physicsClientId=self.client)
        return p.createMultiBody(0, collision, visual, xyz, physicsClientId=self.client)

    def positions_for_targets(self):
        positions = dict(self.targets)
        for name, (parent, factor, offset) in self.mimics.items():
            positions[name] = positions[parent]*factor + offset
        return positions

    def reset_arm(self):
        self.targets = dict(zip(self.targets, HOME))
        for name, value in self.positions_for_targets().items():
            p.resetJointState(self.robot, self.joints[name], value, physicsClientId=self.client)

    def command(self, names, values):
        for name, value in zip(names, values):
            if name in self.targets and math.isfinite(value):
                lower, upper, _ = self.limits[name]
                self.targets[name] = min(upper, max(lower, value))

    def step(self):
        for name, value in self.positions_for_targets().items():
            p.setJointMotorControl2(self.robot, self.joints[name], p.POSITION_CONTROL,
                                   targetPosition=value, force=self.limits[name][2],
                                   positionGain=0.25, velocityGain=1,
                                   physicsClientId=self.client)
        for _ in range(4):
            p.stepSimulation(physicsClientId=self.client)

    def twist_to_world(self, twist, frame):
        """Rotate linear and angular rates at the tool center into world axes."""
        state = p.getLinkState(self.robot, self.links[frame], computeForwardKinematics=True,
                               physicsClientId=self.client)
        rotation = np.asarray(p.getMatrixFromQuaternion(state[5])).reshape(3, 3)
        return np.concatenate((rotation @ np.asarray(twist[:3]),
                               rotation @ np.asarray(twist[3:])))

    def cartesian_command(self, twist, grip, dt, frame=None):
        """Damped differential IK; optionally express rates in a moving link frame."""
        if len(twist) != 6 or not np.isfinite(twist).all() or not math.isfinite(grip):
            return
        if frame is not None:
            twist = self.twist_to_world(twist, frame)
        names, positions = self.joint_states()
        linear, angular = p.calculateJacobian(
            self.robot, self.links['ag95_ag95_base_link'], [0., 0., .15], positions,
            [0.] * len(positions), [0.] * len(positions), physicsClientId=self.client)
        columns = [names.index(f'joint{i}') for i in range(1, 7)]
        jacobian = np.vstack((linear, angular))[:, columns]
        velocity = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + np.eye(6) * .0025, np.asarray(twist))
        # Scale together to preserve direction near singular configurations.
        velocity /= max(1., np.max(np.abs(velocity)) / .5)
        values = [positions[i] + float(v) * dt for i, v in zip(columns, velocity)]
        arm = [f'joint{i}' for i in range(1, 7)]
        gripper = 'ag95_left_outer_knuckle_joint'
        self.command(arm + [gripper], values + [self.targets[gripper] + grip * dt])

    def joint_states(self):
        names = list(self.joints)
        values = p.getJointStates(self.robot, list(self.joints.values()),
                                  physicsClientId=self.client)
        return names, [state[0] for state in values]

    def spawn(self, kind, count, seed):
        if kind not in (*KINDS, 'mixed') or not 1 <= count <= 48:
            raise ValueError('choose a listed type and count between 1 and 48')
        used = {obj['slot'] for obj in self.objects.values()}
        available = [i for i in range(48) if i not in used]
        if count > len(available):
            raise ValueError(f'only {len(available)} free object slots remain; clear the scene first')
        rng = random.Random(seed)
        slots = rng.sample(available, count)
        for n, slot in enumerate(slots):
            chosen = KINDS[n % len(KINDS)] if kind == 'mixed' else kind
            parts, height = geometry(chosen)
            shape_types, half_extents, radii, lengths, positions, orientations = [], [], [], [], [], []
            for component in parts:
                shape_types.append({'box': p.GEOM_BOX, 'sphere': p.GEOM_SPHERE,
                                    'cylinder': p.GEOM_CYLINDER}[component['shape']])
                half_extents.append([v/2 for v in component['size']])
                radii.append(component['size'][0]/2)
                lengths.append(component['size'][2])
                positions.append(component['xyz'])
                orientations.append(p.getQuaternionFromEuler((0., 0., component['yaw'])))
            collision = p.createCollisionShapeArray(
                shapeTypes=shape_types, halfExtents=half_extents, radii=radii,
                lengths=lengths, collisionFramePositions=positions,
                collisionFrameOrientations=orientations, physicsClientId=self.client)
            color = PALETTE[n % len(PALETTE)]
            visual = p.createVisualShapeArray(
                shapeTypes=shape_types, halfExtents=half_extents, radii=radii,
                lengths=lengths, visualFramePositions=positions,
                visualFrameOrientations=orientations, rgbaColors=[color]*len(parts),
                physicsClientId=self.client)
            x, y = slot_position(slot)
            body = p.createMultiBody(0.05, collision, visual, (x, y, 0.062 + height/2),
                                     p.getQuaternionFromEuler((0., 0., rng.uniform(-math.pi, math.pi))),
                                     physicsClientId=self.client)
            p.changeDynamics(body, -1, lateralFriction=0.9, restitution=0.,
                             physicsClientId=self.client)
            self.objects[self.next_id] = dict(body=body, slot=slot, kind=chosen,
                                              parts=parts, color=color)
            self.next_id += 1

    def clear(self):
        for obj in self.objects.values():
            p.removeBody(obj['body'], physicsClientId=self.client)
        self.objects.clear()

    def part_poses(self):
        for oid, obj in self.objects.items():
            xyz, quat = p.getBasePositionAndOrientation(obj['body'], physicsClientId=self.client)
            for i, part in enumerate(obj['parts']):
                pose = p.multiplyTransforms(xyz, quat, part['xyz'],
                                            p.getQuaternionFromEuler((0., 0., part['yaw'])))
                yield oid*100+i, part, obj['color'], pose

    def render(self, frame, width=320, height=240):
        state = p.getLinkState(self.robot, self.links[frame], computeForwardKinematics=True,
                               physicsClientId=self.client)
        xyz, quat = np.array(state[4]), state[5]
        rotation = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        # ROS optical: X right, Y down, Z forward; OpenGL view uses Y up.
        view = p.computeViewMatrix(xyz, xyz + rotation[:, 2], -rotation[:, 1])
        near, far, fov = 0.02, 3., 58.
        projection = p.computeProjectionMatrixFOV(fov, width/height, near, far)
        result = p.getCameraImage(width, height, view, projection,
                                  renderer=p.ER_TINY_RENDERER, physicsClientId=self.client)
        color = np.asarray(result[2], dtype=np.uint8).reshape(height, width, 4)[:, :, :3].copy()
        buffer = np.asarray(result[3], dtype=np.float32).reshape(height, width)
        depth = far*near / (far - (far-near)*buffer)
        depth[buffer >= 0.99999] = 0.
        focal = height / (2*math.tan(math.radians(fov)/2))
        return color, depth.astype('<f4'), focal

    def close(self):
        p.disconnect(physicsClientId=self.client)
