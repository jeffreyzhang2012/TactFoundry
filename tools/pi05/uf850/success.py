"""Object-specific placement scoring from simulator ground truth."""
import numpy as np
import pybullet as p
from generate import JOINTS, bowl_contact


def placement_checks(*, grasped_and_lifted, distance, upright, plate_contact,
                     released, robot_contact, linear_speed, angular_speed):
    return dict(grasped_and_lifted=bool(grasped_and_lifted),
                centered=bool(distance < .025), upright=bool(upright > .8),
                supported_by_plate=bool(plate_contact), released=bool(released),
                clear_of_robot=not bool(robot_contact),
                stationary=bool(linear_speed < .01 and angular_speed < .1))


class PlacementEvaluator:
    def __init__(self, world, body, plate):
        self.world, self.body, self.plate = world, body, plate
        self.initial_height = p.getBasePositionAndOrientation(body, physicsClientId=world.client)[0][2]
        self.max_height = self.initial_height
        self.bilateral = self.lifted = self.grasped_and_lifted = False

    def observe(self):
        w = self.world
        xyz, quat = p.getBasePositionAndOrientation(self.body, physicsClientId=w.client)
        plate_xyz = p.getBasePositionAndOrientation(self.plate, physicsClientId=w.client)[0]
        bilateral = bowl_contact(w, self.body)  # helper checks the supplied body's jaw contacts
        lifted = xyz[2] - self.initial_height > .04
        self.bilateral |= bilateral
        self.lifted |= lifted
        self.grasped_and_lifted |= bilateral and lifted
        self.max_height = max(self.max_height, xyz[2])
        linear, angular = p.getBaseVelocity(self.body, physicsClientId=w.client)
        plate_contacts = p.getContactPoints(bodyA=self.body, bodyB=self.plate, physicsClientId=w.client)
        robot_contacts = p.getContactPoints(bodyA=self.body, bodyB=w.robot, physicsClientId=w.client)
        distance = float(np.linalg.norm(np.asarray(xyz[:2]) - plate_xyz[:2]))
        checks = placement_checks(grasped_and_lifted=self.grasped_and_lifted,
            distance=distance, upright=p.getMatrixFromQuaternion(quat)[8],
            plate_contact=any(c[9] > .01 for c in plate_contacts),
            released=w.targets[JOINTS[-1]] < .1,
            robot_contact=any(c[9] > .05 for c in robot_contacts),
            linear_speed=np.linalg.norm(linear), angular_speed=np.linalg.norm(angular))
        return checks, xyz, distance
