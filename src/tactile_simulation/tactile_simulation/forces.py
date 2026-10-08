"""Aggregate solved contact reactions acting on a simulated jaw."""
import numpy as np


def jaw_contact_force(contacts, robot, links, origin):
    force, torque, weighted_point = np.zeros(3), np.zeros(3), np.zeros(3)
    normal_load = 0.
    for contact in contacts:
        if contact[1] == contact[2]:
            continue
        if contact[1] == robot and contact[3] in links:
            sign, point = 1., np.asarray(contact[5])
        elif contact[2] == robot and contact[4] in links:
            sign, point = -1., np.asarray(contact[6])
        else:
            continue
        reaction = sign * (np.asarray(contact[7]) * contact[9] +
                           np.asarray(contact[11]) * contact[10] +
                           np.asarray(contact[13]) * contact[12])
        force += reaction
        torque += np.cross(point - origin, reaction)
        normal_load += contact[9]
        weighted_point += point * contact[9]
    point = weighted_point / normal_load if normal_load > 0. else np.asarray(origin)
    return dict(force=force, torque=torque, point=point, normal_load=float(normal_load))
