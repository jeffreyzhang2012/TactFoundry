import numpy as np
from builtin_interfaces.msg import Time
from visualization_msgs.msg import Marker
from tactile_simulation.forces import jaw_contact_force
from tactile_simulation.force_markers import force_markers


def contact(a, b, link_a, link_b):
    return (0, a, b, link_a, link_b, (1., 0., 0.), (1., 0., 0.),
            (0., 0., 1.), -.001, 10., 2., (1., 0., 0.), 0., (0., 1., 0.))


def test_reaction_direction_friction_torque_and_zero():
    sample = contact(1, 2, 3, -1)
    reading = jaw_contact_force([sample], 1, {3}, np.zeros(3))
    assert np.allclose(reading['force'], [2., 0., 10.])
    assert np.allclose(reading['torque'], [0., -10., 0.])
    assert reading['normal_load'] == 10.
    reverse = jaw_contact_force([sample], 2, {-1}, np.zeros(3))
    assert np.allclose(reverse['force'], -reading['force'])
    empty = jaw_contact_force([contact(1, 1, 3, 4)], 1, {3}, np.zeros(3))
    assert not empty['normal_load'] and not empty['force'].any()


def test_marker_units_saturation_and_contact_release():
    reading = dict(force=np.array([100., 0., 0.]), point=np.zeros(3), normal_load=100.)
    markers = force_markers({'left': reading}, Time()).markers
    assert markers[0].action == Marker.ADD
    assert markers[0].points[1].x == .25
    assert '100.00 N' in markers[1].text
    reading['force'][:] = 0.
    reading['normal_load'] = 0.
    released = force_markers({'left': reading}, Time()).markers
    assert released[0].action == Marker.DELETE
    assert '0.00 N' in released[1].text
