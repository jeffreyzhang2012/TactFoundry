import numpy as np
from tactile_simulation.gamepad import Gamepad


def test_mapping_release_timeout_and_invalid_packets():
    pad = Gamepad('yz')
    buttons = [0] * 15
    buttons[0] = 1
    pad.update([1., 1., 1., 1., 0., -1.], buttons, 1.)
    twist, _ = pad.command(1.1)
    assert twist[0] == 0. and -twist[1] == twist[2] > 0.
    assert twist[3:] == [.4, -.4, .4]
    assert not any(pad.command(1.3)[0])
    buttons[0] = 0
    pad.update([0.] * 6, buttons, 2.)
    assert not any(pad.command(2.)[0])
    buttons[0] = 1
    pad.update([float('nan')] * 6, buttons, 3.)
    assert not any(pad.command(3.)[0])
    pad.update([], [], 4.)
    assert not any(pad.command(4.)[0])


def test_horizontal_mode_and_gripper():
    pad = Gamepad()
    buttons = [0] * 15
    buttons[10] = buttons[11] = buttons[0] = 1
    pad.update([.5, 1., 0., 0., 0., 0.], buttons, 0.)
    twist, grip = pad.command(0.)
    speed = twist[2]
    assert speed > 0.
    assert np.allclose(twist, [-speed/2, -speed, speed, 0., 0., 0.])
    assert grip == .5
    buttons[11] = 0
    pad.update([0., 1., 0., 0., 0., 0.], buttons, 1.)
    assert pad.command(1.)[0] == [0., 0., speed, 0., 0., 0.]


def test_shoulders_jog_gripper_with_centered_sticks():
    pad = Gamepad()
    buttons = [0] * 15
    for left, right, expected in [(1, 0, -.5), (0, 1, .5), (1, 1, 0.)]:
        buttons[9], buttons[10] = left, right
        pad.update([0.] * 6, buttons, 0.)
        assert pad.command(0.) == ([0.] * 6, expected)
        assert pad.command(.3) == ([0.] * 6, 0.)


def test_sticks_move_without_enable_button():
    pad = Gamepad()
    pad.update([0., 1., 0., 0., 0., 0.], [0] * 15, 0.)
    twist, grip = pad.command(0.)
    assert twist[2] > 0. and grip == 0.
