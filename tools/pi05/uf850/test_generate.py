import pytest
from generate import episode

@pytest.mark.parametrize('seed',[1,7,101])
def test_teacher_places_bowl_using_actual_contacts(seed):
    result, data = episode(seed,render=False)
    assert result['success'] and result['lifted'] and result['bilateral_contact']
    assert result['max_bowl_height']>.82
    assert data['state'].shape==(result['frames'],7)
    assert data['actions'].shape==data['state'].shape
    assert result['gripper_motor_effort_nm']==3.


@pytest.mark.parametrize('seed',[1001,1019])
def test_teacher_from_varied_empty_gripper_pose(seed):
    result,data=episode(seed,render=False,varied_start=True,grasp_dwell=20)
    assert result['success'] and result['bilateral_contact']
    assert result['frames']>=246
    assert result['varied_start'] and result['grasp_dwell_frames']==20
