import numpy as np
import pytest
from openpi import transforms
from openpi.models.model import ModelType
from training import JointInputs, make_config

def test_joint_contract_and_delta_round_trip():
    state=np.array([.1,-.2,-.6,-3.1,.4,0.,.25],dtype=np.float32)
    actions=np.stack([state+.01,state+.02]); actions[:,6]=.45
    observation={'observation/state':state,'observation/image':np.zeros((224,224,3),np.uint8),
                 'observation/wrist_image':np.zeros((224,224,3),np.uint8),'actions':actions,'prompt':'place bowl'}
    converted=JointInputs(ModelType.PI05)(observation)
    assert converted['state'].shape==(7,)
    assert not converted['image_mask']['right_wrist_0_rgb']
    delta=transforms.DeltaActions(transforms.make_bool_mask(6,-1))(converted)
    assert np.allclose(delta['actions'][:,:6],[[.01]*6,[.02]*6])
    assert np.allclose(delta['actions'][:,6],.45)
    restored=transforms.AbsoluteActions(transforms.make_bool_mask(6,-1))(delta)
    assert np.allclose(restored['actions'],actions)
    with pytest.raises(ValueError):
        JointInputs(ModelType.PI05)(dict(observation,**{'observation/state':np.zeros(8)}))

def test_lora_config_is_local_and_uses_850_actions(tmp_path):
    cfg=make_config(tmp_path,'local/850',tmp_path/'weights')
    assert cfg.model.pi05 and cfg.model.action_horizon==10
    assert cfg.ema_decay is None and not cfg.wandb_enabled
    assert cfg.policy_metadata['robot']=='uf850_ag95'
    assert cfg.data.create(cfg.assets_dirs,cfg.model).action_sequence_keys==('actions',)
