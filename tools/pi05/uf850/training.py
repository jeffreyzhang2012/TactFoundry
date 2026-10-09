"""Local OpenPI extension: explicit 850 joint-action contract, no upstream patch."""
import dataclasses
import json
from pathlib import Path
import numpy as np
from openpi import transforms
from openpi.models import pi0_config
from openpi.policies import libero_policy
from openpi.training import config, optimizer, weight_loaders

NAME='pi05_uf850_lora'

@dataclasses.dataclass(frozen=True)
class JointInputs(transforms.DataTransformFn):
    model_type: object
    def __call__(self,data):
        state=np.asarray(data['observation/state'])
        if state.shape != (7,) or not np.isfinite(state).all():
            raise ValueError('850 state must be six arm angles and AG95 master angle in radians')
        if 'actions' in data and np.asarray(data['actions']).shape[-1] != 7:
            raise ValueError('850 actions require seven absolute joint targets')
        # Reuse image-channel handling only. State is our joint representation;
        # no benchmark Cartesian/gripper sign conversion is performed.
        return libero_policy.LiberoInputs(self.model_type)(data)

@dataclasses.dataclass(frozen=True)
class RobotData(config.DataConfigFactory):
    def create(self,assets_dirs,model_config):
        repack=transforms.Group(inputs=[transforms.RepackTransform({
            'observation/image':'image','observation/wrist_image':'wrist_image',
            'observation/state':'state','actions':'actions','prompt':'prompt'})])
        data=transforms.Group(inputs=[JointInputs(model_config.model_type)],
                              outputs=[libero_policy.LiberoOutputs()])
        mask=transforms.make_bool_mask(6,-1)
        data=data.push(inputs=[transforms.DeltaActions(mask)],outputs=[transforms.AbsoluteActions(mask)])
        return dataclasses.replace(self.create_base_config(assets_dirs,model_config),
            repack_transforms=repack,data_transforms=data,
            model_transforms=config.ModelTransformFactory()(model_config),action_sequence_keys=('actions',))

def make_config(root,repo_id,weights,*,steps=300,batch_size=4,exp_name='pilot',resume=False):
    profile_path=Path(root)/'training-profile.json'
    profile=json.loads(profile_path.read_text()) if profile_path.exists() else {}
    model_class=pi0_config.Pi0Config
    options={}
    if 'pure_noise_probability' in profile:
        from conditional_flow import ConditionalPi0Config
        model_class=ConditionalPi0Config
        options['pure_noise_probability']=profile['pure_noise_probability']
    model=model_class(pi05=True,action_horizon=10,**options,
                              paligemma_variant='gemma_2b_lora',action_expert_variant='gemma_300m_lora')
    freeze=model.get_freeze_filter()
    if profile.get('freeze_vision',False):
        from flax import nnx
        from openpi.shared import nnx_utils
        freeze=nnx.Any(freeze,nnx_utils.PathRegex('.*PaliGemma/img/.*'))
    dataset_path=Path(root)/'dataset-profile.json'
    dataset_profile=json.loads(dataset_path.read_text()) if dataset_path.exists() else {}
    return config.TrainConfig(name=NAME,exp_name=exp_name,model=model,
        data=RobotData(repo_id=repo_id,base_config=config.DataConfig(prompt_from_task=True)),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(Path(weights)/'params')),
        freeze_filter=freeze,ema_decay=None,batch_size=batch_size,num_workers=0,
        num_train_steps=steps,log_interval=10,save_interval=100,keep_period=None,
        assets_base_dir=str(Path(root)/'assets'),checkpoint_base_dir=str(Path(root)/'checkpoints'),
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=20,peak_lr=1e-4,decay_steps=steps,decay_lr=1e-5),
        wandb_enabled=False,resume=resume,policy_metadata={'robot':'uf850_ag95','fps':20,
            'dataset':repo_id,
            'training_profile':profile,'dataset_profile':dataset_profile,
            'actions':'absolute joint1..6 + AG95 master angle in radians','gripper_motor_effort_nm':3.})

def register(cfg):
    config._CONFIGS_DICT[cfg.name]=cfg
