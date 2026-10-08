"""Reweight real training frames at the instruction-dependent initial reach."""
from contextlib import contextmanager
import json
from pathlib import Path
import numpy as np
from tasks import OBJECTS


def training_indices(manifest):
    rows=manifest['train']
    if {row['layout_seed'] for row in rows} & {row['layout_seed'] for row in manifest['heldout']}:
        raise ValueError('Training and validation layouts overlap')
    counts=np.zeros((3,3),dtype=int)
    for row in rows:
        identity=OBJECTS.index(row['object']);counts[identity,row['slot_order'][identity]]+=1
    if np.any(counts==0):raise ValueError('Every object needs examples in every slot')
    mapped=[];offset=0
    for row in rows:
        identity=OBJECTS.index(row['object']);slot=row['slot_order'][identity]
        balance=counts[identity].sum()/3/counts[identity,slot]
        for frame in range(12,row['frames']):
            # Omit pre-task idle observations. The fork at frame 12 has the
            # same pixels/state for three targets and requires the instruction.
            weight=round(100*balance) if frame==12 else round(8*balance) if frame<42 else 1
            mapped.extend([offset+frame]*max(1,weight))
        offset+=row['frames']
    return np.asarray(mapped,dtype=np.int64),offset,counts


class FrameSubset:
    def __init__(self,dataset,indices):self.dataset,self.indices=dataset,indices
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):return self.dataset[int(self.indices[index])]


@contextmanager
def selection_sampling(repo_id,manifest_path,output):
    from openpi.training import data_loader
    manifest=json.loads(Path(manifest_path).read_text())
    indices,total,counts=training_indices(manifest)
    original=data_loader.create_torch_dataset
    def create(data_config,action_horizon,model_config):
        dataset=original(data_config,action_horizon,model_config)
        if data_config.repo_id!=repo_id:return dataset
        if len(dataset)!=total:raise ValueError('Sampling manifest does not match the training dataset')
        return FrameSubset(dataset,indices)
    profile={'name':'instruction_dependent_initial_reach','source_repo':repo_id,
        'source_training_frames':total,'virtual_sampling_frames':len(indices),'demonstrations_added':0,
        'idle_frames_excluded':12,'first_reach_frame':12,'first_reach_weight':100,
        'following_reach_frames':[13,41],'following_reach_weight':8,
        'balance_first_reach_by':'object / slot counts','object_slot_counts':counts.tolist(),
        'note':'Only train frames are resampled. Action chunks retain their original temporal context; no fabricated trajectories.'}
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(profile,indent=2))
    data_loader.create_torch_dataset=create
    try:yield profile
    finally:data_loader.create_torch_dataset=original
