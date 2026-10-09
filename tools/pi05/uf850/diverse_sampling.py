"""Balance instruction-dependent reaches in diverse, physically verified demos."""
from contextlib import contextmanager
from collections import Counter
import json
from pathlib import Path
import numpy as np
from focus_sampling import FrameSubset


def training_indices(manifest):
    rows=manifest['train']
    if {r['layout_seed'] for r in rows}&{r['layout_seed'] for r in manifest['heldout']}:
        raise ValueError('Training and validation layouts overlap')
    counts=Counter((r['color'],r['kind'],r['slot']) for r in rows)
    mean=len(rows)/len(counts)
    indices=[];offset=0
    for row in rows:
        balance=min(3.,max(.5,mean/counts[(row['color'],row['kind'],row['slot'])]))
        for frame in range(12,row['frames']):
            weight=max(1,round((300 if frame==12 else 3 if frame<42 else 1)*balance))
            indices.extend([offset+frame]*weight)
        offset+=row['frames']
    return np.asarray(indices,dtype=np.int64),offset,counts


@contextmanager
def selection_sampling(repo_id,manifest_path,output):
    from openpi.training import data_loader
    manifest=json.loads(Path(manifest_path).read_text())
    indices,total,counts=training_indices(manifest)
    original=data_loader.create_torch_dataset
    def create(data_config,action_horizon,model_config):
        dataset=original(data_config,action_horizon,model_config)
        if data_config.repo_id!=repo_id:return dataset
        if len(dataset)!=total:raise ValueError('Sampling manifest does not match dataset')
        return FrameSubset(dataset,indices)
    profile={'name':'diverse_instruction_reaches','source_repo':repo_id,
        'source_training_frames':total,'virtual_sampling_frames':len(indices),
        'idle_frames_excluded':12,'first_reach_frame':12,'first_reach_weight':300,
        'following_reach_weight':3,'balance':'color / shape / spatial slot',
        'coverage':[{ 'color':c,'kind':k,'slot':s,'episodes':n} for (c,k,s),n in sorted(counts.items())],
        'demonstrations_added':0}
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(profile,indent=2))
    data_loader.create_torch_dataset=create
    try:yield profile
    finally:data_loader.create_torch_dataset=original
