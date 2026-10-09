import json
import numpy as np
from diverse_tasks import layout_spec,HELDOUT_PAIRS,COLORS,KINDS
from diverse_sampling import training_indices


def test_heldout_combinations_and_unambiguous_labels():
    for seed in range(10001,10101):
        train=layout_spec(seed,'train')
        held=layout_spec(seed+10000,'heldout')
        labels=[(r['color'],r['kind']) for r in train['objects']]
        assert len(set(labels))==3
        assert not set(labels)&HELDOUT_PAIRS
        assert any((r['color'],r['kind']) in HELDOUT_PAIRS for r in held['objects'])
        assert sorted(train['slot_order'])==[0,1,2]


def test_sampling_retains_original_chunks_and_excludes_validation():
    rows=[{'layout_seed':1,'frames':45,'color':'red','kind':'cube','slot':s} for s in range(3)]
    indices,total,counts=training_indices({'train':rows,'heldout':[{'layout_seed':2}]})
    assert total==135
    assert set(indices)<=set(range(12,45))|set(range(57,90))|set(range(102,135))
    assert np.sum(indices==12)==300
    assert np.sum(indices==13)==3
    assert np.sum(indices==44)==1


def test_pure_noise_cannot_leak_action_target():
    import jax
    import jax.numpy as jnp
    from conditional_flow import noisy_actions
    actions=jnp.zeros((3,10,32))
    a,target,times=noisy_actions(jax.random.key(7),actions,1.)
    b,_,_=noisy_actions(jax.random.key(7),actions+1.,1.)
    assert np.array_equal(a,b)
    assert np.array_equal(a[0],a[1])
    assert np.all(np.asarray(times)==1.)
    c,_,_=noisy_actions(jax.random.key(7),actions,0.)
    d,_,_=noisy_actions(jax.random.key(7),actions+1.,0.)
    assert not np.array_equal(c,d)


def test_new_profile_freezes_vision_and_old_profiles_are_unchanged(tmp_path):
    from training import make_config
    from flax import nnx
    (tmp_path/'training-profile.json').write_text(json.dumps({'pure_noise_probability':.5,'freeze_vision':True}))
    cfg=make_config(tmp_path,'local/new',tmp_path/'base')
    assert cfg.model.pure_noise_probability==.5
    predicate=nnx.filterlib.to_predicate(cfg.freeze_filter)
    assert predicate(('PaliGemma','img','embedding','kernel'),None)
    assert not predicate(('action_out_proj','kernel'),None)
