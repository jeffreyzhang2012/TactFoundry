import numpy as np
import pytest
from focus_sampling import FrameSubset,training_indices
from tasks import OBJECTS


def fixture():
    return {'train':[{'layout_seed':i,'object':name,'slot_order':order,'frames':60}
                      for i,order in enumerate([(0,1,2),(1,2,0),(2,0,1)]) for name in OBJECTS],
            'heldout':[{'layout_seed':100}]}


def test_sampling_preserves_source_frames_and_excludes_idle_and_validation():
    indices,total,counts=training_indices(fixture())
    assert total==540 and np.array_equal(counts,np.ones((3,3)))
    assert np.all(indices%60>=12) and indices.max()<total
    assert np.sum(indices==12)==100 and np.sum(indices==13)==8 and np.sum(indices==42)==1
    source=[{'actions':np.arange(i,i+10)} for i in range(total)]
    view=FrameSubset(source,indices)
    np.testing.assert_array_equal(view[0]['actions'],source[12]['actions'])
    assert view[0] is source[12]


def test_sampling_rejects_layout_leakage():
    manifest=fixture();manifest['heldout']=[{'layout_seed':0}]
    with pytest.raises(ValueError,match='overlap'):training_indices(manifest)
