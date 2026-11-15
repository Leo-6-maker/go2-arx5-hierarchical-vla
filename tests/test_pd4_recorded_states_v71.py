import pytest

from scripts.sla.diagnose_pd4_recorded_states_v71 import admission,OPERATION,PROBES


@pytest.mark.parametrize('key,value',[('maximum_clone_calls',160),('planned_variants',160),
    ('original_sampling_steps',20),('indices',list(reversed(PROBES))),('new_model_generations',1),
    ('optimizer_updates',1),('physics',True),('native_execution_admitted',True)])
def test_only_declared_saved16_original64_and_labels_are_admitted(key,value):
    spec=dict(operation=OPERATION,indices=PROBES.copy(),planned_variants=96,maximum_clone_calls=81,
        original_sampling_steps=10,new_model_generations=0,optimizer_updates=0,physics=False,native_execution_admitted=False)
    admission(spec);spec[key]=value
    with pytest.raises(PermissionError):admission(spec)
