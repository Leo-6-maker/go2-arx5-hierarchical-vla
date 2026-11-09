"""Memory-only operation contract seams; no real GPU qualification here."""
import ast,inspect
import pytest
from go2arx5_vla.world_model import target_gr00t_core_v1 as core
from go2arx5_vla.world_model import task_learning_core_restore_memory_v1 as memory


def config():
    return dict(operation=memory.OPERATION,formal_training=False,original_Trainer_execution=True,
        GR00T_updates=0,memory_policy='CORE_FROZEN_BACKBONE_SWAP_NO_WEIGHT_CACHE_V5',
        expected_global_step=2,maximum_model_loads=1,
        **{k:dict(path='/sealed/'+k,sha256='a'*64)for k in
            ('checkpoint_reference','checkpoint_source_spec','trained_probe_reference')})


def test_separately_named_zero_update_profile_does_not_claim_full8_or_formal_training():
    c=config();assert memory.configuration(c)is c and core.full_restore_only(c)
    calls=[n for n in ast.walk(ast.parse(inspect.getsource(memory.run)))if isinstance(n,ast.Call)]
    assert not any(isinstance(n.func,ast.Attribute)and n.func.attr in('train','step','backward')for n in calls)


@pytest.mark.parametrize('change',[dict(formal_training=True),dict(GR00T_updates=2),
    dict(expected_global_step=4),dict(maximum_model_loads=2),dict(original_Trainer_execution=False),
    dict(memory_policy='QUANTIZED_OR_FEATURE_CACHED')])
def test_memory_profile_rejects_new_training_scope_or_endpoint(change):
    with pytest.raises(PermissionError,match='ZERO_UPDATES'):
        memory.configuration(dict(config(),**change))


@pytest.mark.parametrize('bad',[{},dict(path='relative',sha256='a'*64),dict(path='/sealed',sha256='not-a-pin')])
def test_memory_inputs_require_real_absolute_pinned_descriptors(bad):
    with pytest.raises(PermissionError,match='PINNED_REAL'):
        memory.configuration(dict(config(),checkpoint_reference=bad))
