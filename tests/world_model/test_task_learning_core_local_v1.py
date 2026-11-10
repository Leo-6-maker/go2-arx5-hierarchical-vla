from collections import Counter

from go2arx5_vla.world_model.task_learning_core_local_v1 import sample_at, sampling_tables
from go2arx5_vla.world_model.task_learning_features_v1 import sample_index
from go2arx5_vla.world_model.task_learning_core_local_v1 import training_activation_policy


def test_original_exposure_and_event_quarter_have_fixed_denominators():
    original=list(range(3200));events=[10,20,30]
    seed=20261109
    assert [sample_at(i,'ORIGINAL',original,events,seed)for i in range(3200)] == [
        original[sample_index(i,3200,seed)]for i in range(3200)]
    draws=[sample_at(i,'CONTACT_EVENTS',original,events,seed)for i in range(128)]
    assert all(draws[i]in events for i in range(3,128,4))
    assert [draws[i]for i in range(128)if i%4!=3] == [
        original[sample_index(i,3200,seed)]for i in range(96)]
    assert [sample_at(i,'CONTACT_EVENTS',original,events,seed)for i in range(64,128)]==draws[64:]


def test_exposure_repeats_are_never_new_windows():
    cache={'rows':[dict(episode=f'e{i//15}',anchor=i%15,index=i)for i in range(1450)]}
    episodes=[]
    for index in range(100):
        anchors=[r['anchor']for r in cache['rows']if r['episode']==f'e{index}']
        if not anchors:continue
        episodes.append(dict(id=f'e{index}',split='TRAIN',optimization_anchors=anchors))
    # Construct3200 exposure entries without manufacturing new identities.
    episodes[0]['optimization_anchors'] += [0]*1750
    events={'windows':[dict(episode='e0',anchor=0,actual_contact_onset=True)]}
    original,event=sampling_tables(cache,{'episodes':episodes},events)
    assert len(original)==3200 and len(set(original))==1450
    assert event==[0] and Counter(original)[0]==1751


def test_resident_execution_has_no_activation_offload_and_keeps_other_storage_rules():
    old=training_activation_policy(dict(training_activation_storage='CPU_OFFLOAD_2G_V1',
        operation='QUALIFY_GPU_RESIDENT',qualification_only=True))
    resident=training_activation_policy({})
    assert old['per_microbatch_activation_offload_bytes']==2*1024**3
    assert resident['activation_storage']=='CUDA_RESIDENT'
    assert resident['per_microbatch_activation_offload_bytes']==0
    assert resident['quantization'] is False and resident['gradient_checkpointing'] is False
    assert resident['parameter_storage_offload'] is False


def test_unknown_storage_execution_cannot_silently_fall_back_to_offload():
    import pytest
    with pytest.raises(ValueError,match='DECLARED_TRAINING_ACTIVATION_STORAGE'):
        training_activation_policy({'training_activation_storage':'UNKNOWN'})


def test_candidate_updates_cannot_request_cpu_activation_offload():
    import pytest
    with pytest.raises(ValueError,match='CPU_OFFLOAD_ONLY_FOR_NONCANDIDATE_COMPARISON'):
        training_activation_policy({'training_activation_storage':'CPU_OFFLOAD_2G_V1','operation':'TRAIN'})
