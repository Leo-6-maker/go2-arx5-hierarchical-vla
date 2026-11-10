from collections import Counter

from go2arx5_vla.world_model.task_learning_core_local_v1 import sample_at, sampling_tables
from go2arx5_vla.world_model.task_learning_features_v1 import sample_index


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
