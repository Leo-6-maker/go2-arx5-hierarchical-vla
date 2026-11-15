import copy
import numpy as np
import pytest
from go2arx5_vla.world_model.sla_label_replay_v61 import anchor_comparison,STATE_ATOL


def inputs():
    actual=dict(tcp=np.eye(4),qpos=np.zeros(34),qvel=np.zeros(32),native_step=5703)
    history=dict(actual_state_history=np.zeros((9,106)),past_consumed_EEF10_BASE7=np.zeros((8,17)))
    record=dict(tcp=np.eye(4),qpos=np.zeros(34),qvel=np.zeros(32),reference=np.eye(4),
        history=np.zeros((9,106)),past_references=np.zeros((8,17)),base=np.zeros(7),native_step=5703)
    return actual,history,record


def test_actual_clock_state_and_consumed_history_match():
    a,h,r=inputs()
    result=anchor_comparison(a,np.eye(4),h,np.zeros(7),r)
    assert result['passes'] and not result['hidden_state_restored']
    assert result['actual_original_prefix_replayed'] and not result['label_reanchored']


@pytest.mark.parametrize('part',['tcp','qvel','history','past_references','reference','clock'])
def test_qpos_only_match_never_admits_changed_control_history(part):
    a,h,r=inputs()
    if part=='clock':a['native_step']+=25
    elif part in ('tcp','qvel'):a[part].flat[0]+=2*STATE_ATOL
    elif part=='history':h['actual_state_history'][0,0]+=2*STATE_ATOL
    elif part=='past_references':h['past_consumed_EEF10_BASE7'][0,0]+=2*STATE_ATOL
    elif part=='reference':r['reference'][0,3]+=2*STATE_ATOL
    assert not anchor_comparison(a,np.eye(4),h,np.zeros(7),r)['passes']


def test_archived_builder_requires_closed_producer_and_actual_bytes(tmp_path,monkeypatch):
    import json,hashlib
    from go2arx5_vla.world_model import sla_label_replay_v61 as replay,corelab_expert_semantics_v3 as expert
    from go2arx5_vla.world_model.task_learning_scene_v1 import MEMBER
    from go2arx5_vla.world_model.storage import digest
    source=tmp_path/'source';builder=source/MEMBER;builder.parent.mkdir(parents=True);builder.write_text('immutable archived builder')
    run=tmp_path/'original';(run/'control').mkdir(parents=True)
    original=dict(registered_root=str(run),source_root=str(source),source_sha='a'*40,source_files={MEMBER:digest(builder)})
    spec=run/'EXECUTION_SPEC.json';spec.write_text(json.dumps(original));pin=digest(spec)
    lease=run/'control/LEASE.json';lease.write_text(json.dumps(dict(status='CLOSED',worker_exit_code=0,spec_sha256=pin,source_sha='a'*40)))
    monkeypatch.setattr(expert,'native_scene_admission',lambda value:dict(old_builder=value['source_files'][MEMBER]))
    proof,actual_builder=replay._archived_scene(str(spec),pin)
    assert proof['old_builder']==actual_builder==digest(builder)
    # Changed bytes at a new transfer boundary never inherit cached custody.
    replay._archived_scene.cache_clear();builder.write_text('changed builder')
    with pytest.raises(ValueError,match='ACTUAL_BUILDER_CUSTODY'):replay._archived_scene(str(spec),pin)
    builder.write_text('immutable archived builder');lease.write_text(json.dumps(dict(status='ACTIVE',worker_exit_code=0,spec_sha256=pin,source_sha='a'*40)))
    with pytest.raises(ValueError,match='CLOSED_PRODUCER'):replay._archived_scene(str(spec),pin)
