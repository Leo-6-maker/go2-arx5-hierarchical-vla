"""Synthetic fixtures for actual-ACK collection, no native physics."""
from copy import deepcopy
import pytest
from scripts.pi0294.dec1.test_source_single_pose_prefix import harness
from scripts.pi0294.dec1.test_source_episode_wire import outer,start,advance


@pytest.mark.parametrize('mode',['I_TIME','I_PROGRESS'])
def test_wire_proof_collects_only_completed_actual25_tick_ACKs(harness,mode):
    h=outer(harness,mode);prepared=start(h);forecast=deepcopy(h.executor.pending['forecast']);rows=[]
    assert h.executor.outer['actual_acks']==[]
    for reference in range(8):
        target,opening=h.executor.reference_action(reference)
        for tick in range(25):
            assert len(h.executor.outer['actual_acks'])==reference
            advance(h,target,opening);rows.append(h.observed(h.engine))
        actual=h.executor.outer['actual_acks'][-1]
        assert actual['reference']==reference and actual['tick_before']==reference*25
        assert actual['tick_after']==(reference+1)*25
        assert actual['ack_id']==actual['command_id']==h.executor.adapter.state()['predecessor_ack_id']
        assert actual['predecessor_ack_id']==(None if reference==0 else h.executor.outer['actual_acks'][-2]['command_id'])
    proof=h.executor.complete_prefix(h.engine,rows)
    assert proof['s1']==forecast and proof['nonce']==prepared['nonce']
    assert len(proof['actual_reference_acks'])==8
    assert [r['proposal_sha256'] for r in proof['actual_reference_acks']]==proof['proposal_digests']
    assert proof['maximum_live_shadow_difference']==0.
    assert proof['episode_id']=='CPU_SOURCE_EPISODE' and proof['executor_id']=='CPU_SINGLE_POSE'


def test_missing_actual_ACK_record_is_not_filled_from_shadow_predictions(harness):
    h=outer(harness);start(h);rows=[]
    for reference in range(8):
        target,opening=h.executor.reference_action(reference)
        for tick in range(25):advance(h,target,opening);rows.append(h.observed(h.engine))
    h.executor.outer['actual_acks'].pop()
    with pytest.raises(PermissionError,match='EXACT_PREFIX200_ACK_REQUIRED'):
        h.executor.complete_prefix(h.engine,rows)
    assert h.engine.steps==200 and len(h.clones)==1 and len(h.stops)==1
