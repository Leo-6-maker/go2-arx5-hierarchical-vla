from copy import deepcopy
import hashlib,json
import pytest
from scripts.pi0294.multitask.source_collection_plan import plan,require_dependency_closed,require_two_recovery_pilots
from scripts.pi0294.multitask.book_late_recovery import KIND,conditions


def test_new_collection_cannot_start_at_train_completion_or_borrow_active_owner(tmp_path):
    def put(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return hashlib.sha256(path.read_bytes()).hexdigest()
    root=tmp_path/'manager';producer=tmp_path/'producer';seal=put(root/'CAPSULE_SEAL.json',dict(files={}))
    put(root/'QUEUE.json',dict(producer_root=str(producer)))
    state=dict(status='RUNNING',TRAIN=dict(status='CLOSED_AUDITED',step=4096),DEV=dict(status='PENDING'),diagnostics=dict(uniform='PENDING',phase='PENDING'))
    put(root/'STATE.json',state)
    with pytest.raises(PermissionError,match='ENTIRE_OLD_CHAIN'):require_dependency_closed(root,expected_seal_sha256=seal,unit_reader=lambda u:dict(MainPID='0'))
    state.update(status='COMPLETE',DEV=dict(status='CLOSED_AUDITED'),diagnostics=dict(uniform='CLOSED_AUDITED_96_GENERATIONS',phase='CLOSED_AUDITED_96_GENERATIONS'));put(root/'STATE.json',state)
    trainpin=put(producer/'control/LEASE.json',dict(status='CLOSED'));put(root/'BOOK_PHASE_B2_S4096_COMPLETED_CLOSED.private.json',dict(lease_sha256=trainpin))
    for name in('BOOK_PHASE_TRAIN32_UNIFORM4096','BOOK_PHASE_TRAIN32_PHASE4096','BOOK_PHASE4096_DEV10'):
        actor=tmp_path/name;pin=put(actor/'control/LEASE.json',dict(status='CLOSED'))
        put(root/(name+'_STAGED.private.json'),dict(root=str(actor)))
        put(root/(name+'_COMPLETED_CLOSED.private.json'),dict(lease_sha256=pin));put(root/(name+'_FINAL_INDEPENDENT.private.json'),dict(lease_sha256=pin))
    put(root/'steps/DEV_closed_read.json',dict(status='COMPLETE'))
    with pytest.raises(PermissionError,match='STILL_LIVE'):require_dependency_closed(root,expected_seal_sha256=seal,unit_reader=lambda u:dict(MainPID='17'))
    result=require_dependency_closed(root,expected_seal_sha256=seal,unit_reader=lambda u:dict(MainPID='0'))
    assert result['original_STOP_retained']and result['new_native_reset_camera_safety_qualification_still_required']
    put(root/'steps/DEV_closed_read.json',dict(status='START_INTENT'))
    with pytest.raises(PermissionError,match='UNRESOLVED_OLD_STEP'):require_dependency_closed(root,expected_seal_sha256=seal,unit_reader=lambda u:dict(MainPID='0'))


def test_two_recovery_pilots_require_real_handover_and_task_result_before_expansion():
    r=dict(scientific_kind=KIND,status='COMPLETE_LATE_TRAIN_RECOVERY_PANEL',formal_DEV_conditions=0,
        conditions=[dict(id=c['id'],qualified_expert_recovery=True,success=True,hard_native_safety=False,late_recovery=dict(takeover_index=80))for c in conditions('pilot')])
    assert require_two_recovery_pilots(r)
    for key,value in(('qualified_expert_recovery',False),('success',False),('hard_native_safety',True),('late_recovery',dict(takeover_index=None))):
        bad=deepcopy(r);bad['conditions'][0][key]=value
        with pytest.raises(PermissionError):require_two_recovery_pilots(bad)
    p=plan();assert p['active_queue_hot_edit']is False and p['first_attempts_max']['total']==18
    assert p['no_new_model_training_in_this_queue']is True
