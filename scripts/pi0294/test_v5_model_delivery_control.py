"""CPU byte/escrow checks; fixtures do not qualify native IO or a GPU."""
import base64
from copy import deepcopy
import json
import pytest
from scripts.pi0294.test_v5_evaluation_transfer import source
from scripts.pi0294.test_v5_core_stage_accounting import closed
from scripts.pi0294.test_v5_closed_training_handoff import fixture
from scripts.pi0294.pv1 import mix_delivery_control as d, mix_accounting as a
from scripts.pi0294.pv1 import mix_stage_handoff as h


@pytest.fixture(autouse=True)
def synthetic_legacy_header(monkeypatch):
    # Focused fixtures omit the old campaign's hundreds of historical fields.
    # Keep every C/D typed validator; native captured-byte test runs full check.
    monkeypatch.setattr(d.c,'checked',a.validate_global)


def inputs(source):
    g=json.loads(base64.b64decode(json.loads(source['capture'])['global_base64']))
    _,l, _=fixture()
    pool=d.s.whole_package()
    l['CPU_reserved_s'].update(CoreLab=pool['Core_CPU_s'],local_PC=pool['local_CPU_s'])
    l['CPU_s'].update(CoreLab=0.,local_PC=0.)
    l['network_transfer_accounted_bytes']=0
    # The synthetic transfer uses a different complete endpoint from closed().
    capture=json.loads(source['capture']);terminal={k:base64.b64decode(v)for k,v in capture['terminal_capture_base64'].items()}
    l.pop('V5_CD_terminal_training_mirrors',None)
    l=h.project_training_to_LOCAL(local_bytes=a.raw(l),global_bytes=a.raw(g),terminal_capture=terminal,arm='C50')['proposed_LOCAL']
    from scripts.pi0294.test_v5_core_stage_accounting import paid
    pg,pl,_=paid();field='V5_CD_existing_Core_GPU_CPU_projection'
    for v in (g,l):
        v[field]=deepcopy(pg[field]);v[a.PACKAGE][field]=deepcopy(pg[field])
        v[a.PACKAGE]['remaining_resources']['Core_CPU_s']+=160000
    g['unknown_effects']=g['pending_updates']=[]
    l['unknown_effects']=l['pending_updates']=l['pending_queries']=[]
    # Rebind the capture to the actual revised GLOBAL bytes.
    capture['global_base64']=base64.b64encode(a.raw(g)).decode()
    capture['handoff']=h.closed_endpoint(global_bytes=a.raw(g),arm='C50',**terminal)
    small=json.loads(source['small']);small['training_capture_sha256']=a.sha(a.raw(capture));small['training_handoff_sha256']=a.sha(a.raw(capture['handoff']))
    return g,l,a.raw(capture),a.raw(small)


def test_one_control_prepay_no_extra_budget_or_stage_completion(source):
    g,l,cap,small=inputs(source);before=deepcopy((g,l))
    result=d.prepay(g,l,capture_bytes=cap,small_bytes=small,arm='C50',now_epoch=1200.)
    v,w=result['proposed_global'],result['proposed_LOCAL']
    assert (g,l)==before and v[a.PACKAGE]==w[a.PACKAGE]
    assert result['delivery_marker']['additional_reservation']=={}
    assert v['reserved']==g['reserved'] and v['PBS_submissions']==g['PBS_submissions']
    assert v[a.PACKAGE]['completed_stages']==['TRAIN:C50:1024']
    plan=d.t.prepare(cap,small,arm='C50')
    proof=d.upload_admission(v,plan,now_epoch=1201.,checkpoint_complete_bytes=base64.b64decode(plan['private_restoration']['original_complete_base64']))
    assert proof['new_GPU_PBS_queries_updates']==0 and not proof['model_or_physics_admission']
    with pytest.raises(PermissionError):d.prepay(v,w,capture_bytes=cap,small_bytes=small,arm='C50',now_epoch=1202.)


@pytest.mark.parametrize('case',range(8))
def test_no_old_arm_changed_tensor_missing_prepay_expired_or_unclosed_upload(source,case):
    g,l,cap,small=inputs(source)
    v=d.prepay(g,l,capture_bytes=cap,small_bytes=small,arm='C50',now_epoch=1200.)['proposed_global']
    plan=d.t.prepare(cap,small,arm='C50');cp=base64.b64decode(plan['private_restoration']['original_complete_base64']);now=1201.
    if case==0:plan['public_plan']['candidate']='B_COVERAGE'
    if case==1:next(iter(plan['public_plan']['files'].values()))['sha256']='0'*64
    if case==2:v[a.PACKAGE]['prepaid_control'].pop('EVAL:C50:N294101')
    if case==3:now=v['V5_CD_successor_scope']['deadline_epoch']
    if case==4:cp+=b' '
    if case==5:v['unknown_effects']=[dict(kind='UNKNOWN')]
    if case==6:v[a.PACKAGE]['completed_stages'].append('EVAL:C50:N294101')
    if case==7:plan['public_plan']['checkpoint_step']=4096
    with pytest.raises((PermissionError,KeyError)):
        d.upload_admission(v,plan,now_epoch=now,checkpoint_complete_bytes=cp)
