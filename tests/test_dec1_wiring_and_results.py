"""Synthetic CPU transports only: no native safety/physics/model qualification."""
import importlib.util
import itertools
from pathlib import Path
import sys
from copy import deepcopy
import numpy as np
import pytest

PATCH=Path(__file__).resolve().parents[1]
for name,rel in [('go2arx5_vla.adapters.execution_progress_binding','src/go2arx5_vla/adapters/execution_progress_binding.py'),
                 ('scripts.pi0294.dec1.paired_results','scripts/pi0294/dec1/paired_results.py'),
                 ('scripts.pi0294.dec1.source_interface','scripts/pi0294/dec1/source_interface.py')]:
    spec=importlib.util.spec_from_file_location(name,PATCH/rel)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
from go2arx5_vla.adapters.execution_progress_binding import BoundProgressExecutor,NativeReceipt,NativeStopReceipt,NativeShadowReceipt,NativeShadowRejected,NativeShadowUnsupported
from go2arx5_vla.adapters.execution_progress import ActualFeedback
from go2arx5_vla.adapters.groot_to_wbc import ReferenceAdapterConfig
from scripts.pi0294.dec1.paired_results import aggregate,MODEL,CHECKPOINT,R0


def wire(start=0.):
    a=np.zeros((40,10));a[:,0]=start+np.arange(40)*.002
    a[:,3]=1;a[:,7]=1
    a[3,9]=.9;a[4:,9]=.1
    return a


class Fake:
    def __init__(self):
        self.t=0.;self.ack=None;self.calls=[];self.stops=[];self.records=[]
        self.bad_raw=False;self.bad_full=False;self.bad_prefix=False
        self.bad_feedback=False;self.bad_ack=False;self.bad_owner=False;self.throw_dispatch=False
    def feedback(self):
        phase=int(round(self.t/.05));g=0. if phase<3 else .9 if phase==3 else .1
        return ActualFeedback(self.t if not self.bad_feedback else self.t-.2,
            [phase*.002,0,0],[0,0,0,1],g,'world','tcp',self.ack)
    def raw(self,v):
        self.calls.append('raw40');assert v.shape==(40,10) and not v.flags.writeable
        if self.bad_raw:raise PermissionError('RAW40_FAIL')
    def full(self,p):
        self.calls.append('mapped40');assert p.mapped_full.point_count==40
        if self.bad_full:raise PermissionError('MAPPED40_FAIL')
    def shadow(self,p):
        self.calls.append('prefix8');assert len(p.prefix)==8
        return NativeReceipt('ep','source',p.command_id,'shadow:'+p.command_id,199 if self.bad_prefix else 200)
    def dispatch(self,p):
        self.calls.append('dispatch')
        if self.throw_dispatch:raise RuntimeError('TRANSPORT_OUTCOME_UNKNOWN')
        self.t=p.controller_timestamp+.05;self.ack=p.command_id
        return NativeReceipt('ep','other' if self.bad_owner else 'source',
            'other' if self.bad_ack else p.command_id,'live:'+p.command_id,25,self.t)
    def stop(self,reason):
        self.stops.append(reason)
        return NativeStopReceipt('ep','source','stop:ep',True)
    def record(self,v):self.records.append(v)
    def bound(self,mode='I_PROGRESS'):
        return BoundProgressExecutor(ReferenceAdapterConfig('world','world','tcp',stale_timeout_s=3.),
            mode=mode,episode_id='ep',executor_id='source',ticks_per_reference=25,
            validate_raw_eef10=self.raw,validate_mapped_full=self.full,prefix_shadow=self.shadow,
            dispatch_once=self.dispatch,actual_feedback=self.feedback,stop=self.stop,record=self.record)


def load(f,b,start=0.,chunk='chunk'):
    b.load_raw(raw_eef10=wire(start*.04),chunk_id=chunk,observation_timestamp=start,
        point_timestamps=start+np.arange(40)*.05,source_frame='world',tcp_semantic_id='tcp',controller_timestamp=start)


def test_actual_wiring_order_ticks_ack_and_record_before_single_dispatch():
    f=Fake();b=f.bound();load(f,b);p,ack,actual=b.step(controller_timestamp=0.)
    assert f.calls==['raw40','mapped40','prefix8','dispatch']
    assert actual.predecessor_ack_id==ack.command_id==p.command_id
    assert b.state()['live_ticks']==25 and b.state()['shadow_ticks']==200
    assert [r['kind'] for r in f.records]==['RAW_H40_ACCEPTED','PREFIX_SHADOW_INTENT','MAPPED_PREFIX_SHADOW','EXECUTION_INTENT','EXECUTION_ACK']
    assert all(r['episode_id']=='ep' and r['executor_id']=='source' for r in f.records)


@pytest.mark.parametrize('bad', ['bad_raw','bad_full','bad_prefix'])
def test_native_raw_mapped_or_prefix_rejection_never_dispatches_and_stops_once(bad):
    f=Fake();setattr(f,bad,True);b=f.bound()
    with pytest.raises(PermissionError):load(f,b);b.step(controller_timestamp=0.)
    assert 'dispatch' not in f.calls and len(f.stops)==1
    if bad=='bad_prefix':assert b.state()['shadow_ticks']==199
    with pytest.raises(PermissionError,match='LATCHED'):b.step(controller_timestamp=0.)
    assert len(f.stops)==1


def test_stale_feedback_before_mapping_and_dispatch_rejected():
    f=Fake();f.bad_feedback=True;b=f.bound()
    with pytest.raises(PermissionError):load(f,b)
    assert f.calls==['raw40'] and len(f.stops)==1


@pytest.mark.parametrize('bad',['bad_ack','bad_owner','throw_dispatch'])
def test_unknown_dispatch_effect_or_wrong_owner_ack_latches_no_retry(bad):
    f=Fake();b=f.bound();load(f,b);setattr(f,bad,True)
    with pytest.raises((PermissionError,RuntimeError)):b.step(controller_timestamp=0.)
    assert f.calls.count('dispatch')==1 and b.state()['unknown_dispatches']==1
    with pytest.raises(PermissionError):b.step(controller_timestamp=.05)
    assert f.calls.count('dispatch')==1 and len(f.stops)==1


def test_out_of_order_feedback_after_actual_ack_stops_and_ticks_stay_charged():
    f=Fake();b=f.bound();load(f,b)
    dispatch=f.dispatch
    def stale(p):
        r=dispatch(p);f.bad_feedback=True;return r
    b.dispatch=stale
    with pytest.raises(PermissionError):b.step(controller_timestamp=0.)
    assert b.state()['live_ticks']==25 and b.state()['unknown_dispatches']==0 and len(f.stops)==1


def test_new_chunk_reset_gripper_order_and_mode_shared_values():
    a=Fake();c=Fake();x=a.bound('I_TIME');y=c.bound('I_PROGRESS');load(a,x);load(c,y)
    grips=[]
    for k in range(8):
        p,_,_=x.step(controller_timestamp=k*.05);q,_,_=y.step(controller_timestamp=k*.05)
        assert p.phase==pytest.approx(q.phase,abs=1e-7)
        assert np.allclose(p.preview.xyz_rot6d,q.preview.xyz_rot6d,atol=1e-9,rtol=0)
        assert p.gripper.normalized_command==q.gripper.normalized_command
        grips.append(p.gripper.normalized_command)
    assert grips[:5]==[0.,0.,0.,.9,.1]
    load(a,x,.4,'new');load(c,y,.4,'new')
    assert x.state()['adapter']['phase'] is None and y.state()['adapter']['phase'] is None
    p,_,_=x.step(controller_timestamp=.4);q,_,_=y.step(controller_timestamp=.4)
    assert p.phase==pytest.approx(0) and q.phase==pytest.approx(0)


def test_duplicate_chunk_rejected_and_no_extra_dispatch():
    f=Fake();b=f.bound();load(f,b)
    with pytest.raises(PermissionError):load(f,b)
    assert 'dispatch' not in f.calls and len(f.stops)==1


def test_changed_frozen_preview_parameters_rejected():
    f=Fake()
    with pytest.raises(ValueError,match='FROZEN'):
        BoundProgressExecutor(ReferenceAdapterConfig('world','world','tcp',stale_timeout_s=3.,lambda_m_per_rad=.2),
            mode='I_TIME',episode_id='ep',executor_id='source',ticks_per_reference=25,
            validate_raw_eef10=f.raw,validate_mapped_full=f.full,prefix_shadow=f.shadow,
            dispatch_once=f.dispatch,actual_feedback=f.feedback,stop=f.stop,record=f.record)


GEOMS={i:f'GEOMETRY_{i}' for i in range(1,9)}
def rows():
    return [dict(geometry_slot_not_sealed=str(i),task=t,executor=e,interface=m,
        model=MODEL,checkpoint_manifest_sha256=CHECKPOINT,statistics_sha256=R0,geometry_identity=GEOMS[i],noise_identity='SEALED_NOISE',
        native_qualification='QUALIFIED',status='SUCCESS' if e=='E_TARGET' and m=='I_PROGRESS' else 'VALID_TASK_FAILURE',
        task_success='true' if e=='E_TARGET' and m=='I_PROGRESS' else 'false')
        for i,t,e,m in itertools.product(range(1,9),('T1','T2'),('E_SOURCE','E_TARGET'),('I_TIME','I_PROGRESS'))]


def result(r):return aggregate(r,geometry_identities=GEOMS,noise_identity='SEALED_NOISE',bootstrap_draws=200)


def test_result_eight_geometry_blocks_are_resampling_unit_not64_cells():
    v=result(rows());assert v['planned_cells']==64 and v['independent_geometry_blocks']==8
    assert len(v['geometry_block_differences'])==8 and all(b==[0.,1.,1.] for b in v['geometry_block_differences'])
    assert v['effect_estimates']==dict(Delta_source=0.,Delta_target=1.,interaction=1.)
    assert v['resampling']['blocks']==8 and v['intervals']['Delta_target']==[1.,1.]


@pytest.mark.parametrize('mutation', ['missing_row','duplicate','missing_key','model','noise','geometry','status','outcome','qualification','checkpoint_digest','statistics_digest'])
def test_strict_result_identity_missingness_and_qualification_rejections(mutation):
    r=rows()
    if mutation=='missing_row':r.pop()
    elif mutation=='duplicate':r[-1]=deepcopy(r[0])
    elif mutation=='missing_key':r[0].pop('task_success')
    elif mutation=='model':r[0]['model']='A_CONTROL_AT4096'
    elif mutation=='noise':r[0]['noise_identity']='OTHER_BANK'
    elif mutation=='geometry':r[0]['geometry_identity']='ANOTHER_SCENE'
    elif mutation=='status':r[0]['status']='PASSED'
    elif mutation=='outcome':r[0]['task_success']='true'
    elif mutation=='qualification':r[0]['native_qualification']='NOT_QUALIFIED'
    elif mutation=='checkpoint_digest':r[0]['checkpoint_manifest_sha256']='WRONG_WEIGHTS_SAME_LABEL'
    elif mutation=='statistics_digest':r[0]['statistics_sha256']='R1'
    with pytest.raises(ValueError):result(r)


@pytest.mark.parametrize('status',['NOT_RUN','ENGINEERING_MISSING'])
def test_explicit_missing_and_unrun_rows_remain_unknown_not_zero_success_rate(status):
    r=rows();r[0].update(status=status,task_success='UNKNOWN',native_qualification='NOT_QUALIFIED')
    v=result(r)
    assert not v['complete_for_paired_inference'] and v['effect_estimates'] is None
    assert v['by_task_executor_interface']['T1/E_SOURCE/I_TIME']['success_rate'] is None
    assert v['status_counts'][status]==1


def test_native_fault_distinct_from_valid_task_or_reference_failure():
    r=rows();r[0].update(status='NATIVE_SAFETY_ERROR',task_success='false')
    v=result(r);assert v['status_counts']['NATIVE_SAFETY_ERROR']==1 and v['effect_estimates'] is None
    for status in ['REFERENCE_REJECT','S1_REJECT']:
        r=rows();r[0].update(status=status)
        v=result(r);assert v['status_counts'][status]==1 and v['complete_for_paired_inference']


def test_stop_callback_must_confirm_actual_stop_and_stays_latched():
    f=Fake();f.bad_full=True;b=f.bound();load(f,b)
    b.stop=lambda reason:None
    with pytest.raises(PermissionError,match='STOP_NOT_ACKNOWLEDGED'):b.step(controller_timestamp=0.)
    assert b.state()['stopped'] and not b.state()['stop_confirmed']
    with pytest.raises(PermissionError,match='LATCHED'):b.step(controller_timestamp=.05)
    assert 'dispatch' not in f.calls


def test_source_original_full_h40_gate_used_without_new_safety_criteria():
    from scripts.pi0294.dec1.source_interface import full_h40_validators
    good=wire();events=[]
    raw,mapped=full_h40_validators(actual_state10=lambda:good[0],observation_time=lambda:0.,record_gate=events.append)
    raw(good)
    bad=good.copy();bad[-1,0]+=2.
    with pytest.raises(PermissionError,match='ORIGINAL_FULL_H40'):raw(bad)
    assert len(events)==2 and events[1]['result']['decision']=='REJECT'


def test_source_preview_shadow_same_anchor_different_lookahead_never_reuses_single_pose_pass(monkeypatch):
    from scripts.pi0294.dec1.source_interface import SourcePrefixShadow
    from scripts.pi0294 import s1_preflight as native
    from dataclasses import replace
    calls=[]
    def forbidden(*args,**kwargs):
        calls.append('SINGLE_POSE_PREDICTOR');raise AssertionError('MUST_NOT_BE_USED_FOR_PREVIEW9')
    monkeypatch.setattr(native,'predict',forbidden)
    f=Fake();b=f.bound();load(f,b)
    p=b._adapter.propose(f.feedback(),controller_timestamp=0.)
    different=p.preview.eef_position.copy();different[-1,1]+=.5
    q=replace(p,preview=replace(p.preview,eef_position=different))
    assert np.array_equal(p.preview.eef_position[0],q.preview.eef_position[0])
    assert not np.array_equal(p.preview.eef_position[1:],q.preview.eef_position[1:])
    shadow=SourcePrefixShadow(object(),episode_id='ep',executor_id='source',guard=lambda:None,
        charge=lambda *a,**k:None,record=lambda v:None)
    assert not shadow.supported
    for proposal in (p,q):
        receipt=shadow(proposal)
        assert receipt.decision=='NOT_SUPPORTED' and receipt.ticks==0
        with pytest.raises(NativeShadowUnsupported):shadow.bind_before_dispatch(proposal)
    assert calls==[]


def test_bound_source_unsupported_preview_does_not_dispatch_and_is_known_zero_effect():
    from scripts.pi0294.dec1.source_interface import SourcePrefixShadow
    f=Fake();b=f.bound();load(f,b)
    b.shadow=SourcePrefixShadow(object(),episode_id='ep',executor_id='source',guard=lambda:None,
        charge=lambda *a,**k:None,record=lambda v:None)
    with pytest.raises(NativeShadowUnsupported):b.step(controller_timestamp=0.)
    assert 'dispatch' not in f.calls and b.state()['shadow_ticks']==0
    assert b.state()['unknown_shadows']==0 and b.state()['unsupported_shadows']==1
    assert b.state()['last_shadow_decision']=='NOT_SUPPORTED' and b.state()['stop_confirmed']


@pytest.mark.parametrize('ticks',[0,37,199])
def test_known_native_s1_reject_preserves_partial_actual_ticks_without_unknown(ticks):
    f=Fake();b=f.bound();load(f,b)
    def reject(proposal):
        receipt=NativeShadowReceipt('ep','source',proposal.command_id,'native-reject',ticks,decision='REJECT')
        raise NativeShadowRejected(receipt)
    b.shadow=reject
    with pytest.raises(NativeShadowRejected):b.step(controller_timestamp=0.)
    s=b.state()
    assert s['shadow_ticks']==ticks and s['unknown_shadows']==0 and s['known_shadow_rejections']==1
    assert s['last_shadow_decision']=='REJECT' and 'dispatch' not in f.calls and len(f.stops)==1
    assert any(r['kind']=='MAPPED_PREFIX_SHADOW' and r['ticks']==ticks and r['decision']=='REJECT' for r in f.records)


def test_unreceipted_shadow_exception_remains_unknown_no_false_known_reject():
    f=Fake();b=f.bound();load(f,b)
    def error(p):raise RuntimeError('SHADOW_OUTCOME_UNAVAILABLE')
    b.shadow=error
    with pytest.raises(RuntimeError):b.step(controller_timestamp=0.)
    s=b.state();assert s['unknown_shadows']==1 and s['known_shadow_rejections']==0
    assert s['shadow_ticks']==0 and 'dispatch' not in f.calls


@pytest.mark.parametrize('result',[None,False,{'decision':False},{'decision':'PASS'}])
def test_source_full_h40_gate_none_or_wrong_decision_is_rejected(monkeypatch,result):
    from scripts.pi0294.dec1 import source_interface as source
    monkeypatch.setattr(source,'gate_action',lambda *a:result)
    raw,_=source.full_h40_validators(actual_state10=lambda:wire()[0],observation_time=lambda:0.,record_gate=lambda r:None)
    with pytest.raises(PermissionError,match='MALFORMED'):raw(wire())
