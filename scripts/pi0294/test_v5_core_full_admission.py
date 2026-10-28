"""Full-budget CPU fixtures; never native GPU/endpoint qualification."""
import base64,json
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.test_pv1_core_evaluation_accounting import captured_global
from scripts.pi0294.test_v5_core_stage_accounting import paid,record,physical,started
from scripts.pi0294.test_v5_core_native_identity import request as identity_request
from scripts.pi0294.test_pv1_core_evaluation_identity import DEVICE,GPU
from scripts.pi0294.pv1 import mix_core_admission as n,mix_core_accounting as c
from scripts.pi0294.pv1 import mix_accounting as a,successor_cd as s,mix_stage_handoff as h
from scripts.pi0294.pv1 import mix_core_context as context,mix_evaluation as plan,evaluation_core_identity as original
from scripts.pi0294.pv1.evaluation_admission import resource_floor

@pytest.fixture
def full(captured_global):
    repo=Path(__file__).resolve().parents[4]
    private=repo/'work/private/v5-execution-20261025/C50_BOOTSTRAP_RETRY_PREPARED.private.json'
    localfile=repo/'work/private/v5-execution-20261025/C50_CORE_CONTROL_PREPAY.private.json'
    if not private.exists()or not localfile.exists():pytest.skip('private actual population/LOCAL absent')
    population=base64.b64decode(json.loads(private.read_bytes())['small']['plans/PV1_DATA_PLAN_AND_SPLIT_SEAL.json'])
    small,local,capture=paid();closed_global=a.raw(small);proof=h.closed_endpoint(global_bytes=closed_global,arm='C50',**capture)
    tc=a.raw(dict(schema='V5_CD_REAL_TRAINING_TERMINAL_BYTES_V1',arm='C50',global_base64=base64.b64encode(closed_global).decode(),handoff=proof,terminal_capture_base64={k:base64.b64encode(b).decode()for k,b in capture.items()}))
    service=a.raw(plan.candidate_plan(json.loads(population)['groups'],'C50'));prior=a.raw([])
    row=record(small,capture);m,r=identity_request()
    row.update(GPU_uuid=GPU,planned_panel_sha256=a.sha(service))
    r.update({k:row[k]for k in ('request_id','submission_nonce','source_sha','evaluation_candidate','checkpoint_manifest_sha256','training_handoff_sha256','planned_panel_sha256','cold_input_audit_sha256','created_epoch')})
    r.update(run_uuid=row['child_run_uuid'],native_unit=row['child_run_uuid']+'-neural.service')
    m.update(run_uuid=r['run_uuid']);m['code']['vla_commit']=r['source_sha']
    m['execution']['sealed_inputs']={context.CAPTURE:a.sha(tc),'plans/PV1_DATA_PLAN_AND_SPLIT_SEAL.json':a.sha(population),'plans/SOURCE_EVALUATION_PLAN.json':a.sha(service),'plans/PRIOR_EVALUATION_OUTCOMES.json':a.sha(prior)}
    mb=a.raw(m);r['run_manifest_sha256']=a.sha(mb);rb=a.raw(r);row.update(run_manifest_sha256=a.sha(mb),request_sha256=a.sha(rb))
    p=c.register(small,local,row,a.raw(physical(small)),terminal_capture=capture,now_epoch=1202.)
    process=started(row);process['kernel_limits']=original.kernel_limits('400000 100000',str(48*1024**3),'64')
    process['native_unit_properties']=dict(MainPID=str(process['pid']),ActiveState='active',RuntimeMaxUSec='8h',ControlGroup='/fixture')
    process['native_GPU_raw']=dict(devices=dict(stdout=DEVICE,returncode=0),processes=dict(stdout='',returncode=0));process['device']=original.parse_GPU_rows(DEVICE,'',GPU_uuid=GPU)
    p=c.bind(p['proposed_global'],p['proposed_LOCAL'],request_id=row['request_id'],process=process,now_epoch=1205.)
    small,local=p['proposed_global'],p['proposed_LOCAL'];g=deepcopy(captured_global)
    for k in (s.FIELD,a.PACKAGE,c.FIELD):g[k]=deepcopy(small[k])
    for k in ('used','reserved'):
        for name,value in small[k].items():g[k][name]+=value
    for k in ('CPU_s','CPU_reserved_s','storage_reserved_bytes','new_storage_current_bytes','optimizer_updates','generation_queries','model_loads'):g[k]=g.get(k,0)+small[k]
    g['new_storage_peak_bytes']=max(g['new_storage_current_bytes'],g['new_storage_peak_bytes']);g['PBS_submissions']+=small['PBS_submissions'];c.checked(g)
    l=json.loads(base64.b64decode(json.loads(localfile.read_bytes())['value']['LOCAL_base64']));old=deepcopy(l)
    for k,v in local.items():l[k]=deepcopy(v)
    for site,v in old['CPU_s'].items():l['CPU_s'][site]+=v
    l['GPU_allocated_s']+=old['GPU_allocated_s'];l['PBS_submissions']=old['PBS_submissions']+local['PBS_submissions']
    l['optimizer_updates']+=old['optimizer_updates'];l['neural_queries']+=old['neural_queries']
    c.paired(g,l);used,reserved=resource_floor(g,l);used['Core_storage_bytes']=max(used['Core_storage_bytes'],335287617560);used['local_storage_bytes']=9126805504
    snap=dict(schema=n.SNAPSHOT,task=s.TASK,scope_grant_sha256=s.GRANT_SHA,T_CD=r['T_CD'],deadline_epoch=r['deadline_epoch'],GLOBAL_sha256=a.sha(a.raw(g)),LOCAL_sha256=a.sha(a.raw(l)),observed_epoch=1205.,heartbeat_only_raw_ledger=False,used=used,reserved=reserved,Core_storage_upper_bytes=335287617560,LOCAL_storage_upper_bytes=9126805504)
    return dict(manifest_bytes=mb,request_bytes=rb,global_bytes=a.raw(g),local_bytes=a.raw(l),snapshot_bytes=a.raw(snap),training_capture_bytes=tc,population_bytes=population,plan_bytes=service,prior_bytes=prior,native_facts_bytes=a.raw(process),now_epoch=1205.)

def test_full_raw_closed_training_process_and_remaining_package_join(full):
    v=n.recompute(**full)
    assert v['all_remaining_costs_fit']and not v['shortfalls']and not v['model_permission']and not v['physics_authorized']
    assert v['dimensions']['H200_s']['matrix']==86580 and v['dimensions']['H200_s']['final']==72000
    assert v['dimensions']['PBS_submissions']['matrix']==3 and v['allocation']['new_optimizer_updates']==0

def edit(x,key,fn):
    v=json.loads(x[key]);fn(v);x[key]=a.raw(v)

@pytest.mark.parametrize('case',range(12))
def test_changed_or_partial_actual_proofs_never_get_resource_GO(full,case):
    if case==0:edit(full,'snapshot_bytes',lambda v:v.update(observed_epoch=1000.))
    if case==1:edit(full,'snapshot_bytes',lambda v:v.update(GLOBAL_sha256='0'*64))
    if case==2:edit(full,'snapshot_bytes',lambda v:v.update(heartbeat_only_raw_ledger=True))
    if case==3:edit(full,'snapshot_bytes',lambda v:v['used'].update(Hopper_storage_bytes=0))
    if case==4:edit(full,'snapshot_bytes',lambda v:v['reserved'].update(H200_s=0))
    if case==5:edit(full,'snapshot_bytes',lambda v:v['used'].update(H200_s=121*3600))
    if case==6:edit(full,'native_facts_bytes',lambda v:v.update(observed_epoch=1000.))
    if case==7:edit(full,'native_facts_bytes',lambda v:v['native_unit_properties'].update(MainPID='999'))
    if case==8:edit(full,'native_facts_bytes',lambda v:v['native_GPU_raw']['processes'].update(stdout=GPU+', 999, independent, 9000\n'))
    if case==9:edit(full,'training_capture_bytes',lambda v:v['terminal_capture_base64'].pop('OS_exit_bytes'))
    if case==10:edit(full,'request_bytes',lambda v:v.update(evaluation_candidate='B_COVERAGE',checkpoint_step=4096))
    if case==11:edit(full,'request_bytes',lambda v:v.update(shared_GPU_grant_sha256='a'*64))
    with pytest.raises((PermissionError,KeyError)):n.recompute(**full)
