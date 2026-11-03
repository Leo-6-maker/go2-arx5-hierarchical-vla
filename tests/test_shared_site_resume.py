from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json,hashlib
import pytest
from scripts.pi0294.multitask.shared_site import validate_vanda,site_paths,project_quota,quota_from_hpc_space,PROJECT_SCRATCH
from scripts.pi0294.multitask import shared_resume,shared_training
from scripts.pi0294.multitask.activation_offload import RESIDENT_ACTIVATION_POLICY,SavedActivationOffload

def test_resident_activations_preserve_loss_gradient_update_rng_without_cpu_copy():
    import torch
    torch.manual_seed(112)
    a=torch.nn.Linear(3,2).double();b=deepcopy(a);x=torch.randn(4,3,dtype=torch.float64)
    off=SavedActivationOffload(b,policy=RESIDENT_ACTIVATION_POLICY,device_type='cpu',minimum_bytes=1)
    oa=torch.optim.AdamW(a.parameters());ob=torch.optim.AdamW(b.parameters())
    rng=torch.get_rng_state().clone()
    for _ in range(3):
        oa.zero_grad();ob.zero_grad();la=a(x).square().mean();la.backward()
        with off.context():lb=b(x).square().mean();lb.backward()
        assert torch.equal(la,lb)
        assert all(torch.equal(pa.grad,pb.grad)for pa,pb in zip(a.parameters(),b.parameters()))
        oa.step();ob.step()
        assert all(torch.equal(pa,pb)for pa,pb in zip(a.parameters(),b.parameters()))
    assert torch.equal(rng,torch.get_rng_state())
    assert off.audit()['copied_tensors']==off.audit()['copied_bytes']==off.audit()['restored_tensors']==0

def native_context():
    return dict(schema='SHARED_RAW_VANDA_A40_EXECUTION_V1',job_id='123.server',owner_uid=7,owner_user='own',
        allocation_deadline_epoch=2000,predicted_new_peak_bytes=1024,
        project_receipt=dict(observed_epoch=1000,balance_su=1,project=dict(name='CFP04-CF-007',active=True,
            users=['own'],start_date='1970-01-01',end_date='2099-01-01')),
        quota_receipt=dict(user='own',scope=PROJECT_SCRATCH,report_epoch=1000,observed_epoch=1000,
            hard_bytes=50*1024**3,used_bytes=1))

def test_vanda_uses_actual_allocation_not_a_core_hostname_bypass():
    c=native_context();j=dict(job_state='R',Job_Owner='own@site',project='CFP04-CF-007',exec_host='GN-A40-1/0',Resource_List=dict(ngpus=1))
    args=dict(now=1001,uid=7,user='own',hostname='GN-A40-1',job_id='123.server')
    assert validate_vanda(c,j,**args)==c
    for field,value in [('job_state','Q'),('Job_Owner','other@site'),('project','other'),('exec_host','foreign/0')]:
        bad=dict(j,**{field:value})
        with pytest.raises(PermissionError):validate_vanda(c,bad,**args)
    with pytest.raises(PermissionError):validate_vanda(c,j,**dict(args,hostname='stdct-login-01'))
    for field,value in [('project_receipt',dict(c['project_receipt'],observed_epoch=-9999)),
                        ('quota_receipt',dict(c['quota_receipt'],hard_bytes=1)),('allocation_deadline_epoch',1000)]:
        with pytest.raises(PermissionError):validate_vanda(dict(c,**{field:value}),j,**args)

def test_default_core_still_rejects_another_host(tmp_path,monkeypatch):
    monkeypatch.setattr('os.uname',lambda:SimpleNamespace(nodename='stdct-login-01'))
    with pytest.raises(PermissionError,match='ORIGINAL_NODE'):site_paths(tmp_path,{}, {})

def test_resume_without_a_pinned_closed_producer_cannot_start(tmp_path):
    p=shared_training.recipe(*['1'*64]*4)
    assert shared_resume.prepare(tmp_path,{},p,None)is None
    m=dict(resume_checkpoint=dict(path=str(tmp_path)))
    with pytest.raises(FileNotFoundError):shared_resume.prepare(tmp_path,m,p,None)
    (tmp_path/'RESUME_CUSTODY.json').write_text('{}')
    with pytest.raises(PermissionError,match='SEALED_RESUME'):shared_resume.prepare(tmp_path,m,p,None)
    with pytest.raises(PermissionError,match='ONLY_FORMAL'):shared_resume.prepare(tmp_path,m,shared_training.recipe(*['1'*64]*4,steps=10),None)

def test_saved_rng_identity_matches_actual_cpu_state():
    import torch,random,numpy as np
    from scripts.pi0294.native_train import rng_identity
    saved=dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
    assert shared_resume.rng_identity_from_saved(saved)==rng_identity()

def test_project_quota_is_not_personal_scratch_or_shared_statvfs_free():
    text='''Path: /ifs/hpc/scratch/projects/CFP04/CFP04-CF-007
Type: directory
Enforced: Yes
FSLogical(W/O Overhead): 569.19G
Thresholds On: fslogicalsize
Hard Threshold: 97.66T
Hard Exceeded: No
'''
    q=project_quota(text,user='own',report_epoch=900,observed_epoch=1000)
    assert 569.19*1024**3<q['used_bytes']<570*1024**3
    assert 97*1024**4<q['hard_bytes']<97.66*1024**4
    with pytest.raises(PermissionError):project_quota(text.replace('directory','user'),user='own',report_epoch=900,observed_epoch=1000)
    personal=quota_from_hpc_space('/scratch/own 479.32G 500.00G 95.9%', 'own',1000)
    assert personal['hard_bytes']-personal['used_bytes']<21*1024**3

def test_waiting_in_pbs_does_not_require_reauthenticating_on_a_compute_node():
    c=native_context()
    c.update(authorization_mode='SAME_PBS_ALLOCATION_AFTER_FRESH_SUBMISSION',
        submission_epoch=1001,submission_nonce='once-123',science_run_id='same-arm',
        allocation_deadline_epoch=90000)
    c['quota_receipt'].update(observed_epoch=10000,report_epoch=9999)
    j=dict(job_state='R',Job_Owner='own@site',project='CFP04-CF-007',exec_host='gn-a40-1/0',
        Resource_List=dict(ngpus=1),Variable_List=dict(PV1_SUBMISSION_NONCE='once-123',PV1_REQUEST_ID='same-arm'))
    args=dict(now=10001,uid=7,user='own',hostname='gn-a40-1',job_id='123.server')
    assert validate_vanda(c,j,**args)==c
    j['Variable_List']['PV1_SUBMISSION_NONCE']='another-job'
    with pytest.raises(PermissionError,match='EXACT_ONCE'):validate_vanda(c,j,**args)

def test_submission_must_follow_actual_entitlement_observation_even_by_microseconds():
    # Regression for actual job1514182: both permissions and balance were valid,
    # but two wall-clock reads in reversed assignment order differed by 6us.
    c=native_context()
    c.update(authorization_mode='SAME_PBS_ALLOCATION_AFTER_FRESH_SUBMISSION',
        submission_epoch=1000.1,submission_nonce='one-submission',science_run_id='same-arm')
    j=dict(job_state='R',Job_Owner='own@site',project='CFP04-CF-007',exec_host='gn-a40-1/0',
        Resource_List=dict(ngpus=1),Variable_List=dict(
            PV1_SUBMISSION_NONCE='one-submission',PV1_REQUEST_ID='same-arm'))
    args=dict(now=1001,uid=7,user='own',hostname='gn-a40-1',job_id='123.server')
    c['project_receipt']['observed_epoch']=c['submission_epoch']+0.000006
    with pytest.raises(PermissionError,match='FRESH_PROJECT_ENTITLEMENT'):
        validate_vanda(c,j,**args)
    c['project_receipt']['observed_epoch']=c['submission_epoch']-0.000006
    assert validate_vanda(c,j,**args)==c

def test_same_experiment_complete_state_resumes_exact_sampler_without_a_new_schedule(tmp_path):
    from scripts.pi0294.multitask.book_m32_export import COUNTS
    from scripts.pi0294.multitask.shared_task_sampling import TaskBalancedSampler
    from scripts.pi0294.multitask import shared_checkpoint
    def write(path,value):
        path.parent.mkdir(exist_ok=True,parents=True)
        path.write_text(json.dumps(value,indent=2)+'\n')
        return hashlib.sha256(path.read_bytes()).hexdigest()
    data=SimpleNamespace(rows=[],manifest_sha='a'*64,statistics_sha='')
    counts={'T1':[601]*10,'T2':[601]*10,'DOOR':[601]*5,
        'BOOK':[601]*21+list(COUNTS[15:])+[513,529,497,521,481,497,361]}
    for task,windows in counts.items():
        for i,n in enumerate(windows):
            row=dict(id=task+str(i),task=task,split='TRAIN',windows=n,sha256=f'{i:064x}')
            data.rows.extend((row,a)for a in range(n))
    cp=tmp_path/'checkpoint-2048'
    members={name:write(cp/name,{'CPU_fixture_only':True})for name in (
        'target_parameters.pt','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json',
        'statistics.json','TRAIN_DEV_SEAL.json','TRAIN_POPULATION.json','TASK_DEV_POPULATION.json',
        'FROZEN_PARENT_TENSORS.json','ADAPTER_PARAMETER_CONTRACT.json','RAW_PARENT_REFERENCE.json','M32_MEMORY_EXECUTION.json')}
    data.statistics_sha=members['statistics.json']
    recipe=shared_training.recipe(data.manifest_sha,data.statistics_sha,members['TRAIN_POPULATION.json'],members['TASK_DEV_POPULATION.json'])
    saved=dict(recipe=recipe,sampler=TaskBalancedSampler(data,start=65536).state_dict(65536),
        dataset_manifest_sha256=data.manifest_sha,statistics_sha256=data.statistics_sha,
        total_committed_trace_count=65536,worker_trace=dict(count=65536))
    members['SAMPLER_STATE.json']=write(cp/'SAMPLER_STATE.json',saved)
    digest=write(cp/'CHECKPOINT_COMPLETE.json',dict(
        schema=shared_checkpoint.SCHEMA,global_step=2048,cursor=65536,recipe=recipe,
        standalone_model=False,full_resume_with_required_parent=True,
        required_raw_parent_files_identity=shared_checkpoint.PARENT_ID,files=members))
    folder=tmp_path/'resume_custody'
    pins={'RUN_MANIFEST.json':write(folder/'RUN_MANIFEST.json',dict(
        science_run_id='same-arm',run_uuid='old-worker',recipe=recipe))}
    pins['RESULT.json']=write(folder/'RESULT.json',dict(optimizer_updates=3000))
    pins['LEASE.json']=write(folder/'LEASE.json',dict(status='CLOSED',run_uuid='old-worker',
        manifest_sha256=pins['RUN_MANIFEST.json'],result_sha256=pins['RESULT.json']))
    pins['TERMINAL.json']=write(folder/'TERMINAL.json',dict(effects_closed=True,producer_run_uuid='old-worker'))
    pins['RESTORE.json']=write(folder/'RESTORE.json',dict(checkpoint_manifest_sha256=digest,
        model_reconstruction=True,optimizer_state_entries=42,optimizer_updates=2048,
        scheduler_restored=True,RNG_restored=True,sampler_cursor=65536,frozen_parent_unchanged=True))
    custody=dict(science_run_id='same-arm',checkpoint_manifest_sha256=digest,files=pins)
    custody_digest=write(tmp_path/'RESUME_CUSTODY.json',custody)
    manifest=dict(science_run_id='same-arm',resume_checkpoint=dict(path=str(cp),manifest_sha256=digest),
        sealed_inputs={'RESUME_CUSTODY.json':custody_digest})
    result=shared_resume.prepare(tmp_path,manifest,recipe,data)
    assert result['step']==2048 and result['cursor']==65536
    assert result['saved_sampler']['recipe']['max_steps']==8192
    with pytest.raises(PermissionError,match='SAME_SCIENTIFIC_ARM'):
        shared_resume.prepare(tmp_path,dict(manifest,science_run_id='other-arm'),recipe,data)
    bad=dict(custody);bad['files']=dict(pins)
    bad['files']['LEASE.json']=write(folder/'LEASE.json',dict(status='ACTIVE',run_uuid='old-worker'))
    manifest['sealed_inputs']['RESUME_CUSTODY.json']=write(tmp_path/'RESUME_CUSTODY.json',bad)
    with pytest.raises(PermissionError,match='PREVIOUS_NATIVE_EFFECT'):
        shared_resume.prepare(tmp_path,manifest,recipe,data)
