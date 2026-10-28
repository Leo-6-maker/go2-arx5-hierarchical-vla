"""Inference reference and container environment CPU fixtures; no native GO."""
import base64
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import pytest
from scripts.pi0294.pv1 import mix_vanda_reference as r, mix_vanda_container as container
from scripts.pi0294.pv1 import mix_accounting as a, mix_vanda_cohost as cohost
from scripts.pi0294.test_v5_vanda_cohost import captured


def test_actual_original_metadata_is_not_the_rewritten_public_transfer_size(captured):
    workspace = Path(os.environ.get('PI0294_PRIVATE_WORKSPACE',str(Path(__file__).resolve().parents[4])))
    base = workspace/'work/private/v5-execution-20261025'
    for arm,pattern in [('C50','C50_REAL_CLOSED_HANDOFF_READONLY_*'),('D25','D25_VANDA_FIRST_REAL_CLOSED_HANDOFF_READONLY_*')]:
        v = json.loads(sorted(base.glob(pattern))[-1].read_bytes())['value']
        plan = r.prepare(captured['global_bytes'],a.raw(v['closed_capture']),a.raw(v['small_metadata']),arm=arm)
        assert len(plan['files']) == 9
        assert sum(x['bytes'] for x in plan['files'].values()) == 12576324650
        assert plan['tensor_copies'] == plan['source_hardlinks'] == 0
        assert plan['source_chmod'] is False and plan['source_full_resume_retained'] is True
        assert plan['handoff']['global_sha256'] == a.sha(captured['global_bytes'])


def fixture_plan(tmp_path):
    source = tmp_path/'source'; source.mkdir()
    names = ['model-00001.safetensors','model-00002.safetensors','model-00003.safetensors',
             'config.json','model.safetensors.index.json','statistics.json',
             'processor/processor_config.json','processor/statistics.json','processor/embodiment_id.json']
    files = {}
    for i,name in enumerate(names):
        p = source/name; p.parent.mkdir(exist_ok=True); data = ('cpu-fixture-'+str(i)).encode(); p.write_bytes(data)
        files[name] = dict(path=str(p),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
    complete = a.raw(dict(global_step=1024,files={n:v['sha256'] for n,v in files.items()}))
    return dict(schema='V5_CD_VANDA_CLOSED1024_READONLY_INFERENCE_REFERENCE_V1',arm='C50',files=files,
                checkpoint_root=str(source),checkpoint_manifest_sha256=a.sha(complete),
                original_complete_base64=base64.b64encode(complete).decode(),tensor_copies=0,
                source_hardlinks=0,source_chmod=False)


def test_reference_view_leaves_original_inode_ctime_nlink_and_mode_unchanged(tmp_path):
    plan = fixture_plan(tmp_path); audit = r.audit_reference(plan)
    before = {n:r.metadata(x['path']) for n,x in plan['files'].items()}
    view = tmp_path/'view'; receipt = r.create_view(view,plan,audit)
    assert receipt['new_tensor_bytes'] == 0 and receipt['source_inodes_unchanged']
    assert {n:r.metadata(x['path']) for n,x in plan['files'].items()} == before
    assert all((view/n).is_symlink() and (view/n).resolve() == Path(x['path']) for n,x in plan['files'].items())
    assert not any((view/n).exists() for n in ('optimizer.pt','RNG_STATE.pt','SAMPLER_STATE.json'))
    with pytest.raises(PermissionError): r.create_view(view,plan,audit)
    # The explicit input role does NOT relax ordinary raw-output link guards.
    with pytest.raises(PermissionError): cohost.tree_bytes(view)


@pytest.mark.parametrize('change',[None,'device','inode','bytes','mtime_ns','ctime_ns','digest','not_rechecked'])
def test_closed_stream_reuse_needs_every_actual_kernel_witness_field(tmp_path,monkeypatch,change):
    """Closed endpoint parser is an explicit fixture; real temp inodes are used."""
    plan=fixture_plan(tmp_path);rows=[]
    for name,member in plan['files'].items():
        rows.append(dict(r.metadata(member['path']),path=member['path'],sha256=member['sha256']))
    witness=dict(paths=rows,all_paths_rechecked=True,all_unique_inodes_independently_rehashed=True)
    proof=dict(arm='C50',checkpoint_manifest_sha256=plan['checkpoint_manifest_sha256'])
    audit=dict(checkpoint_root=plan['checkpoint_root'],streamed_witness=witness)
    if change in ('device','inode','bytes','mtime_ns','ctime_ns'):rows[0][change]+=1
    elif change=='digest':rows[0]['sha256']='0'*64
    elif change=='not_rechecked':witness['all_paths_rechecked']=False
    monkeypatch.setattr(r.delivery,'endpoint',lambda raw:({},proof,audit))
    if change:
        with pytest.raises(PermissionError):r.reuse_closed_stream_witness(plan,b'EXPLICIT_CPU_CLOSED_ENDPOINT_FIXTURE')
        assert not(tmp_path/'view').exists()
    else:
        result=r.reuse_closed_stream_witness(plan,b'EXPLICIT_CPU_CLOSED_ENDPOINT_FIXTURE')
        assert result['current_tensor_bytes_restreamed']==0 and result['model_loader_independent_full_SHA_still_required']
        assert result['schema']=='V5_CD_VANDA_UNCHANGED_CLOSED_STREAM_WITNESS_V1'
        r.create_view(tmp_path/'view',plan,result)


@pytest.mark.parametrize('change',['bytes','symlink','changed_after_audit','orphan_audit'])
def test_source_substitution_cannot_create_reference_view(tmp_path, change):
    plan = fixture_plan(tmp_path); audit = r.audit_reference(plan); member = Path(next(iter(plan['files'].values()))['path'])
    if change == 'bytes':
        member.write_bytes(b'changed')
        with pytest.raises(PermissionError): r.audit_reference(plan)
    elif change == 'symlink':
        data = member.read_bytes(); member.unlink(); other=tmp_path/'other'; other.write_bytes(data); member.symlink_to(other)
        with pytest.raises(PermissionError): r.audit_reference(plan)
    else:
        if change == 'changed_after_audit': member.write_bytes(member.read_bytes()+b'x')
        else: audit['reference_plan_sha256']='0'*64
        with pytest.raises(PermissionError): r.create_view(tmp_path/'view',plan,audit)
        assert not (tmp_path/'view').exists()


def test_current_PBS_identity_survives_clean_container_command():
    env = container.physics_environment({'PBS_JOBID':'1234567.stdct-mgmt-02'},code_root='/sealed/code')
    assert env['PBS_JOBID']=='1234567.stdct-mgmt-02' and env['CUDA_VISIBLE_DEVICES']==''
    command = container.physics_command({'PBS_JOBID':'1234567.stdct-mgmt-02'},code_root='/sealed/code',
        container='/original/image.sif',readonly=['/registered/input'],writable=['/registered/output'],
        module='scripts.pi0294.pv1.mix_vanda_render',args=['--root','/registered/output'])
    assert '--cleanenv' in command and 'PBS_JOBID=1234567.stdct-mgmt-02' in command
    assert 'LIBGL_ALWAYS_SOFTWARE=1' in command and 'MUJOCO_GL=egl' in command
    assert 'MUJOCO_EGL_DEVICE_ID' not in env


@pytest.mark.parametrize('env',[{}, {'PBS_JOBID':'123.hopper-m-02'}, {'PBS_JOBID':'1234567.stdct-mgmt-02','APPTAINERENV_CUDA_VISIBLE_DEVICES':'0'},
    {'PBS_JOBID':'1234567.stdct-mgmt-02','SINGULARITYENV_MUJOCO_GL':'glfw'}])
def test_missing_PBS_or_hidden_environment_cannot_start_source_container(env):
    with pytest.raises(PermissionError): container.physics_environment(env,code_root='/sealed/code')


def test_mount_injection_and_unapproved_worker_are_rejected():
    args=dict(code_root='/sealed/code',container='/original.sif',readonly=['/input'],writable=['/output'],module='scripts.pi0294.core')
    for key,value in [('module','new.unknown.controller'),('readonly',['/tmp:/elsewhere:rw'])]:
        changed=deepcopy(args); changed[key]=value
        with pytest.raises(PermissionError): container.physics_command({'PBS_JOBID':'1234567.stdct-mgmt-02'},**changed)
