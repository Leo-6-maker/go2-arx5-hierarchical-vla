"""Same-science recovery, no original-session counter or output relaxation."""
from copy import deepcopy
import ast,base64,json,os
from pathlib import Path
import pytest
from scripts.pi0294.generalization import minimal_execution as e,minimal_saved512_contract as c
from scripts.pi0294.generalization import minimal_saved512_execution as execution,minimal_execution_adapter as adapters
from scripts.pi0294.generalization import minimal_session_gate as gates
from scripts.pi0294.generalization.minimal_saved512_effects import Saved512Effects


def test_saved_E1_has_no_update_path_and_new_session_totals_are_exact():
    rows=c.sessions();assert [x['optimizer_limit']for x in rows]==[0,512]
    assert [x['query_limit']for x in rows]==[25,37]
    assert sum(x['optimizer_limit']for x in rows)==512 and sum(x['query_limit']for x in rows)==62
    assert Saved512Effects.training_sessions(None)==[rows[1]['id']]
    assert Saved512Effects.load_limits(None)==(3,2)
    tree=ast.parse(Path('scripts/pi0294/generalization/minimal_saved512_worker.py').read_text())
    body=next(x for x in ast.walk(tree)if isinstance(x,ast.FunctionDef)and x.name=='finalize_saved_E1')
    assert not any(isinstance(x,ast.Call)and isinstance(x.func,ast.Attribute)and x.func.attr in('train','step','run_training')for x in ast.walk(body))


def test_six_distinct_native_tickets_require_exact_current_debits():
    request=execution.fixed_request_fields();rows=gates.plan(request)
    assert len(rows)==6 and rows==c.tickets()
    expected=[(0,0,0),(0,0,1),(0,24,2),(0,0,0),(0,0,1),(512,36,1)]
    for row,counts in zip(rows,expected):
        actual=dict(zip(c.KEYS,counts));c.check_phase(row['session_id'],row['phase'],actual)
        assert gates.descriptor(row['session_id'],row['phase'],request=request)==row
        for key in c.KEYS:
            bad=dict(actual);bad[key]+=1
            with pytest.raises(PermissionError):c.check_phase(row['session_id'],row['phase'],bad)
    for phase in('FIRST_UPDATE','RESTORE_FULL','OFFICIAL_EXPORT'):
        with pytest.raises(PermissionError):c.check_phase(c.sessions()[0]['id'],phase,dict(zip(c.KEYS,(0,0,1))))
    old=dict(platform_training_version=adapters.remaining.VERSION,profile=adapters.remaining.PROFILE)
    assert len(gates.plan(old))==9 and len(gates.plan())==13


def test_recovery_is_explicit_not_an_alias_and_one_eight_hour_profile():
    assert adapters.select(c.VERSION,c.PROFILE)is execution
    with pytest.raises(PermissionError):adapters.select(e.VERSION,c.PROFILE)
    r=execution.fixed_request_fields();r.update(request_id='a'*36,source_sha='f'*40)
    assert r['reserved_H200_s']==28920 and r['CPU_upper_s']==78000
    assert execution.native_resources()['Hopper_storage_bytes']==0 and r['new_output_bound_bytes']==70000000000
    assert r['paid_output_allowance_bytes']==144294967296


def test_original_graph_eval_fix_and_strict_noise_contract_remain_present():
    source=Path('scripts/pi0294/generalization/minimal_worker.py').read_text()
    assert 'self.instance.model.eval()'in source and 'not any(module.training'in source
    body=Path('scripts/pi0294/generalization/minimal_inference_export.py').read_text()
    assert body.index("need(exact(inputs)")<body.index("need(exact(('physical_H40'")


def test_real_saved_state_result_is_bound_to_failed_predecessor_not_new_native_permission():
    where=os.environ.get('ARX5_CURRENT_REMAINING_CAPTURE_DIR')
    if not where:pytest.skip('Actual private saved-state receipt not provided')
    latest=sorted(Path(where).glob('SAVED512_RECOVERY_FINITE_AUDIT03_STATUS_*.private.json'))[-1]
    x=json.loads(latest.read_bytes());a=e.raw(x['records']['RESULT.json']['value']);b=json.dumps(x['records']['OS_EXIT.json']['value']).encode()
    assert e.sha(a)==c.RESULT and e.sha(b)==c.OS_EXIT
    binding=c.binding(a,b);assert binding['E1_new_optimizer_calls']==0 and binding['native_admission']is False
    for key,value in [('native_exit_status',0),('E1_actual_output_reconstruction_qualified',True),('E2_formal_updates',512)]:
        bad=json.loads(a);bad[key]=value
        with pytest.raises(PermissionError):c.binding(e.raw(bad),b)


def test_exact_legacy_probe0600_is_readonly_content_checked_not_tensor_policy(tmp_path):
    from scripts.pi0294.generalization.minimal_saved512_inputs import verify_witness
    import os
    path=tmp_path/'probe.npz';path.write_bytes(b'actual saved small fixture');path.chmod(0o600)
    st=path.stat();row=dict(inode=st.st_ino,bytes=st.st_size,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns,sha256=e.sha(path.read_bytes()))
    allowed={str(path):row['sha256']};verify_witness(path,row,science=tmp_path,legacy_probes=allowed)
    assert path.stat().st_mode&0o777==0o600
    with pytest.raises(PermissionError):verify_witness(path,row,science=tmp_path,legacy_probes={})
    with pytest.raises(PermissionError):verify_witness(path,row,science=tmp_path,legacy_probes={str(path):'0'*64})
    old=path.read_bytes();path.write_bytes(b'x'*len(old));os.utime(path,ns=(st.st_atime_ns,st.st_mtime_ns))
    with pytest.raises(PermissionError):verify_witness(path,row,science=tmp_path,legacy_probes=allowed)
    tensor=tmp_path/'model.safetensors';tensor.write_bytes(b'tensor fixture');tensor.chmod(0o600);ts=tensor.stat()
    tw=dict(inode=ts.st_ino,bytes=ts.st_size,mtime_ns=ts.st_mtime_ns,ctime_ns=ts.st_ctime_ns,sha256=e.sha(tensor.read_bytes()))
    with pytest.raises(PermissionError):verify_witness(tensor,tw,science=tmp_path,legacy_probes={})


def test_retained_previous_guard_keeps_original_close_reserve():
    from scripts.pi0294.generalization import minimal_remaining_physical as prior
    where=os.environ.get('ARX5_CURRENT_REMAINING_CAPTURE_DIR')
    if not where:pytest.skip('Actual adopted complete history required')
    import gzip
    v=json.loads((Path(where)/'SAVED512_RECOVERY_CONTROL01_NATIVE_INDEPENDENT_AFTER_EXACT_STALE_ZERO.private.json').read_bytes())
    g=json.loads(gzip.decompress(base64.b64decode(v['GLOBAL_gzip_base64'])))
    m,row=adapters.remaining.basic(g);p=deepcopy(m['terminal']['physical']);custody=m['physical_custody'];now=p['observed_epoch']
    p['next_output_upper_bytes']=0
    with pytest.raises(PermissionError):prior.check_proof(p,custody,row,now_epoch=now,observed_epoch=now)
    p['next_output_upper_bytes']=prior.CONTROL_GROWTH_RESERVE
    assert prior.check_proof(p,custody,row,now_epoch=now,observed_epoch=now)==p
    tree=ast.parse(Path('scripts/pi0294/generalization/minimal_saved512_physical.py').read_text())
    calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='inspect'and isinstance(n.func.value,ast.Attribute)and n.func.value.attr=='prior']
    assert len(calls)==1 and isinstance(calls[0].keywords[0].value,ast.Name)and calls[0].keywords[0].value.id=='CONTROL_GROWTH_RESERVE'
