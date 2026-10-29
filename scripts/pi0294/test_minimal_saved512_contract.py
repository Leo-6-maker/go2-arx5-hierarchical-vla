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
