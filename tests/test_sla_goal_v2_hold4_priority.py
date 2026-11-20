import pytest

from scripts.sla import goal_v2_hold4_after_students as hold
from scripts.sla.goal_v2_hold4_after_students import require_unqualified_tgt


@pytest.mark.parametrize('state,attempts,failure,eligible', [
    ('TGT_RGB16_FAIL',1,'ValueError: SCALE20K_CORE_REPLAY_STRICT_OUTPUT_OR_GUARD_MISMATCH',True),
    ('TGT_RGB16_REQUIRES_RECONCILIATION',0,'PermissionError: TGT_RGB16_RESOURCE_REJECT:host',True),
    ('TGT_RGB16_FAIL',1,'Unresolved worker exit',False),
    ('TGT_RGB16_PASS',16,None,False),
])
def test_only_reconciled_unqualified_target_releases_lower_priority_collection(state, attempts, failure, eligible):
    report = dict(status='TGT5K_STRICT_RGB16_UNQUALIFIED',native_started=False,
                  rgb16_status=state,rgb16_attempts=attempts,rgb16_failure=failure)
    if eligible:
        require_unqualified_tgt(report)
    else:
        with pytest.raises(ValueError,match='TGT_PRIORITY'):
            require_unqualified_tgt(report)
    report['native_started'] = True
    with pytest.raises(ValueError,match='TGT_PRIORITY'):
        require_unqualified_tgt(report)


def test_recovered_student_prefix_must_have_cold_actual_conditions(tmp_path, monkeypatch):
    block = tmp_path/'original-rows7-12'
    block.mkdir()
    (block/'control').mkdir()
    original = block/'EXECUTION_SPEC.json'
    result = block/'RESULT.private.json'
    hold.write(original, dict(condition_stop=12))
    hold.write(result, dict(new_verified_native_requests=3, unresolved_case_launches=0, failure=None))
    hold.write(block/'control/LEASE.json', dict(status='CLOSED', spec_sha256=hold.pin(original)['sha256']))
    card = tmp_path/'TAIL_SPEC.json'
    hold.write(card, dict(campaign='canonical', hold_observer='prior', block2_resource_refusal='located',
        unit='tail.service', source_sha='candidate', root=str(tmp_path), targets={'12':str(block)}))
    effect = tmp_path/'RESULT.private.json'
    data = dict(status='ADAPT0_TAIL_STOP_OR_SITE_BOUNDARY', source_sha='candidate', carried_native_requests=6,
        native_requests=3, native_ticks=10, model_generations=2,
        results=[dict(start=6, stop=9, effect=dict(spec=hold.pin(original), result=hold.pin(result)))])
    hold.write(effect, data)
    monkeypatch.setattr(hold, 'cold', lambda *_args, **_kw: dict(pid=0, cgroup_pids=[]))
    monkeypatch.setattr(hold, 'unit_state', lambda _: dict(pid=0, cgroup_pids=[]))
    dep = dict(spec=hold.pin(card), owner=dict(unit='tail.service'))
    kw = dict(deadline=100, stopped=lambda:False)
    assert hold.sealed_adapt0_panel(dep, 'canonical', 'prior', **kw)['remaining_completed']==3
    hold.write(effect, dict(data, native_requests=0, results=[]))
    with pytest.raises(PermissionError,match='ACTUAL_RECONCILED'):
        hold.sealed_adapt0_panel(dep, 'canonical', 'prior', **kw)
