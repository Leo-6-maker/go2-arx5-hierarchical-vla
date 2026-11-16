"""Three preserved contact events; passive geometry only, no native execution."""
import json
from pathlib import Path
import time
import traceback
import numpy as np

from scripts.pi0292.durable import write
from scripts.wm02.mfree_c20_query_audit_v3 import collision
from go2arx5_vla.world_model.mfree_unattended_ledger_v1 import read, sha


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()] if Path(path).exists() else []


def run(spec):
    import mujoco
    assert mujoco.__version__ == '3.2.7'
    root=Path(spec['registered_root']); old=Path(spec['previous_campaign']); started=time.time()
    records=[]
    # Reuse the already completed independent student geometry analysis.
    student=read(old/'readonly-attribution-v3/COLLISION.private.json')[0]
    geometry=read(old/'evidence-pack/F1_CONTACT_MECHANISM.private.json')
    records.append(dict(role='STUDENT_F1', original=student, decomposition=geometry,
                        passive_recomputed=False, reused_evidence=True))
    for case in spec['expert_unsafe_cases']:
        case=Path(case); value=read(case/'RESULT.private.json'); native_spec=read(case/'EXECUTION_SPEC.json')
        assert value['unsafe'] and not value['technical_failure']
        audit=value['trace_audit']; assert audit['integrations_after_first_STOP']==0
        result=collision(case,native_spec,value)
        commands=lines(case/'native-01/COMMANDS.private.jsonl')
        samples=lines(case/'native-01/MOBILE_SAMPLES.private.jsonl')
        first=result['first_illegal']; stop_step=first['step'] if first else value['actual_native_integrations']
        query_commands=[c for c in commands if c.get('phase')=='mobile-continuous-expert-query'
                        and c.get('native_step',stop_step+1)<=stop_step]
        query=query_commands[-1] if query_commands else None
        boundaries=[c for c in commands if c.get('phase')=='mobile-accepted-boundary-050'
                    and c.get('native_step',stop_step+1)<=stop_step]
        nearby=[s for s in samples if s.get('native_end_step',0)>=stop_step-1250]
        capture_path=case/'data/LIVE_CAPTURE.private.json'
        record=dict(role='EXPERT_'+native_spec['mobile_condition']['family'], case=str(case),
            spec_sha256=sha(case/'EXECUTION_SPEC.json'), result_sha256=sha(case/'RESULT.private.json'),
            collision=result, last_expert_query=query,
            last_query_age_s=None if query is None else .002*(stop_step-query['native_step']),
            last_boundary=boundaries[-1] if boundaries else None,
            boundary_age_s=None if not boundaries else .002*(stop_step-boundaries[-1]['native_step']),
            nearby_task_samples=nearby, command_phase_counts={p:sum(c.get('phase')==p for c in commands)
                 for p in {c.get('phase') for c in commands}},
            command_examples=commands[-8:], original_stop=value['physics_stops'],
            original_trace_audit=audit, shutdown_clean=value['shutdown_clean'],
            failed_capture_never_successful_BC=True,
            full_WBC_state_restore_available=False,
            solver_reference_age='NOT_ESTABLISHED_BY_TASK_SAMPLES_USE_NATIVE_TRACE_VERSIONS',
            new_physical_requests=0,new_integrations=0)
        records.append(record)
        write(root/'PROGRESS.private.json',dict(epoch=time.time(),events=len(records),physical_requests=0,
              passive_FK_only=True))
    write(root/'THREE_CONTACT_EVENTS.private.json',records)
    request=dict(schema='SLA_CLOSE_GOAL_SAFETY_RE_ADMISSION_REQUEST_V1',status='NOT_ADMITTED',
        F1_F2_expert_quarantine='ACTIVE', student_quarantine='ACTIVE', M_ASSIST='ACTIVE',
        existing_contact_failures_preserved=True, native_requests_permitted_now=0,
        exact_causal_controller_state_replay_proven=False,
        proposed_prevention_hypothesis='INDEPENDENT_CONSERVATIVE_PAD_CLEARANCE_AND_TRACKING_MONITOR_BEFORE_CONTACT_WITH_FIXED_WBC',
        implemented_or_qualified=False,
        required_followup=['Resolve reference/solver version age from saved native trace',
            'Specify a new shared high-level approach/clearance contract and monitoring stop conditions',
            'Independent authorized reviewer binds source, scenes, safety profile and finite panel before any native request'],
        no_SLA_privileged_input=True, WBC_or_original_threshold_change=False,
        conditional_expert_panel_cap=8, event_evidence=dict(path=str(root/'THREE_CONTACT_EVENTS.private.json'),
                                                        sha256=sha(root/'THREE_CONTACT_EVENTS.private.json')))
    write(root/'SAFETY_RE_ADMISSION_REQUEST.private.json',request)
    out=dict(schema='SLA_CLOSE_GOAL_THREE_CONTACT_READONLY_ACTUAL_V1',exit_code=0,events=3,
        student_evidence_reused=1,new_expert_passive_analyses=2, physical_requests=0,
        native_integrations=0, model_generations=0, optimizer_updates=0,
        elapsed_wall_s=time.time()-started,CPU_s=time.process_time(),
        admission=False,quarantine_preserved=True,
        event_summaries=[dict(role=r['role'],first_illegal=(r.get('collision')or r.get('original'))['first_illegal'],
                             last_expert_query=r.get('last_expert_query'),last_query_age_s=r.get('last_query_age_s')) for r in records],
        evidence_sha256={p.name:sha(p)for p in root.glob('*.private.json')})
    write(root/'RESULT.private.json',out)
    return out


if __name__=='__main__':
    import sys
    s=read(sys.argv[2]);root=Path(s['registered_root'])
    try:run(s);code=0
    except BaseException:
        write(root/'RESULT.private.json',dict(exit_code=1,failure=traceback.format_exc(),physical_requests=0,native_integrations=0));code=1
    from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close
    close(s,code)
    raise SystemExit(code)
