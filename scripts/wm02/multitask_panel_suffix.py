"""Recover only the undispatched suffix of the registered matched panel."""
from pathlib import Path

from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import unit_state
from go2arx5_vla.world_model import scale20k_native as scale
from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
from go2arx5_vla.world_model.multitask_score_v6 import Task
from go2arx5_vla.world_model.storage import digest
from scripts.pi0292.durable import read


def budgets(conditions):
    query_count = sum(120 if Task.parse(row['instruction']).operation == 'HOLD' else 180
                      for row,_ in conditions)
    return len(conditions), 5703*len(conditions)+250*query_count, query_count


def verified_prefix(spec, conditions):
    """Read the cold parents and their actual case receipts in frozen order."""
    from .sim_only_campaign_v51 import fatal_control_violation

    segments = spec.get('previous_panel_segments', [])
    expected_ids = [f"{row['id']}-{arm}" for row,arm in conditions]
    group = len(spec['packages'])
    prefix = []
    roots = set()
    immutable = ('scope','plan_commit','evaluation_stage','compiled_population','packages',
                 'interface_predecessor','mobile_world_lock','ROS_DOMAIN_ID','owner')
    for index, item in enumerate(segments):
        parent = read_pin(item['spec'])
        prior = read_pin(item['result'])
        root = Path(parent['registered_root'])
        lease = read(root/'control/LEASE.json')
        state = unit_state('wm02-'+root.name)
        block = parent['scope'] == scale.SCOPE and 'condition_stop' in parent
        end = len(prefix)+len(scale.next_block(parent,conditions,len(prefix))) if block else len(conditions)
        bounded_stop = (prior['exit_code'] == 1
                        and read(root/'PROGRESS.private.json')['phase'] == 'FINITE_STOP_OR_DEADLINE')
        complete_block = block and prior['exit_code'] == 0 and prior['new_verified_native_requests'] == end-len(prefix)
        if (root in roots or item['result']['path'] != str(root/'RESULT.private.json')
                or item['spec']['path'] != str(root/'EXECUTION_SPEC.json')
                or any(parent[key] != spec[key] for key in immutable)
                or parent.get('previous_panel_segments', []) != segments[:index]
                or lease['status'] != 'CLOSED' or lease['spec_sha256'] != item['spec']['sha256']
                or state['pid'] or state['cgroup_pids'] or prior['source_sha'] != parent['source_sha']
                or not (bounded_stop or complete_block) or prior['unresolved_case_launches'] or prior['failure']):
            raise ValueError('ONLY_CLOSED_RECONCILED_FINITE_PANEL_SEGMENTS_MAY_BE_CARRIED')
        roots.add(root)
        known = [row for row in prior['records'] if row['role'] == 'STUDENT']
        if known[:len(prefix)] != prefix:
            raise ValueError('EARLIER_STUDENT_TERMINALS_MUST_REMAIN_IDENTICAL')
        added = known[len(prefix):]
        case_paths = [Path(path) for path in parent['cases']]
        if (not added or len(added) % group or len(known) >= len(conditions)
                or [r['id'] for r in known] != expected_ids[:len(known)]
                or [p.parent.name for p in case_paths] != expected_ids[len(prefix):end]
                or prior['attempted_case_launches'] != len(added)
                or prior['new_verified_native_requests'] != len(added)):
            raise ValueError('RESUME_REQUIRES_THE_EXACT_MATCHED_CONDITION_PREFIX')
        worker = prior['model_service']
        generated = sum(row['actual_model_generations'] for row in added)
        if (not worker or worker['exit_code']
                or worker['attempted_generations'] != generated or worker['completed_generations'] != generated):
            raise ValueError('CARRIED_MODEL_GENERATION_EFFECTS_MUST_RECONCILE')
        for path,row in zip(case_paths[:len(added)],added,strict=True):
            result = read_pin(dict(path=str(path.parent/'RESULT.private.json'),sha256=row['result_sha256']))
            case_lease = read(path.parent/'control/LEASE.json')
            if (path.parent.parent != root/'cases' or case_lease['status'] != 'CLOSED'
                    or case_lease['spec_sha256'] != digest(path)
                    or result['source_sha'] != parent['source_sha'] or result['physical_requests'] != 1
                    or result['actual_native_integrations'] != row['native_ticks']
                    or result['model_generations'] != row['actual_model_generations']
                    or row['technical_failure'] or result['technical_failure'] or fatal_control_violation(row)
                    or row['score'] is None or row['score']['missing_native_intervals']
                    or row['score']['native_step'] != row['native_ticks']):
                raise ValueError('CARRIED_TERMINAL_MUST_HAVE_ORIGINAL_COMPLETE_NATIVE_EVIDENCE')
        prefix = known
    return prefix
