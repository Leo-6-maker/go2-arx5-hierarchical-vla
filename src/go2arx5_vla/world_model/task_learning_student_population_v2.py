"""Join already executed DEV groups without replaying a condition or its model.

Group workers perform the original native classification. At their immutable
cold boundary we check those cached classifications against their actual child
artifacts and retain every attempted condition, including controlled rejection.
"""
import json
from pathlib import Path

from . import task_learning_student_v1 as student
from . import task_learning_grasp_v1 as task
from .task_learning_endpoint_decision_v2 import descriptor, load, native_counts
from .storage import digest


def aggregate(panel_reference, groups):
    from .task_learning_student_batch_v1 import selection
    panel = load(panel_reference)
    wanted = {r['native_run']: r for r in student.validate(panel)}
    outcomes, evidence, attempted = [], [], set()
    cold_safe=True
    for root in map(Path, groups):
        task.need(root.is_absolute() and root.resolve() == root and not root.is_symlink(),
                  'CANONICAL_ACTUAL_DEV_GROUP')
        inp = json.loads((root/'INPUT.private.json').read_bytes())
        lease = json.loads((root/'control/LEASE.json').read_bytes())
        result = json.loads((root/'RESULT.private.json').read_bytes())
        task.need(inp['panel'] == panel_reference['path'] and inp['panel_sha256'] == panel_reference['sha256']
            and inp['source_sha'] == panel['source_sha'] and lease.get('status') == 'CLOSED'
            and lease.get('input_sha256') == digest(root/'INPUT.private.json')
            and lease.get('worker_exit_code') == result.get('exit_code')
            and result.get('source_sha') == panel['source_sha']
            and result.get('panel_sha256') == panel_reference['sha256']
            and result.get('assignment_slice') == inp['assignment_slice'], 'ACTUAL_CLOSED_SEALED_DEV_GROUP')
        rows = selection(inp, panel)
        names = {r['native_run'] for r in rows}
        path = root/'OUTCOMES.private.json'
        values = json.loads(path.read_bytes()) if path.exists() else []
        actual_names={v['native_run']for v in values}
        task.need(not attempted.intersection(actual_names), 'DEV_CONDITION_CANNOT_BE_RETRIED_OR_COUNTED_TWICE')
        attempted.update(actual_names)
        task.need(result.get('selected_requests') == len(rows)
            and result.get('completed_in_group') == len(values)
            and result.get('summary') == student.summarize(panel, values)
            and all(v['native_run'] in names for v in values), 'UNCHANGED_GROUP_CLASSIFICATION_AND_DENOMINATORS')
        if result['exit_code'] == 0:
            task.need(len(values) == len(rows) and result.get('failure') is None,
                      'SUCCESSFUL_GROUP_HAS_ALL_SELECTED_ACTUAL_CONDITIONS')
        cold_safe=cold_safe and result.get('unresolved_pair')is None
        # One trusted immutable source boundary, not a new whole-source scan
        # inside every control tick. The classifier/math must be the published
        # implementation that wrote these outcomes.
        source = Path(inp['source_root'])
        member = 'src/go2arx5_vla/world_model/task_learning_student_v1.py'
        task.need(inp['source_files'].get(member) == digest(source/member) == digest(Path(student.__file__)),
                  'SAME_ORIGINAL_CACHED_NATIVE_CLASSIFICATION_IMPLEMENTATION')
        for value in values:
            name = value['native_run'];native = root/'runs'/name
            task.need(name in wanted and all(value[k] == wanted[name][k] for k in
                ('method','layout','restart','parent','split','query_seeds')), 'EXACT_ORIGINAL_PANEL_CONDITION')
            s = json.loads((native/'EXECUTION_SPEC.json').read_bytes())
            lp = native/'control/LEASE.json';wp = native/'WORKER_EXIT.private.json'
            l=json.loads(lp.read_bytes());w=json.loads(wp.read_bytes())if wp.exists()else{}
            revoked=(not w and not(native/'SUPERVISOR.private.json').exists()
                and l.get('terminal_reason')=='NEVER_DISPATCHED_PERMISSION_REVOKED'
                and l.get('worker_exit_code')is None and value.get('worker_exit_code')is None
                and value['technical_valid']is False)
            task.need(s['registered_root'] == str(native) and s['source_sha'] == panel['source_sha']
                and s['stable_task_panel_sha256'] == panel_reference['sha256']
                and l['status'] == 'CLOSED' and l['spec_sha256'] == digest(native/'EXECUTION_SPEC.json')
                and(revoked or(w and l.get('worker_exit_code') == w.get('exit_code') == value.get('worker_exit_code'))),
                'ACTUAL_NATIVE_CHILD_IDENTITY_AND_EXIT')
            # Streaming hashes revalidate the large frozen evidence once at a
            # group boundary, without reparsing or rewriting its whole trace.
            result_path = native/'native-01/RESULT.json'
            pin=value.get('result_sha256')
            task.need((not result_path.exists()and value['technical_valid']is False)if pin is None
                else digest(result_path)==pin,'ORIGINAL_NATIVE_RESULT_STILL_SEALED')
            gpu = Path(s['target_gr00t_gpu_root'])
            gs = json.loads((gpu/'EXECUTION_SPEC.json').read_bytes())
            gl = json.loads((gpu/'control/LEASE.json').read_bytes())
            gw = json.loads((gpu/'WORKER_EXIT.private.json').read_bytes())
            result_path=gpu/'fit-01/RESULT.private.json';pin=value.get('GPU_result_sha256')
            result_matches=((not result_path.exists()and value['technical_valid']is False)if pin is None
                else digest(result_path)==pin)
            task.need(gl.get('status') == 'CLOSED' and gl.get('spec_sha256') == digest(gpu/'EXECUTION_SPEC.json')
                and gl.get('worker_exit_code') == gw.get('exit_code')
                and gs.get('producer_native_spec') == descriptor(native/'EXECUTION_SPEC.json')
                and result_matches,
                'ACTUAL_PAIRED_GPU_EXIT_AND_RESULT')
        outcomes.extend(values)
        evidence.append(dict(root=str(root),input=descriptor(root/'INPUT.private.json'),
            lease=descriptor(root/'control/LEASE.json'),result=descriptor(root/'RESULT.private.json'),
            outcomes=descriptor(path) if path.exists() else None,
            assignment_slice=inp['assignment_slice'],exit_code=result['exit_code']))
    summary = student.summarize(panel, outcomes)
    return dict(schema='TASK_LEARNING_FULL_NATIVE_DEV_POPULATION_ACTUAL_V2',
        status='COMPLETE_COMMON_DEV80' if summary['complete'] else 'PARTIAL_COMMON_DEV_NOT_A_SELECTION_RESULT',
        panel=panel_reference,source_sha=panel['source_sha'],global_step=panel['B1_global_step'],
        groups=evidence,outcomes=outcomes,summary=summary,native=native_counts(panel,outcomes),
        cold_continuation_safe=bool(cold_safe),
        planned_conditions=80,attempted_conditions=len(outcomes),unrun_conditions=80-len(outcomes),
        methods_selected_from_partial=False,TEST_results_read=False,
        scientific_continuation_authorized=False)
