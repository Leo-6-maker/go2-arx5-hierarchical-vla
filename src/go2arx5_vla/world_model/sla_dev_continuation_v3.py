"""Continue the untouched suffix of a fixed SLA DEV16 after a closed group."""
import json
from pathlib import Path
import subprocess

from .stable_task_scene_v1 import pinned
from .storage import digest

SCHEMA = 'SLA_DEV16_COLD_SUFFIX_INPUT_V3'
REGISTRATION_PROOF_KEYS = frozenset(('S1_loader_reconstruction', 'S1_prelaunch_reconciliation'))


def reconciled_prelaunch(panel, inp, names, journal, registry):
    """Prove that the known rejected registration had no run or registry effect."""
    from .task_learning_grasp_v1 import need
    proof = panel['registration'].get('S1_prelaunch_reconciliation')
    if proof is None:
        return None
    need(set(proof) == {'input', 'result', 'lease', 'registration_intent'}, 'EXACT_S1_PREFLIGHT_PROOF')
    old, result, lease, intent = [pinned(proof[k]) for k in ('input', 'result', 'lease', 'registration_intent')]
    group = Path(proof['input']['path']).parent
    need(proof['result']['path'] == str(group / 'RESULT.private.json')
         and proof['lease']['path'] == str(group / 'control/LEASE.json')
         and lease['status'] == 'CLOSED' and lease['worker_exit_code'] == result['exit_code'] == 1
         and lease['input_sha256'] == proof['input']['sha256']
         and old['native_base'] == inp['native_base'] and old['gpu_base'] == inp['gpu_base']
         and old['assignment_slice'] == result['assignment_slice'] and result['completed_in_group'] == 0
         and result['failure'] == 'ValueError: TASK_LEARNING_GRASP_SLA_DEPLOYMENT_USES_ACTUAL_TRAINED_AND_QUALIFIED_ADAPTER_CODE'
         and result['unresolved_pair']['stage'] == 'REGISTRATION_NOT_COMPLETED_NO_REPLAY',
         'CLOSED_EFFECT_FREE_S1_LOADER_PREFLIGHT_REQUIRED')
    root = Path(result['unresolved_pair']['native_root'])
    name = root.name
    partner = root.with_name(name + '-gpu')
    key = name.upper()
    need(root.parent == group / 'runs' and name in names
         and proof['registration_intent']['path'] == str(journal.path(key + '-REGISTER'))
         and intent['status'] == 'INTENT'
         and not any(journal.path(k).exists() for k in
                     (key + '-PAIR-REGISTRY', key + '-RUN', key + '-GPU-RUN'))
         and not any(entry.get('run_uuid') in (name, name + '-gpu') for entry in registry['entries'])
         and all(not (r / p).exists() for r in (root, partner) for p in
                 ('control/LEASE.json', 'SUPERVISOR.private.json', 'native-01', 'fit-01', 'WORKER_EXIT.private.json'))
         and subprocess.check_output(['systemctl', '--user', 'show', 'wm02-' + group.name,
                                      '--property=MainPID', '--value'], text=True).strip() == '0',
         'NO_EFFECTS_FROM_RECONCILED_S1_PREFLIGHT')
    return name


def retained_prefix(panel):
    from scripts.pi0294.e1.journal import Journal
    from .task_learning_grasp_v1 import need

    rows, completed_groups, inp = _closed_prefix(panel)
    native_base = pinned(inp['native_base'])
    campaign = Path(native_base['campaign_root'])
    journal = Journal(campaign / 'control/journal', campaign.name)
    registry = json.loads(Path(native_base['registry']).read_bytes())
    names = {row['native_run'] for row in panel['assignments'][len(rows):]}
    reconciled = reconciled_prelaunch(panel, inp, names, journal, registry)
    for name in names:
        key = name.upper()
        suffixes = ('-RECONCILED-REGISTER', '-PAIR-REGISTRY', '-RUN', '-GPU-RUN') if name == reconciled else (
            '-REGISTER', '-PAIR-REGISTRY', '-RUN', '-GPU-RUN')
        need(not any(journal.path(key + suffix).exists() for suffix in
                     suffixes)
             and not any(entry.get('run_uuid') in (name, name + '-gpu') for entry in registry['entries'])
             and not any((group / 'runs' / child).exists() for group in completed_groups
                         for child in (name, name + '-gpu')),
             'ONLY_ORIGINAL_UNSTARTED_DEV_ROWS_MAY_CONTINUE')
    return rows


def _closed_prefix(panel, seen=frozenset()):
    from .task_learning_grasp_v1 import need
    from .sla_native_v1 import summarize

    proof = panel['dev16_predecessor']
    prior, inp, result, lease, rows = [pinned(proof[key]) for key in ('panel', 'input', 'result', 'lease', 'outcomes')]
    group = Path(proof['input']['path']).parent
    need(group not in seen and len(seen) < 16, 'FINITE_DEV_PREFIX_CHAIN')
    need(panel['population'] == prior['population'] == 'DEV16'
         and {k: v for k, v in panel.items() if k not in ('source_sha', 'registration', 'dev16_predecessor')}
         == {k: v for k, v in prior.items() if k not in ('source_sha', 'registration', 'dev16_predecessor')}
         and {k: v for k, v in panel['registration'].items()
              if k not in REGISTRATION_PROOF_KEYS | {'source_root', 'source_files'}}
         == {k: v for k, v in prior['registration'].items()
             if k not in REGISTRATION_PROOF_KEYS | {'source_root', 'source_files'}},
         'SAME_ORIGINAL_DEV16_CONDITIONS_POLICY_AND_RUNTIME_INPUTS_REQUIRED')
    earlier, groups, earlier_input = (
        _closed_prefix(prior, seen | {group}) if 'dev16_predecessor' in prior else ([], (), inp))
    start = len(earlier)
    need(group not in groups and inp['native_base'] == earlier_input['native_base']
        and inp['gpu_base'] == earlier_input['gpu_base']
        and all(proof[key]['path'] == str(group / name) for key, name in (
        ('input', 'INPUT.private.json'), ('result', 'RESULT.private.json'),
        ('lease', 'control/LEASE.json'), ('outcomes', 'OUTCOMES.private.json')))
        and inp['panel'] == proof['panel']['path'] and inp['panel_sha256'] == proof['panel']['sha256']
        and inp['source_sha'] == prior['source_sha'] == result['source_sha'] == lease['source_sha']
        and lease['status'] == 'CLOSED' and lease['worker_exit_code'] == result['exit_code'] == 1
        and lease['input_sha256'] == proof['input']['sha256']
        and result['panel_sha256'] == proof['panel']['sha256']
        and result['unresolved_pair'] is None and result['completed_in_group'] == len(rows) - start
        and result['retained_prefix_conditions'] == start and rows[:start] == earlier
        and start < len(rows) < 16 and inp['assignment_slice'] == result['assignment_slice'] == [start, 16]
        and result['summary'] == summarize(prior, rows)
        and not any(row['hard_safety_stop'] or row.get('query_completion_unknown_after_intent') for row in rows),
        'ACTUAL_CLOSED_DEV_PREFIX_WITH_NO_UNRESOLVED_EFFECT_REQUIRED')
    main_pid = subprocess.check_output(['systemctl', '--user', 'show', 'wm02-' + group.name,
                                       '--property=MainPID', '--value'], text=True).strip()
    need(main_pid == '0', 'DEV_PREFIX_GROUP_MUST_BE_TERMINAL')
    for row in rows[start:]:
        for name in (row['native_run'], row['native_run'] + '-gpu'):
            root = group / 'runs' / name
            spec = root / 'EXECUTION_SPEC.json'
            child = json.loads((root / 'control/LEASE.json').read_bytes())
            worker = json.loads((root / 'WORKER_EXIT.private.json').read_bytes())
            need(child['status'] == 'CLOSED' and child['spec_sha256'] == digest(spec)
                 and type(worker['exit_code']) is int and child['worker_exit_code'] == worker['exit_code'],
                 'REAL_EXIT_OF_EACH_PRIOR_NATIVE_AND_GPU_WORKER_REQUIRED')
    return rows, (*groups, group), inp


def select(inp, panel, rows):
    from .task_learning_grasp_v1 import need

    retained = retained_prefix(panel)
    need(inp['source_sha'] == panel['source_sha'] and inp['assignment_slice'] == [len(retained), len(rows)]
         and inp['population_predecessor'] == panel['dev16_predecessor']['result'],
         'EXACT_UNSTARTED_DEV16_SUFFIX_REQUIRED')
    return rows[len(retained):]
