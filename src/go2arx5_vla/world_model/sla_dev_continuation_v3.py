"""Continue the untouched suffix of a fixed SLA DEV16 after a closed group."""
import json
from pathlib import Path
import subprocess

from .stable_task_scene_v1 import pinned
from .storage import digest

SCHEMA = 'SLA_DEV16_COLD_SUFFIX_INPUT_V3'


def retained_prefix(panel):
    from scripts.pi0294.e1.journal import Journal
    from .task_learning_grasp_v1 import need
    from .sla_native_v1 import summarize

    proof = panel['dev16_predecessor']
    prior, inp, result, lease, rows = [pinned(proof[key]) for key in ('panel', 'input', 'result', 'lease', 'outcomes')]
    group = Path(proof['input']['path']).parent
    need(panel['population'] == prior['population'] == 'DEV16'
         and {k: v for k, v in panel.items() if k not in ('source_sha', 'registration', 'dev16_predecessor')}
         == {k: v for k, v in prior.items() if k not in ('source_sha', 'registration')}
         and {k: v for k, v in panel['registration'].items() if k not in ('source_root', 'source_files')}
         == {k: v for k, v in prior['registration'].items() if k not in ('source_root', 'source_files')},
         'SAME_ORIGINAL_DEV16_CONDITIONS_POLICY_AND_RUNTIME_INPUTS_REQUIRED')
    need(all(proof[key]['path'] == str(group / name) for key, name in (
        ('input', 'INPUT.private.json'), ('result', 'RESULT.private.json'),
        ('lease', 'control/LEASE.json'), ('outcomes', 'OUTCOMES.private.json')))
        and inp['panel'] == proof['panel']['path'] and inp['panel_sha256'] == proof['panel']['sha256']
        and inp['source_sha'] == prior['source_sha'] == result['source_sha'] == lease['source_sha']
        and lease['status'] == 'CLOSED' and lease['worker_exit_code'] == result['exit_code'] == 1
        and lease['input_sha256'] == proof['input']['sha256']
        and result['panel_sha256'] == proof['panel']['sha256']
        and result['unresolved_pair'] is None and result['completed_in_group'] == len(rows)
        and 0 < len(rows) < 16 and inp['assignment_slice'] == result['assignment_slice'] == [0, 16]
        and result['summary'] == summarize(prior, rows)
        and not any(row['hard_safety_stop'] or row.get('query_completion_unknown_after_intent') for row in rows),
        'ACTUAL_CLOSED_DEV_PREFIX_WITH_NO_UNRESOLVED_EFFECT_REQUIRED')
    main_pid = subprocess.check_output(['systemctl', '--user', 'show', 'wm02-' + group.name,
                                       '--property=MainPID', '--value'], text=True).strip()
    need(main_pid == '0', 'DEV_PREFIX_GROUP_MUST_BE_TERMINAL')
    for row in rows:
        for name in (row['native_run'], row['native_run'] + '-gpu'):
            root = group / 'runs' / name
            spec = root / 'EXECUTION_SPEC.json'
            child = json.loads((root / 'control/LEASE.json').read_bytes())
            worker = json.loads((root / 'WORKER_EXIT.private.json').read_bytes())
            need(child['status'] == 'CLOSED' and child['spec_sha256'] == digest(spec)
                 and type(worker['exit_code']) is int and child['worker_exit_code'] == worker['exit_code'],
                 'REAL_EXIT_OF_EACH_PRIOR_NATIVE_AND_GPU_WORKER_REQUIRED')
    native_base = pinned(inp['native_base'])
    campaign = Path(native_base['campaign_root'])
    journal = Journal(campaign / 'control/journal', campaign.name)
    registry = json.loads(Path(native_base['registry']).read_bytes())
    names = {row['native_run'] for row in panel['assignments'][len(rows):]}
    for name in names:
        key = name.upper()
        need(not any(journal.path(key + suffix).exists() for suffix in ('-REGISTER', '-PAIR-REGISTRY', '-RUN', '-GPU-RUN'))
             and not any(entry.get('run_uuid') in (name, name + '-gpu') for entry in registry['entries'])
             and not any((group / 'runs' / child).exists() for child in (name, name + '-gpu')),
             'ONLY_ORIGINAL_UNSTARTED_DEV_ROWS_MAY_CONTINUE')
    return rows


def select(inp, panel, rows):
    from .task_learning_grasp_v1 import need

    retained = retained_prefix(panel)
    need(inp['source_sha'] == panel['source_sha'] and inp['assignment_slice'] == [len(retained), len(rows)]
         and inp['population_predecessor'] == panel['dev16_predecessor']['result'],
         'EXACT_UNSTARTED_DEV16_SUFFIX_REQUIRED')
    return rows[len(retained):]
