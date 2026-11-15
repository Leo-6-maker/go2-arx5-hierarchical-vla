"""Frozen S16 visit bank for offline fitting; grants no native execution.

This preparatory recipe does not start a trainer or change RP8's outcome.
Every effective batch visits each of the fixed sixteen windows twice; the
existing learner's absolute visit cursor supplies independent flow noise.
"""
import hashlib
import json

OPERATION = 'MICROFIT_S16_REFERENCE500_OFFLINE_V7'
DATA_SHA = '29fd45ff31a2031f70f5d32402e6bfefe2f8b644e8e81ad1d2003d4537a1a6f5'
STATS_SHA = '024951afc22e03586635f17b5ee8f473aead1960f107d8768384643844afb9e1'
UPDATES, BATCH = 500, 32


def fixed_indices(manifest, selection):
    if (selection['data_manifest_sha256'] != DATA_SHA
            or selection['statistics_sha256'] != STATS_SHA
            or manifest['statistics']['sha256'] != STATS_SHA
            or selection['refit_statistics'] is not False
            or selection['inference_noise_ids'] != [0, 1]):
        raise ValueError('ORIGINAL_FROZEN_N20_AND_PRESELECTED_S16_ONLY')
    rows = [(ep, anchor) for ep in manifest['episodes'] for anchor in range(61)]
    lookup = {(ep['id'], anchor): i for i, (ep, anchor) in enumerate(rows)}
    selected = selection['S16']
    if len(selected) != 16 or len({(r['episode'], r['anchor']) for r in selected}) != 16:
        raise ValueError('SIXTEEN_DISTINCT_WINDOWS_REQUIRED')
    result = []
    for row in selected:
        index = lookup[row['episode'], row['anchor']]
        ep, anchor = rows[index]
        if (ep['split'] != 'TRAIN' or row['native_step'] != 5703 + anchor * 250
                or ep['layout'] != row['layout'] or ep['expert_stage_audit_only'][anchor] != row['stage']):
            raise ValueError('ORIGINAL_FIXED_TRAIN_WINDOW_IDENTITY_REQUIRED')
        result.append(index)
    return tuple(result)


def visit_bank(indices):
    if len(indices) != 16 or len(set(indices)) != 16 or any(type(i) is not int or i < 0 for i in indices):
        raise ValueError('FIXED_DISTINCT_S16_ROW_INDICES_REQUIRED')
    return tuple(indices) * (UPDATES * 2)


def prepare(manifest, selection):
    indices = fixed_indices(manifest, selection)
    # Digest the declared sequence here; the learner must also bind its int64
    # array bytes, as in the existing complete-checkpoint contract.
    bank = visit_bank(indices)
    return dict(operation=OPERATION, selected_indices=indices,
        data_manifest_sha256=DATA_SHA, statistics_sha256=STATS_SHA,
        selected_windows=16, episodes=len({r['episode'] for r in selection['S16']}),
        planned_updates=UPDATES, effective_batch=BATCH, planned_target_visits=len(bank),
        visits_per_selected_window=1000,
        visit_sequence_json_sha256=hashlib.sha256(json.dumps(bank, separators=(',', ':')).encode()).hexdigest(),
        checkpoints=[0, 100, 500], warmup=25, microbatch=8, accumulation=4,
        inference_windows=16, inference_noise_ids=[0, 1], planned_predictions=32,
        requires_RP8_for_offline_fitting=False, native_execution_admitted=False,
        training_worker_implemented=True, training_registered=False,
        optimizer_updates=0, actual_training_visits=0, physical_requests=0,
        refit_statistics=False, native_gates_changed=False)
