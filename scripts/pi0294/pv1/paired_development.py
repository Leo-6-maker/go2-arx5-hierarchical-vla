"""Read-only V4A comparison of original, audited A/B DEV evidence bytes.

This consumes the existing baseline validator. It neither scores trajectories
again nor changes the native evaluator, selection gate, or running capsules.
"""
from collections import Counter
import json

import numpy as np

from .baseline import partial_history
from .evaluation_plan import development
from scripts.pi0294.dec1.authority import BANK


ARMS = ('A_CONTROL', 'B_COVERAGE')
TASKS = ('T1', 'T2')
BANK_ID = 294101


def compare(population_bytes, *, evidence_by_candidate, bootstrap_draws=2000,
            seed=20261024):
    """Validate supplied raw evidence and keep missing cells out of outcomes.

    Complete means all 48 *supplied* first-panel cells are valid. The caller
    must separately establish the complete native attempt inventory and panel
    closure; this pure reader does not claim to enumerate a remote directory.
    Only the complete 12-geometry panel receives a cluster bootstrap interval.
    Partial paired differences are explicitly exploratory, with no interval.
    """
    if (not isinstance(evidence_by_candidate, dict)
            or set(evidence_by_candidate) != set(ARMS)):
        raise PermissionError('V4A_EXACT_TWO_PRIMARY_CANDIDATE_EVIDENCE_SETS')
    if (type(bootstrap_draws) is not int or not 100 <= bootstrap_draws <= 100000
            or type(seed) is not int or not 0 <= seed < 2**64):
        raise PermissionError('V4A_BOUNDED_EXPLICIT_GEOMETRY_CLUSTER_RESAMPLING')
    histories = {a: partial_history(population_bytes, candidate=a,
                                   evidence=evidence_by_candidate[a]) for a in ARMS}
    population = json.loads(population_bytes)
    panels = {a: development(population['groups'], route='COVERAGE_FIRST',
                             candidate=a, new_bank_manifest_sha256=BANK)
              ['panels'][a + '-DEV-N' + str(BANK_ID)] for a in ARMS}
    cards = {a: {c['condition_id']: c for c in panels[a]['cards']} for a in ARMS}
    if set(cards[ARMS[0]]) != set(cards[ARMS[1]]):
        raise PermissionError('V4A_EXACT_SAME_ORIGINAL_CONDITION_POPULATION')
    groups = sorted({c['scene_group'] for c in cards[ARMS[0]].values()})
    if len(groups) != 12:
        raise PermissionError('V4A_ORIGINAL_TWELVE_GEOMETRY_BLOCKS')
    indexed = {}; cells = []; summaries = {}
    for arm in ARMS:
        rows = histories[arm]['prior_rows']
        if any(r['noise_bank'] != BANK_ID for r in rows):
            raise PermissionError('V4A_FIRST_ORIGINAL_NOISE_BANK_ONLY_NO_SILENT_FILTER')
        valid = {r['condition_id']: r for r in rows
                 if r['status'] in ('SUCCESS', 'VALID_TASK_FAIL')}
        engineering = {r['condition_id'] for r in rows
                       if r['status'] == 'ENGINEERING_INCOMPLETE'}
        indexed[arm] = valid
        summaries[arm] = {}
        for task in TASKS:
            observed = [r for r in valid.values() if r['task'] == task]
            successes = sum(r['status'] == 'SUCCESS' for r in observed)
            summaries[arm][task] = dict(
                planned=12, valid_observed=len(observed), successes=successes,
                valid_task_failures=len(observed)-successes,
                failure_stages=dict(Counter(r['failure_stage'] for r in observed
                                            if r['status'] == 'VALID_TASK_FAIL')),
                observed_success_rate=successes/len(observed) if observed else None,
                complete_panel_success_rate=successes/12 if len(observed) == 12 else None,
                engineering_missing=sum(c['task'] == task and k in engineering and k not in valid
                                        for k, c in cards[arm].items()),
                no_supplied_terminal_evidence=sum(
                    c['task'] == task and k not in engineering and k not in valid
                    for k, c in cards[arm].items()))
        for key, card in sorted(cards[arm].items()):
            row = valid.get(key)
            cells.append(dict(candidate=arm, condition_id=key, task=card['task'],
                              scene_group=card['scene_group'], family=card['family'],
                              noise_bank=BANK_ID,
                              status=row['status'] if row else (
                                  'ENGINEERING_MISSING' if key in engineering
                                  else 'NO_SUPPLIED_TERMINAL_EVIDENCE'),
                              task_success=(row['status'] == 'SUCCESS') if row else None,
                              failure_stage=row['failure_stage'] if row else None,
                              adjudication_sha256=histories[arm]['condition_result_sha256'].get(key)))
    pairs = []; paired = {}
    for key, card in sorted(cards[ARMS[0]].items()):
        if not all(key in indexed[a] for a in ARMS):
            continue
        outcomes = [int(indexed[a][key]['status'] == 'SUCCESS') for a in ARMS]
        pairs.append(dict(condition_id=key, scene_group=card['scene_group'],
                          task=card['task'], A=outcomes[0], B=outcomes[1],
                          B_minus_A=outcomes[1]-outcomes[0]))
    for task in TASKS:
        rows = [r for r in pairs if r['task'] == task]
        paired[task] = dict(pairs=len(rows),
            A_successes=sum(r['A'] for r in rows), B_successes=sum(r['B'] for r in rows),
            B_only_success=sum(r['B_minus_A'] == 1 for r in rows),
            A_only_success=sum(r['B_minus_A'] == -1 for r in rows),
            both_success=sum(r['A'] == r['B'] == 1 for r in rows),
            both_fail=sum(r['A'] == r['B'] == 0 for r in rows),
            descriptive_B_minus_A=(sum(r['B_minus_A'] for r in rows)/len(rows)) if rows else None)
    complete = len(pairs) == 24
    family_counts = {}
    for arm in ARMS:
        family_counts[arm] = {}
        for family in sorted({r['family'] for r in cells}):
            family_counts[arm][family] = {
                task: dict(Counter(r['status'] for r in cells
                                   if (r['candidate'], r['family'], r['task'])
                                   == (arm, family, task))) for task in TASKS}
    result = dict(schema='V4A_ORIGINAL_FIRST_NOISE_PAIRED_DEV_RESULTS_V1',
        status='COMPLETE_SUPPLIED_FIRST_PANEL_PAIR' if complete else 'EXPLORATORY_PARTIAL',
        noise_bank=BANK_ID, planned_cells=48, independent_geometry_blocks=12,
        complete_supplied_pair=complete, by_candidate_task=summaries,
        by_candidate_family_task_status=family_counts,
        paired_by_task=paired, paired_conditions=len(pairs), cells=cells,
        effect_estimates=None, intervals=None, geometry_block_differences=None,
        all_supplied_attempt_hashes={a: histories[a]['all_actual_attempts'] for a in ARMS},
        remote_attempt_inventory_complete_not_proven_here=True,
        absent_terminal_evidence_does_not_prove_unstarted_or_terminated=True,
        native_safety_or_unvalidated_attempts_rejected_not_silently_omitted=True,
        complete_DEV48_per_candidate=False, independent_confirmation=False,
        training_or_selection_admission=False, native_or_resource_admission=False,
        interpretation='ONE_DEVELOPMENT_NOISE_BANK_NOT_GENERALIZATION_OR_MODEL_SELECTION',
        new_integrations=0, new_model_queries=0, new_training_updates=0,
        selected='NONE', deployment='NONE')
    if not complete:
        return result
    blocks = np.asarray([[next(r['B_minus_A'] for r in pairs
                              if r['scene_group'] == group and r['task'] == task)
                          for task in TASKS] for group in groups], dtype=float)
    blocks = np.column_stack((blocks, blocks.mean(axis=1)))
    samples = np.random.default_rng(seed).integers(0, 12, size=(bootstrap_draws, 12))
    means = blocks[samples].mean(axis=1)
    names = ('T1_B_minus_A', 'T2_B_minus_A', 'macro_B_minus_A')
    result.update(effect_estimates=dict(zip(names, blocks.mean(axis=0).tolist())),
        intervals={name: np.quantile(means[:, i], [.025, .975]).tolist()
                   for i, name in enumerate(names)},
        geometry_block_differences=[dict(scene_group=g, **dict(zip(names, row)))
                                    for g, row in zip(groups, blocks.tolist())],
        resampling=dict(unit='PAIRED_GEOMETRY_BOTH_TASKS_BOTH_ARMS_TOGETHER',
                        blocks=12, draws=bootstrap_draws, seed=seed,
                        method='EMPIRICAL_PERCENTILE_95',
                        small_sample_or_degenerate_interval_is_not_equivalence=True))
    return result
