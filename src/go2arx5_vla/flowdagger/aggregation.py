"""Bounded immutable label index shared by action-space and latent baselines."""
import copy

from .codec import eef10
from .contracts import digest, summarize


def aggregate(records, *, max_records, model_sha256, statistics_sha256):
    if type(max_records) is not int or max_records <= 0:
        raise ValueError('EXPLICIT_POSITIVE_AGGREGATION_BOUND_REQUIRED')
    rows, seen, training = [], set(), []
    for original in records:
        if len(rows) >= max_records:
            raise PermissionError('NO_SILENT_RING_BUFFER_EVICTION_OR_UNBOUNDED_AGGREGATION')
        row = copy.deepcopy(original)
        claimed = row.pop('record_sha256')
        if digest(row) != claimed:
            raise ValueError('CORRECTION_RECORD_CHANGED')
        if (row['schema'] != 'DAGGER_CORRECTION_LABEL_V1'
                or row['source_domain'] != 'FIXED_BASE_ARX5_TASK_SIM'
                or row['model_sha256'] != model_sha256 or row['statistics_sha256'] != statistics_sha256
                or row['split'] not in ('TRAIN', 'DEV') or not row['lineage_splits']
                or any(s not in ('TRAIN', 'DEV') for s in row['lineage_splits'])
                or type(row['expert_valid']) is not bool):
            raise ValueError('NO_DOMAIN_MODEL_STATS_OR_SPLIT_SWITCH')
        identity = row['observation_id']
        if identity in seen:
            raise ValueError('DUPLICATE_OBSERVATION_NO_LABEL_MULTIPLICATION')
        seen.add(identity)
        eligible = row['expert_valid'] and row['split'] == 'TRAIN'
        if row['optimizer_eligible'] is not eligible or row['split'] == 'TRAIN' and any(s != 'TRAIN' for s in row['lineage_splits']):
            raise ValueError('NO_DEV_OR_INVALID_LABEL_IN_OPTIMIZER')
        if row['expert_valid']:
            eef10(row['expert_action'])
            if not row['expert_queried'] or not row['safety_proofs'].get('expert_safety_proof_sha256'):
                raise ValueError('VALID_CORRECTION_REQUIRES_QUERY_AND_SAFETY_PROVENANCE')
        if eligible:
            training.append(dict(observation_id=identity, label_record_sha256=claimed,
                expert_absolute_H40_eef10=row['expert_action'],
                label_representation='COUNTERFACTUAL_EXPERT_CORRECTION_EEF10',
                actually_executed_by=row['executed_by'], student_visited_state=row['student_visited_state']))
        row['record_sha256'] = claimed
        rows.append(row)
    value = dict(schema='DAGGER_CORRECTION_POOL_V1', status='INDEX_ONLY_NOT_A_TRAINING_OR_MODEL_RESULT',
        source_domain='FIXED_BASE_ARX5_TASK_SIM', model_sha256=model_sha256, statistics_sha256=statistics_sha256,
        record_sha256s=[r['record_sha256'] for r in rows], training_examples=training, summary=summarize(rows),
        training_label_count=len(training), analysis_record_count=len(rows),
        shared_label_pool_for=['ACTION_SPACE_DAGGER', 'FLOWDAGGER_LATENT_STEERING'],
        original_success_BC_bank_modified=False, selected='NONE', deployment='NONE')
    value['pool_sha256'] = digest(value)
    return value
