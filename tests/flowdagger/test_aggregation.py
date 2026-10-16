import copy

import pytest

from go2arx5_vla.flowdagger.aggregation import aggregate
from go2arx5_vla.flowdagger.contracts import digest, label_record
from .test_contracts import fixture


def pool(rows, cap=10):
    return aggregate(rows, max_records=cap, model_sha256='b'*64, statistics_sha256='c'*64)


def test_same_pool_for_both_baselines_not_original_committed_BC_labels():
    row=label_record(**fixture());before=copy.deepcopy(row)
    value=pool([row])
    assert value['training_label_count']==1 and value['analysis_record_count']==1
    example=value['training_examples'][0]
    assert example['label_representation']=='COUNTERFACTUAL_EXPERT_CORRECTION_EEF10'
    assert example['actually_executed_by']=='STUDENT'
    assert value['shared_label_pool_for']==['ACTION_SPACE_DAGGER','FLOWDAGGER_LATENT_STEERING']
    assert row==before and not value['original_success_BC_bank_modified']


def test_dev_kept_in_analysis_not_optimizer():
    values=fixture();values.update(split='DEV',lineage_splits=['DEV'])
    row=label_record(**values);value=pool([row])
    assert value['analysis_record_count']==1 and value['training_label_count']==0


def test_corruption_duplicate_and_budget_fail_closed():
    row=label_record(**fixture())
    with pytest.raises(ValueError):pool([row,row])
    with pytest.raises(PermissionError):pool([row,row],cap=1)
    row['expert_action'][0][0]+=.1
    with pytest.raises(ValueError):pool([row])


def test_resealed_dev_to_train_flag_still_cannot_bypass_lineage():
    values=fixture();values.update(split='DEV',lineage_splits=['DEV'])
    row=label_record(**values);row.update(split='TRAIN',optimizer_eligible=True)
    row.pop('record_sha256');row['record_sha256']=digest(row)
    with pytest.raises(ValueError):pool([row])


def test_wrong_frozen_identity_refused():
    row=label_record(**fixture())
    with pytest.raises(ValueError):aggregate([row],max_records=10,model_sha256='f'*64,statistics_sha256='c'*64)
