import json

import pytest

from scripts.sla.train_named_target_v5 import continuation_prefix
from scripts.sla.train_dense_spatial_v5 import cumulative_counts
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return dict(path=str(path), sha256=sha(path))


def finite_prefix(tmp_path):
    old = tmp_path/'old'
    complete = old/'candidate3000/work/LIMIT-step0500/COMPLETE.json'
    cp = save(complete, dict(schema='SLA_COMPLETE_PEFT_STATE_V1', step=500,
                            cursor=16000, arrays=[{}]*87))
    resume = dict(path=str(complete.parent), sha256=cp['sha256'])
    keys = ('operation','microbatch','training_matmul_precision','parent_manifest','parent_params',
            'tokenizer','dataset','dense_manifest_sha256','target_audit','initial_function_rule',
            'upstream_sha','planned_candidate_updates','planned_target_visits','planned_source_visits',
            'candidate_lineage','GPU_uuid','python','upstream_root','measure_input_transfer_payloads')
    spec = {key:key for key in keys}
    spec.update(planned_candidate_updates=3000, planned_target_visits=96000, planned_source_visits=0)
    pinned_spec = save(old/'EXECUTION_SPEC.json', spec)
    training_spec = save(old/'candidate3000/EXECUTION_SPEC.json', spec)
    save(old/'control/LEASE.json', dict(status='CLOSED', worker_exit_code=75,
                                      spec_sha256=pinned_spec['sha256']))
    save(old/'WORKER_EXIT.private.json', dict(exit_code=75))
    result = dict(schema='SLA_CORE_N20_FINITE_PREFIX_ACTUAL_V5', exit_code=75,
                  complete3K=False, step=500, target_visits=16000, source_visits=0,
                  physical_requests=0, last_checkpoint=resume,
                  training_spec=training_spec,
                  actual=dict(optimizer_updates=500, completed_training_visits=16008,
                              actual_parent_loads=1))
    spec.update(resume=resume, continuation=save(old/'RESULT.private.json', result))
    return spec


def test_same_logical_budget_retains_discarded_microbatch_cost(tmp_path):
    spec = finite_prefix(tmp_path)
    prefix = continuation_prefix(spec)
    actual = cumulative_counts(prefix['actual'], dict(optimizer_updates=2500,
        completed_training_visits=80000, actual_parent_loads=1))
    assert actual == dict(optimizer_updates=3000, completed_training_visits=96008,
                         actual_parent_loads=2)
    assert prefix['actual']['optimizer_updates'] == 500


def test_running_or_unknown_predecessor_cannot_resume(tmp_path):
    spec = finite_prefix(tmp_path)
    old = tmp_path/'old'
    save(old/'control/LEASE.json', dict(status='ACTIVE', worker_exit_code=75,
                                      spec_sha256=sha(old/'EXECUTION_SPEC.json')))
    with pytest.raises(ValueError, match='CLOSED75'):
        continuation_prefix(spec)


@pytest.mark.parametrize('key,value', [('dataset','changed-data'),
                                      ('planned_candidate_updates',3500), ('microbatch',1)])
def test_continuation_cannot_change_data_or_add_another_budget(tmp_path, key, value):
    spec = finite_prefix(tmp_path)
    spec[key] = value
    with pytest.raises(ValueError, match='ONE_N20_PARENT_DATA_VISIT_BANK_SCHEDULE_AND_TOTAL3K_BUDGET'):
        continuation_prefix(spec)
