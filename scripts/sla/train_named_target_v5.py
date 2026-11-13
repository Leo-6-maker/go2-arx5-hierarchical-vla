"""Finite first fit of fresh N20; reuse CUDA/RGB qualifications and dense loop."""
from pathlib import Path
import json

from go2arx5_vla.sla_adapter.named_dataset_v5 import OPERATION
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha


def admission(spec):
    if (spec['operation'] != OPERATION
            or spec['microbatch'] != 8 or spec['training_matmul_precision'] != 'highest'
            or spec['closeout_reserve_s'] < 120
            or (spec['planned_candidate_updates'],spec['planned_target_visits'],spec['planned_source_visits'])
               != (3000,96000,0)):
        raise ValueError('CLEAN_BASE_FINITE_N20_FM3K_FIXED_B32_MICRO8_RECIPE_REQUIRED')
    # These are the already CLOSED0 site/RGB and real-gradient operations;
    # new data does not replay camera, SFT2000 or save/restore qualification.
    proof = read_pin(spec['prior_core_RGB_qualification'])
    prior = Path(spec['prior_core_RGB_qualification']['path']).parent
    lease = json.loads((prior/'control/LEASE.json').read_bytes())
    worker = json.loads((prior/'WORKER_EXIT.private.json').read_bytes())
    oldspec = json.loads((prior/'EXECUTION_SPEC.json').read_bytes())
    if (proof.get('schema') != 'SLA_R1_CORE_SITE_RGB32_ACTUAL_V3' or not proof.get('passes')
            or not proof['all_fresh_process_bit_exact'] or proof['ordinary_RGB_generations'] != 32
            or lease['status'] != 'CLOSED' or lease['worker_exit_code'] != worker['exit_code']
            or worker['exit_code'] != 0 or lease['spec_sha256'] != sha(prior/'EXECUTION_SPEC.json')
            or Path(spec['python']).resolve(strict=True) != Path(oldspec['python']).resolve(strict=True)
            or any(spec[k] != oldspec[k] for k in ('GPU_uuid','parent_manifest','parent_params','tokenizer',
                                                  'upstream_root','upstream_sha'))):
        raise ValueError('REUSE_EXACT_CLOSED_CORE_RGB_ENVIRONMENT_FOR_NEW_DATA')
    wrapper = read_pin(spec['dense_gradient_qualification'])
    prior = Path(spec['dense_gradient_qualification']['path']).parent
    lease = json.loads((prior/'control/LEASE.json').read_bytes())
    worker = json.loads((prior/'WORKER_EXIT.private.json').read_bytes())
    grad = read_pin(wrapper['child_result'])
    if (wrapper.get('schema') != 'SLA_DENSE_CORE_MICRO1_QUAL4_ACTUAL_V4_1'
            or not wrapper['passes'] or wrapper['noncandidate_updates'] != 4
            or grad.get('schema') != 'SLA_DENSE_PILOT_FINITE_CUDA_ACTUAL_V4_1'
            or grad['operation'] != 'DENSE_CORE_MICRO1_QUAL4' or grad['exit_code'] != 0
            or grad['step'] != 4 or grad['cursor'] != 128
            or grad['actual']['optimizer_updates'] != 4 or grad['physical_requests'] != 0
            or not grad['frozen_fingerprint_unchanged']
            or lease['status'] != 'CLOSED' or lease['worker_exit_code'] != worker['exit_code']
            or worker['exit_code'] != 0):
        raise ValueError('EXISTING_ACTUAL_DENSE_GRADIENT_NOT_NEW_QUALIFICATION_REQUIRED')
    continuation_prefix(spec)


def continuation_prefix(spec):
    prior = spec.get('continuation')
    if not spec.get('resume'):
        if prior:raise ValueError('N20_CONTINUATION_REQUIRES_SAVED_FULL_STATE')
        return None
    if not prior:raise ValueError('N20_RESUME_REQUIRES_CLOSED75_PREFIX')
    result = read_pin(prior)
    root = Path(prior['path']).parent
    lease = json.loads((root/'control/LEASE.json').read_bytes())
    worker = json.loads((root/'WORKER_EXIT.private.json').read_bytes())
    previous = json.loads((root/'EXECUTION_SPEC.json').read_bytes())
    if (result.get('schema') != 'SLA_CORE_N20_FINITE_PREFIX_ACTUAL_V5'
            or result['exit_code'] != 75 or result['complete3K']
            or lease['status'] != 'CLOSED' or lease['worker_exit_code'] != worker['exit_code']
            or worker['exit_code'] != 75 or lease['spec_sha256'] != sha(root/'EXECUTION_SPEC.json')
            or result['last_checkpoint'] != spec['resume'] or result['physical_requests'] != 0
            or result['source_visits'] != 0 or not 0 < result['step'] < 3000
            or result['target_visits'] != 32*result['step']
            or result['actual']['optimizer_updates'] != result['step']):
        raise ValueError('ONLY_ACTUAL_CLOSED75_N20_PREFIX_MAY_CONTINUE')
    keys=('operation','microbatch','training_matmul_precision','parent_manifest','parent_params',
          'tokenizer','dataset','dense_manifest_sha256','target_audit','initial_function_rule',
          'upstream_sha','planned_candidate_updates','planned_target_visits','planned_source_visits',
          'candidate_lineage','GPU_uuid','python','upstream_root','measure_input_transfer_payloads')
    if any(spec[k] != previous[k] for k in keys):
        raise ValueError('ONE_N20_PARENT_DATA_VISIT_BANK_SCHEDULE_AND_TOTAL3K_BUDGET')
    complete = read_pin(dict(path=str(Path(spec['resume']['path'])/'COMPLETE.json'),
                             sha256=spec['resume']['sha256']))
    if (complete['schema'] != 'SLA_COMPLETE_PEFT_STATE_V1' or len(complete['arrays']) != 87
            or complete['step'] != result['step'] or complete['cursor'] != result['target_visits']):
        raise ValueError('FULL87_N20_ADAM_RNG_COMMITTED_STATE_REQUIRED')
    return result


def run(spec):
    from scripts.sla.train_dense_pilot_v4 import run as dense_run
    from go2arx5_vla.sla_adapter.runtime_named_v5 import NamedTargetEngine
    prefix = continuation_prefix(spec)
    cursor = (prefix['step'],prefix['target_visits']) if prefix else (0,0)
    result = dense_run(spec, engine_class=NamedTargetEngine, admission_check=admission,
                       endpoint_updates=3000, candidate_run=True, expected_resume_cursor=cursor)
    if prefix:
        from scripts.sla.train_dense_spatial_v5 import cumulative_counts
        result['segment_actual'] = result['actual']
        result['actual'] = cumulative_counts(prefix['actual'],result['actual'])
    result.update(schema='SLA_FRESH_N20_FINITE_FM3K_ACTUAL_V5', old_2EP_reused=0,
                  fresh_complete_episodes=20, D100_new=False, D300_new=False,
                  complete3K=result['step']==3000 and result['exit_code']==0)
    if result['complete3K']:
        from scripts.sla import train_dense_pilot_v4 as dense
        from go2arx5_vla.sla_adapter.provider_named_v5 import export
        package=Path(spec['registered_root'])/'RGB_PACKAGE'
        export(package,result['last_checkpoint']['path'],active_template=dense.ACTIVE_ENGINE.active,
               dataset=spec['dataset'],tokenizer=spec['tokenizer']['path'],parent_manifest=spec['parent_manifest'])
        result['package']=dict(path=str(package),sha256=sha(package/'MANIFEST.private.json'))
    return result


if __name__ == '__main__':
    from scripts.sla.train_dense_pilot_v4 import main
    raise SystemExit(main(runner=run))
