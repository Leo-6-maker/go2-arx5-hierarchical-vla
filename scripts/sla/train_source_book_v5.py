"""One finite500-update clean-base source BOOK fit; reuse existing GPU proofs."""
import json
from pathlib import Path

from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.runtime_book_v5 import OPERATION,RECIPE,SourceBookEngine,book_probes


def admission(spec):
    if (spec['operation']!=OPERATION or spec.get('resume') or spec['microbatch']!=8
            or spec['training_matmul_precision']!='highest' or spec['closeout_reserve_s']<120
            or (spec['planned_candidate_updates'],spec['planned_target_visits'],spec['planned_source_visits'])!=(500,0,16000)):
        raise ValueError('FINITE_SOURCE_BOOK_CLEAN_BASE500_B32_MICRO8_HIGHEST_REQUIRED')
    prior=Path(spec['prior_closed_GPU0_training']['path']).parent
    wrapper=read_pin(spec['prior_closed_GPU0_training']);old=json.loads((prior/'EXECUTION_SPEC.json').read_bytes())
    lease=json.loads((prior/'control/LEASE.json').read_bytes());worker=json.loads((prior/'WORKER_EXIT.private.json').read_bytes())
    trained=json.loads((prior/'candidate3000/RESULT.private.json').read_bytes())
    rgb=read_pin(spec['prior_GPU0_RGB500'])
    complete500=read_pin(spec['prior_GPU0_complete500'])
    if (lease['status']!='CLOSED' or lease['worker_exit_code']!=worker['exit_code'] or worker['exit_code']not in(0,75)
            or lease['spec_sha256']!=sha(prior/'EXECUTION_SPEC.json')
            or complete500['step']!=500 or len(complete500['arrays'])!=105
            or trained['frozen_fingerprint']!=complete500['bindings']['frozen_fingerprint']
            or trained['step']<500 or trained['actual']['optimizer_updates']<500
            or len(rgb)!=16 or wrapper.get('physical_requests')!=0
            or any(spec[k]!=old[k]for k in('GPU_uuid','parent_manifest','parent_params','tokenizer','upstream_root','upstream_sha'))
            or Path(spec['python']).resolve(strict=True)!=Path(old['python']).resolve(strict=True)):
        raise ValueError('REUSE_ACTUAL_CLOSED_SAME_GPU0_PARENT_RGB_GRADIENT_PROOFS_NO_REPEAT')


def run(spec):
    from scripts.sla import train_dense_pilot_v4 as dense
    result=dense.run(spec,engine_class=SourceBookEngine,admission_check=admission,endpoint_updates=500,
                     candidate_run=True,probe_function=book_probes)
    result.update(schema='SLA_BOOK21_SOURCE_FINITE_FM500_ACTUAL_V5',recipe=RECIPE,
        target_visits=0,source_visits=result['cursor'],source_only=True,target_WBC_qualified=False,
        complete500=result['step']==500 and result['exit_code']==0)
    if result['complete500']:
        from go2arx5_vla.sla_adapter.provider_v3 import write_verified_package
        from go2arx5_vla.sla_adapter.source_book_v5 import manifest_contract
        dataset=Path(spec['dataset']);m=json.loads((dataset/'MANIFEST.private.json').read_bytes());manifest_contract(m)
        cp=Path(result['last_checkpoint']['path']);complete=json.loads((cp/'COMPLETE.json').read_bytes())
        if len(complete['arrays'])!=87 or complete['step']!=500 or complete['cursor']!=16000 or complete['bindings']['recipe']!=RECIPE:
            raise ValueError('ACTUAL_BOOK500_FULL87_ARRAY_STATE_REQUIRED')
        package=Path(spec['registered_root'])/'RGB_PACKAGE'
        write_verified_package(package,cp,complete=complete,active_template=dense.ACTIVE_ENGINE.active,
            statistics=dataset/m['statistics']['file'],tokenizer=spec['tokenizer']['path'],parent_manifest=spec['parent_manifest'])
        result.update(package=dict(path=str(package),sha256=sha(package/'MANIFEST.private.json')),endpoint_RGB=result['RGB_step500'])
    return result


if __name__=='__main__':
    from scripts.sla.train_dense_pilot_v4 import main
    raise SystemExit(main(runner=run))
