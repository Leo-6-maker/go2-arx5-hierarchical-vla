"""One finite clean-base3K spatial candidate; no optimizer or weight warm start."""
import argparse
import json
from pathlib import Path
import signal
import time

from scripts.pi0292.durable import write
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.checkpoint_v1 import save,sha

ENGINE = None


def run(spec):
    from go2arx5_vla.sla_adapter.spatial_runtime_v5 import DenseSpatialEngine,OPERATION,common_optimizer
    q = read_pin(spec['spatial_qualification'])
    old = Path(spec['spatial_qualification']['path']).parent
    lease = json.loads((old/'control/LEASE.json').read_bytes())
    if (q['schema'] != 'SLA_DENSE_SPATIAL_SEAM_4UPDATES_ACTUAL_V5' or not q['passes']
            or not q['clean_initial_all_arrays_bit_exact'] or q['initial_arrays'] != 105
            or q['noncandidate_updates'] != 4 or q['target_visits'] != 128 or q['reference_FM_visits'] != 64
            or lease['status'] != 'CLOSED' or lease['worker_exit_code'] != 0
            or spec['operation'] != OPERATION or spec['spatial_alpha'] != .5
            or spec['microbatch'] != 8 or spec['training_matmul_precision'] != 'highest' or spec.get('resume')):
        raise ValueError('ACTUAL_ALPHA0_AND_POSITIVE_GRADIENT_CLEAN_BASE_BEFORE_MATCHED3K')
    from go2arx5_vla.sla_adapter.site_v2 import validate
    site = validate(spec)
    from scripts.sla.train_v1 import probes
    root = Path(spec['registered_root']);global ENGINE
    ENGINE = DenseSpatialEngine(dict(spec,deadline_epoch=spec['deadline_epoch']-spec['closeout_reserve_s']),root/'work')
    e = ENGINE;initial = e.checkpoint('step0000')
    # Compare the common parent/LoRA/Adam/RNG bytes to the already closed
    # Core dense qualification, without replaying its gradient qualification.
    common = save(e.output/'common-initial',(e.active,common_optimizer(e.opt_state),e.rng),
                  step=0,cursor=0,bindings=e.bindings)
    actual = json.loads((Path(common['path'])/'COMPLETE.json').read_bytes())
    previous = read_pin(spec['common_initial_reference'])
    if [(r['key'],r['sha256']) for r in actual['arrays']] != [(r['key'],r['sha256']) for r in previous['arrays']]:
        raise ValueError('MATCHED_COMMON_INITIAL_PARENT_LORA_ADAM_RNG_BYTES_CHANGED')
    write(root/'COMMON_INITIAL_ACTUAL.private.json',dict(passes=True,common_arrays=len(actual['arrays']),
          checkpoint=common,reference=spec['common_initial_reference'],bit_exact=True,candidate_updates=0))
    result=dict(schema='SLA_CLEAN_BASE_DENSE_SPATIAL3K_ACTUAL_V5',source_sha=spec['source_sha'],
        requested_updates=3000,initial_checkpoint=initial,common_initial_checkpoint=common,
        initial_step=0,initial_cursor=0,physical_requests=0,source_visits=0,D100_new=False,PICK_PLACE_admitted=False)
    last = initial;began = time.time()
    try:
        while e.step < 3000:
            if time.time() >= e.spec['deadline_epoch']:raise TimeoutError('FINITE_SPATIAL_CANDIDATE_CLOSEOUT')
            row = e.update()
            print(json.dumps(dict(event='ACTUAL_SPATIAL_CANDIDATE_UPDATE',**row)),flush=True)
            if e.step in (500,1500,3000):last = e.checkpoint(f'step{e.step:04d}')
            if e.step in (500,1500,3000):result[f'RGB_step{e.step}'] = probes(e,f'step{e.step}-RGB')
    except TimeoutError:
        last = e.checkpoint(f'LIMIT-step{e.step:04d}')
        result.update(exit_code=75,status='FINITE_LIMIT_SAVED_COMMITTED_STATE_NOT_3K_COMPLETE')
    else:
        from go2arx5_vla.sla_adapter.provider_v5 import export
        package = root/'RGB_PACKAGE'
        export(package,last['path'],active_template=e.active,dataset=spec['dataset'],
               tokenizer=spec['tokenizer']['path'],parent_manifest=spec['parent_manifest'])
        result.update(exit_code=0,status='COMPLETE_SPATIAL3K_NOT_NATIVE_CAPABILITY',
                      package=dict(path=str(package),sha256=sha(package/'MANIFEST.private.json')))
    result.update(step=e.step,cursor=e.cursor,actual=e.counts,last_checkpoint=last,
        frozen_fingerprint=e.verify_frozen(),residency=e.residency(),site=site,wall_s=time.time()-began)
    write(root/'RESULT.private.json',result);e.close();return result['exit_code']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('spec',type=Path);a=p.parse_args()
    spec=json.loads(a.spec.read_bytes());root=Path(spec['registered_root'])
    if (root/'RESULT.private.json').exists() or (root/'FAILED.private.json').exists():raise FileExistsError('NO_SPATIAL_CANDIDATE_REPLAY')
    def stop(sig,frame):
        if ENGINE is not None:ENGINE.spec['deadline_epoch']=time.time()
        else:raise TimeoutError('FINITE_SPATIAL_LOAD_INTERRUPTED')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:return run(spec)
    except BaseException as error:
        write(root/'FAILED.private.json',dict(error=repr(error),actual=getattr(ENGINE,'counts',{}),
            committed_step=getattr(ENGINE,'step',0),cursor=getattr(ENGINE,'cursor',0),physical_requests=0,
            latest_complete_state_must_be_read_independently=True))
        if ENGINE is not None:ENGINE.close()
        raise


if __name__=='__main__':raise SystemExit(main())
