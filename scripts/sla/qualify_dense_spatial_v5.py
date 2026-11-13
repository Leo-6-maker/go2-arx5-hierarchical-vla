"""Two noncandidate updates in each clean-parent dense spatial seam arm."""
import argparse
import json
from pathlib import Path
import time

from scripts.pi0292.durable import write
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha


def run(spec):
    proof = read_pin(spec['dense_core_qualification'])
    cache = read_pin(spec['dense_cache_preparation'])
    if (proof['schema'] != 'SLA_DENSE_CORE_MICRO1_QUAL4_ACTUAL_V4_1' or not proof['passes']
            or proof['noncandidate_updates'] != 4 or proof['optimization_visits'] != 128
            or not cache['passes'] or cache['teacher_cache'] != spec['teacher_cache']
            or cache['actual']['complete_cached_TRAIN_windows'] != 122
            or spec['operation'] != 'DENSE_CORE_MICRO1_QUAL4' or spec['microbatch'] != 1
            or spec['spatial_alpha'] not in (0.,.5) or spec.get('resume')):
        raise ValueError('REUSE_CLOSED_DENSE_QUALIFICATION_AND_FROZEN_CACHE_CLEAN_PARENT_SEAM')
    from go2arx5_vla.sla_adapter.site_v2 import validate
    site = validate(spec)
    from go2arx5_vla.sla_adapter.spatial_runtime_v5 import DenseSpatialEngine
    root = Path(spec['registered_root']);engine = None;began = time.time()
    try:
        engine = DenseSpatialEngine(dict(spec,deadline_epoch=spec['deadline_epoch']-120),root/'work')
        initial = engine.checkpoint('step0000')
        for _ in range(2):
            row = engine.update();print(json.dumps(row),flush=True)
        endpoint = engine.checkpoint('step0002')
        if engine.counts['optimizer_updates'] != 2 or engine.counts['completed_training_visits'] != 64:
            raise ValueError('TWO_REAL_UPDATES_AND_64_TARGET_VISITS_REQUIRED')
        if spec['spatial_alpha'] == 0 and (engine.counts['alpha0_gradient_pairs_compared'] != 64
                                         or engine.counts['alpha0_update_pairs_compared'] != 2):
            raise ValueError('ALL_COMMON_GRADIENT_AND_ADAM_UPDATE_PAIRS_REQUIRED')
        result = dict(schema='SLA_DENSE_SPATIAL_SEAM_ARM_ACTUAL_V5',passes=True,alpha=spec['spatial_alpha'],
            actual=engine.counts,step=engine.step,cursor=engine.cursor,initial_checkpoint=initial,last_checkpoint=endpoint,
            frozen_fingerprint=engine.verify_frozen(),residency=engine.residency(),site=site,
            physical_requests=0,candidate_updates=0,wall_s=time.time()-began)
        write(root/'RESULT.private.json',result)
    except BaseException as error:
        write(root/'FAILED.private.json',dict(error=repr(error),actual=getattr(engine,'counts',{}),
            step=getattr(engine,'step',0),cursor=getattr(engine,'cursor',0),physical_requests=0))
        raise
    finally:
        if engine is not None:engine.close()


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('spec',type=Path);a=p.parse_args()
    spec=json.loads(a.spec.read_bytes());root=Path(spec['registered_root'])
    if (root/'RESULT.private.json').exists() or (root/'FAILED.private.json').exists():
        raise FileExistsError('UNKNOWN_OR_TERMINAL_SPATIAL_SEAM_NEVER_REPLAY')
    run(spec)
