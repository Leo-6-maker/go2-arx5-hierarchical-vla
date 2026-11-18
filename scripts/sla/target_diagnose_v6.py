"""One real four-window zero-update V6 full-trajectory gradient diagnosis."""
import json
from pathlib import Path
import signal
import time
import traceback


def run(spec):
    from scripts.pi0292.durable import write
    from scripts.sla.close_goal_g1_v1 import wait_resource_owner
    from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close
    from go2arx5_vla.sla_adapter.site_v2 import validate
    from go2arx5_vla.sla_adapter.target_training_v6 import TargetEngine
    root=Path(spec['registered_root']);engine=None;lock=None;start=time.monotonic();cpu=time.process_time()
    def stop(*_):raise TimeoutError('FINITE_V6_DIAGNOSTIC_STOP')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if(spec['phase']!='T_DIAGNOSTIC_1' or spec['current_operation_optimizer_update_cap']!=0
       or spec['new_physical_requests_budget']or spec['new_model_generations_budget']!=12):
        raise PermissionError('FIRST_ZERO_UPDATE_FOUR_WINDOW_DIAGNOSIS_ONLY')
    try:
        lock=wait_resource_owner(spec);site=validate(spec)
        write(root/'PROGRESS.private.json',dict(epoch=time.time(),phase='LOAD_B1_FOR_DIAGNOSTIC',new_updates=0,new_native_requests=0))
        engine=TargetEngine(spec,root/'work')
        write(root/'PROGRESS.private.json',dict(epoch=time.time(),phase='REAL_FOUR_WINDOW_GRADIENT',new_updates=0,new_native_requests=0))
        result=engine.diagnose()
        result.update(exit_code=0,status='REAL_FULL_TEN_STEP_GRADIENT_AND_REFERENCE_COMPARISON_PASSED',
            source_sha=spec['source_sha'],plan_commit=spec['plan_commit'],bindings=engine.bindings,actual=engine.counts,
            site=site,new_updates=0,new_visits=0,new_native_requests=0,wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu)
    except BaseException:
        result=dict(exit_code=1,status='PRESERVED_DIAGNOSTIC_FAILURE_NO_REPLAY',failure=traceback.format_exc(),source_sha=spec['source_sha'],
            actual=engine.counts if engine else None,new_updates=0,new_visits=0,new_native_requests=0,wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu)
    finally:
        if engine:engine.close()
        if lock:lock.close()
    write(root/'RESULT.private.json',result);close(spec,result['exit_code']);return result['exit_code']
