"""Immutable V5.2 routing, retaining the reviewed execution parent closure."""
import hashlib
import json
from pathlib import Path
import sys


def main():
    operation,path=sys.argv[1:3];spec=json.loads(Path(path).read_bytes())
    if spec.get('plan_commit')!='90d39196bdc345b977ee54ed64e750cf3561aa72'or spec.get('scope')!='SLA_MOBILE_F2_V52':raise PermissionError('PINNED_V52_PLAN_AND_SCOPE')
    root=Path(spec['source_root']);parent=spec['trusted_source_parent']
    if Path(__file__).resolve()!=root/'scripts/wm02/mobile_v52_bootstrap.py':raise PermissionError('EXACT_V52_BOOTSTRAP')
    for name,expected in spec['source_files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:raise ValueError('V52_SOURCE_CHANGED:'+name)
    if hashlib.sha256(Path(parent['spec_path']).read_bytes()).hexdigest()!=parent['spec_sha256']:raise ValueError('EXECUTION_PARENT_CHANGED')
    sys.path[:0]=[str(root),str(root/'src'),parent['source_root'],parent['source_root']+'/src']
    import go2arx5_vla.world_model,go2arx5_vla.sla_adapter,scripts.wm02,scripts.sla
    for module,rel in ((go2arx5_vla.world_model,'src/go2arx5_vla/world_model'),(go2arx5_vla.sla_adapter,'src/go2arx5_vla/sla_adapter'),(scripts.wm02,'scripts/wm02'),(scripts.sla,'scripts/sla')):module.__path__=[str(root/rel),*list(module.__path__)]
    from scripts.wm02.mobile_boundary_runner_v1 import registered
    registered(spec,path)
    if operation=='--followup':
        from scripts.wm02.mobile_e1_followup_v52 import main as followup
        raise SystemExit(followup(spec))
    if operation=='--campaign':
        from scripts.wm02.mobile_campaign_v52 import run as campaign
        raise SystemExit(campaign(path))
    if operation=='--case':
        from scripts.wm02.mobile_boundary_runner_v1 import run_case
        fd=int(sys.argv[sys.argv.index('--world-lock-fd')+1])
        raise SystemExit(run_case(path,fd))
    if operation=='--worker':
        if spec['physics']or spec['new_physical_requests_budget']:raise PermissionError('NEURAL_V52_NO_NATIVE')
        import runpy
        runpy.run_module('scripts.sla.hybrid_ipc_worker_v3',run_name='__main__')
        return
    if operation=='--offline':
        from scripts.sla.mobile_offline_v52 import main as offline
        raise SystemExit(offline(spec))
    if operation in('--train','--rgb'):
        if spec['physics']or spec['new_physical_requests_budget']:raise PermissionError('OFFLINE_V52_NO_NATIVE')
        from scripts.sla.train_mobile_v52 import main as learner
        raise SystemExit(learner(operation,spec))
    if operation not in('--d0','--prepare'):raise PermissionError('UNIMPLEMENTED_V52_ROUTE')
    if spec['physics']or spec['new_physical_requests_budget']or spec['new_model_generations_budget']or spec['optimizer_updates']:raise PermissionError('D0_MUST_BE_ZERO_GENERATION_AND_INTEGRATION')
    if operation=='--d0':
        from scripts.wm02.mobile_d0_v52 import run
    else:
        from scripts.wm02.mobile_prepare_v52 import run
    from scripts.pi0292.durable import write
    from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close
    import traceback
    try:result=run(spec);code=result['exit_code']
    except BaseException:
        code=1;write(Path(spec['registered_root'])/'RESULT.private.json',dict(exit_code=1,failure=traceback.format_exc(),new_generations=0,new_integrations=0,new_training_updates=0))
    close(spec,code);raise SystemExit(code)


if __name__=='__main__':main()
