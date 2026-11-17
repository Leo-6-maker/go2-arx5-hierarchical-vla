"""Fixed V6 source/parent closure, independent of all closed capsules."""
import hashlib,json
from pathlib import Path
import sys


def main():
    operation,path=sys.argv[1:3];spec=json.loads(Path(path).read_bytes());source=Path(spec['source_root']);parent=spec['trusted_source_parent']
    if(spec['plan_commit']!='ae3d3c0dceade437b0104a150cd0402fef650052'or spec['scope']not in('SLA_SOURCE_NATIVE_SHARED_V6','SLA_V6_TARGET_NATIVE','SLA_V6_MULTITASK_NATIVE')):
        raise PermissionError('FIXED_V6_PLAN_AND_EXPLICIT_NEW_SCOPE_REQUIRED')
    if Path(__file__).resolve()!=source/'scripts/wm02/v6_bootstrap.py':raise PermissionError('EXACT_NEW_V6_BOOTSTRAP')
    for name,pin in spec['source_files'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=pin:raise ValueError('V6_IMMUTABLE_SOURCE_CHANGED:'+name)
    if hashlib.sha256(Path(parent['spec_path']).read_bytes()).hexdigest()!=parent['spec_sha256']:raise ValueError('TRUSTED_EXECUTION_PARENT_CHANGED')
    sys.path[:0]=[str(source),str(source/'src'),parent['source_root'],parent['source_root']+'/src']
    import go2arx5_vla.world_model,go2arx5_vla.sla_adapter,scripts.wm02,scripts.sla,scripts.pi0294
    for module,rel in((go2arx5_vla.world_model,'src/go2arx5_vla/world_model'),(go2arx5_vla.sla_adapter,'src/go2arx5_vla/sla_adapter'),(scripts.wm02,'scripts/wm02'),(scripts.sla,'scripts/sla'),(scripts.pi0294,'scripts/pi0294')):
        module.__path__=[str(source/rel),*list(module.__path__)]
    from scripts.wm02.mobile_boundary_runner_v1 import registered
    registered(spec,path)
    if operation=='--case':
        from scripts.wm02.mobile_boundary_runner_v1 import run_case
        fd=int(sys.argv[sys.argv.index('--world-lock-fd')+1])
        raise SystemExit(run_case(path,fd))
    if operation=='--target-campaign':
        from scripts.wm02.target_campaign_v6 import run
        raise SystemExit(run(path))
    if operation=='--target-followup':
        from scripts.wm02.target_followup_v6 import main
        raise SystemExit(main(spec))
    if operation=='--multitask-interface':
        from scripts.wm02.multitask_interface_v6 import run
        raise SystemExit(run(path,spec))
    if spec['physics']or spec['new_physical_requests_budget']:raise PermissionError('THIS_V6_ROUTE_IS_NONPHYSICAL')
    if operation=='--worker':
        import runpy
        runpy.run_module('scripts.sla.hybrid_ipc_worker_v3',run_name='__main__')
        return
    if operation=='--target-pair':
        from scripts.wm02.v6_target_pair import run
    elif operation=='--target-train':
        from scripts.sla.train_target_v6 import main as run
    elif operation=='--source-prepare':
        from scripts.sla.source_prepare_v6 import main as run
    elif operation=='--multitask-scene-audit':
        from scripts.wm02.multitask_scene_audit_v6 import main as run
    elif operation=='--multitask-binding-audit':
        from scripts.wm02.multitask_binding_audit_v6 import main as run
    elif operation=='--multitask-f2-regions':
        from scripts.wm02.multitask_f2_regions_v6 import main as run
    elif operation=='--multitask-population':
        from scripts.wm02.multitask_population_v6 import main as run
    elif operation in('--source-test-prepare','--source-test'):
        from scripts.sla.source_test_v6 import main
        run=lambda spec:main(spec,gpu=operation=='--source-test')
    elif operation=='--source-campaign':
        from scripts.wm02.v6_source_campaign import run
    elif operation=='--source-test-campaign':
        from scripts.wm02.source_test_campaign_v6 import run
    elif operation=='--source-train':
        from scripts.sla.train_source_v6 import main as run
    elif operation=='--source-restore':
        from scripts.sla.train_source_v6 import main
        run=lambda spec:main(spec,restore=True)
    elif operation=='--target-offline':
        from scripts.sla.target_offline_v6 import main as run
    elif operation=='--target-diagnose':
        from scripts.sla.target_diagnose_v6 import run
    else:raise ValueError('V6_ROUTE_NOT_IMPLEMENTED_NO_AUTOMATIC_FALLBACK')
    raise SystemExit(run(spec))


if __name__=='__main__':main()
