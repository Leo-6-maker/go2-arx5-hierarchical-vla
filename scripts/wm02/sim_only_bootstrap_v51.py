"""Immutable source routing for a project-authorized isolated MuJoCo scope."""
import hashlib
import json
from pathlib import Path
import runpy
import sys


def main():
    operation,path=sys.argv[1:3]
    if operation not in ('--campaign','--case','--worker'):
        raise PermissionError('FINITE_SIM_ONLY_V51_ROUTE')
    spec=json.loads(Path(path).read_bytes());root=Path(spec['source_root'])
    parent=spec['trusted_source_parent']
    if Path(__file__).resolve()!=root/'scripts/wm02/sim_only_bootstrap_v51.py':
        raise PermissionError('EXACT_COMMITTED_SIMULATION_BOOTSTRAP')
    for name,sha in spec['source_files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=sha:
            raise ValueError('SEALED_SIMULATION_SOURCE_CHANGED:'+name)
    if hashlib.sha256(Path(parent['spec_path']).read_bytes()).hexdigest()!=parent['spec_sha256']:
        raise ValueError('TRUSTED_EXECUTION_PARENT_CHANGED')
    sys.path[:0]=[str(root),str(root/'src'),parent['source_root'],parent['source_root']+'/src']
    import go2arx5_vla.world_model,go2arx5_vla.sla_adapter,scripts.wm02,scripts.sla
    for module,rel in ((go2arx5_vla.world_model,'src/go2arx5_vla/world_model'),
            (go2arx5_vla.sla_adapter,'src/go2arx5_vla/sla_adapter'),
            (scripts.wm02,'scripts/wm02'),(scripts.sla,'scripts/sla')):
        module.__path__=[str(root/rel),*list(module.__path__)]
    if operation=='--campaign':
        from scripts.wm02.sim_only_campaign_v51 import run
        raise SystemExit(run(path))
    if operation=='--case':
        from scripts.wm02.mobile_boundary_runner_v1 import run_case
        fd=int(sys.argv[sys.argv.index('--world-lock-fd')+1])
        raise SystemExit(run_case(path,fd))
    if spec['physics'] or spec['new_physical_requests_budget']!=0:
        raise PermissionError('NEURAL_WORKER_CANNOT_DISPATCH_NATIVE')
    from scripts.wm02.mobile_boundary_runner_v1 import registered
    registered(spec,path)
    runpy.run_module('scripts.sla.hybrid_ipc_worker_v3',run_name='__main__')


if __name__=='__main__':main()
