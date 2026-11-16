"""Only offline roles; immutable source routing and existing lease admission."""
import hashlib
import json
from pathlib import Path
import runpy
import sys


def main():
    operation,path=sys.argv[1:3]
    routes={'--g1':'scripts.sla.close_goal_g1_v1','--safety':'scripts.wm02.close_goal_safety_v1'}
    if operation not in routes:raise PermissionError('OFFLINE_ONLY_NO_NATIVE_ROLE')
    spec=json.loads(Path(path).read_bytes());root=Path(spec['source_root']);parent=spec['trusted_source_parent']
    if spec['physics'] or spec['new_physical_requests_budget']!=0:raise PermissionError('PHYSICAL_QUARANTINE')
    for name,pin in spec['source_files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=pin:raise ValueError('SEALED_SOURCE_CHANGED:'+name)
    if hashlib.sha256(Path(parent['spec_path']).read_bytes()).hexdigest()!=parent['spec_sha256']:raise ValueError('TRUSTED_RUNTIME_CHANGED')
    sys.path[:0]=[str(root),str(root/'src'),parent['source_root'],parent['source_root']+'/src']
    import go2arx5_vla.world_model,go2arx5_vla.sla_adapter,scripts.sla,scripts.wm02
    for module,relative in ((go2arx5_vla.world_model,'src/go2arx5_vla/world_model'),(go2arx5_vla.sla_adapter,'src/go2arx5_vla/sla_adapter'),(scripts.sla,'scripts/sla'),(scripts.wm02,'scripts/wm02')):
        module.__path__=[str(root/relative),*list(module.__path__)]
    from scripts.wm02.mobile_boundary_runner_v1 import registered
    registered(spec,path)
    runpy.run_module(routes[operation],run_name='__main__')


if __name__=='__main__':main()
