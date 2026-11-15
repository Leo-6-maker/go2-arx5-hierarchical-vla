"""Load committed mobile members over the sealed, read-only execution parent."""
import hashlib
import json
from pathlib import Path
import runpy
import sys


def main():
    flag='--case'if '--case'in sys.argv else '--batch'
    spec=json.loads(Path(sys.argv[sys.argv.index(flag)+1]).read_bytes())
    root=Path(spec['source_root']);parent=spec['trusted_source_parent']
    if Path(__file__).resolve()!=root/'scripts/wm02/mobile_boundary_bootstrap_v1.py':
        raise PermissionError('EXACT_COMMITTED_MOBILE_BOOTSTRAP')
    if hashlib.sha256(Path(parent['spec_path']).read_bytes()).hexdigest()!=parent['spec_sha256']:
        raise PermissionError('SEALED_EXECUTION_PARENT_CHANGED')
    for name,pin in spec['source_files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=pin:
            raise PermissionError('COMMITTED_MOBILE_MEMBER_CHANGED:'+name)
    sys.path[:0]=[str(root),str(root/'src'),parent['source_root'],parent['source_root']+'/src']
    import go2arx5_vla.world_model
    import scripts.wm02
    go2arx5_vla.world_model.__path__.insert(0,str(root/'src/go2arx5_vla/world_model'))
    scripts.wm02.__path__.insert(0,str(root/'scripts/wm02'))
    runpy.run_module('scripts.wm02.mobile_boundary_runner_v1',run_name='__main__')


if __name__=='__main__':main()
