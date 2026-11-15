"""Sealed new neural/export modules over the reviewed complete execution stack."""
import hashlib
import json
from pathlib import Path
import runpy
import sys


def main():
    operation=sys.argv[1];path=Path(sys.argv[2]);spec=json.loads(path.read_bytes())
    root=Path(spec['source_root']);parent=spec['trusted_source_parent']
    if Path(__file__).resolve()!=root/'scripts/wm02/mfree_neural_bootstrap_v1.py':raise PermissionError('EXACT_NEW_MFREE_BOOTSTRAP')
    if hashlib.sha256(Path(parent['spec_path']).read_bytes()).hexdigest()!=parent['spec_sha256']:raise PermissionError('TRUSTED_EXECUTION_PARENT_CHANGED')
    for name,pin in spec['source_files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=pin:raise PermissionError('SEALED_MFREE_SOURCE_CHANGED:'+name)
    sys.path[:0]=[str(root),str(root/'src'),parent['source_root'],parent['source_root']+'/src']
    import go2arx5_vla.world_model,go2arx5_vla.sla_adapter,scripts.wm02,scripts.sla
    for package,relative in ((go2arx5_vla.world_model,'src/go2arx5_vla/world_model'),
            (go2arx5_vla.sla_adapter,'src/go2arx5_vla/sla_adapter'),(scripts.wm02,'scripts/wm02'),(scripts.sla,'scripts/sla')):
        package.__path__.insert(0,str(root/relative))
    if operation in ('--train','--replay'):
        sys.argv=[sys.argv[0],str(path)]+(['--replay']if operation=='--replay'else[])
        module='scripts.sla.train_mfree_c20_v1'
    elif operation=='--export':
        sys.argv=[sys.argv[0],spec['collection_root'],spec['dataset']];module='scripts.sla.export_mfree_c20_v1'
    elif operation=='--server':module='scripts.wm02.mobile_sla_resident_v1'
    else:raise ValueError('FINITE_EXPLICIT_MFREE_OPERATION')
    runpy.run_module(module,run_name='__main__')


if __name__=='__main__':main()
