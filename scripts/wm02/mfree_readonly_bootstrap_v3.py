"""Verify the immutable 72h source before entering a registered finite role."""
import hashlib
import json
from pathlib import Path
import runpy
import sys


def main():
    operation=sys.argv[1]
    if operation!='--audit':raise PermissionError('READ_ONLY_EXISTING_OUTPUTS_ONLY')
    path=Path(sys.argv[2]);spec=json.loads(path.read_bytes());root=Path(spec['source_root']);parent=spec['trusted_source_parent']
    if Path(__file__).resolve()!=root/'scripts/wm02/mfree_readonly_bootstrap_v3.py':raise PermissionError('EXACT_72H_BOOTSTRAP')
    if hashlib.sha256(Path(parent['spec_path']).read_bytes()).hexdigest()!=parent['spec_sha256']:raise PermissionError('IMMUTABLE_REVIEWED_RUNTIME_PARENT')
    for name,pin in spec['source_files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=pin:raise PermissionError('SEALED72_SOURCE_CHANGED:'+name)
    sys.path[:0]=[str(root),str(root/'src'),parent['source_root'],parent['source_root']+'/src']
    import go2arx5_vla.world_model,go2arx5_vla.sla_adapter,scripts.wm02,scripts.sla
    for module,relative in ((go2arx5_vla.world_model,'src/go2arx5_vla/world_model'),(go2arx5_vla.sla_adapter,'src/go2arx5_vla/sla_adapter'),(scripts.wm02,'scripts/wm02'),(scripts.sla,'scripts/sla')):
        module.__path__=[str(root/relative),*list(module.__path__)]
    routes={'--supervisor':'scripts.wm02.mfree_recovery_supervisor_v2','--collector':'scripts.wm02.mfree_recovery_collector_v2',
        '--case':'scripts.wm02.mfree_unattended_case_v1','--batch':'scripts.wm02.mfree_unattended_batch_v1','--neural':'scripts.wm02.mfree_unattended_neural_v1',
        '--server':'scripts.wm02.mfree_snapshot_resident_v1','--train':'scripts.sla.train_mfree_snapshot_v1','--replay':'scripts.sla.train_mfree_snapshot_v1',
        '--audit':'scripts.wm02.mfree_c20_query_audit_v3','--diagnostic':'scripts.sla.mfree_c20_diagnostics_v2','--export':'scripts.sla.export_mfree_snapshot_v1'}
    if operation in ('--train','--replay'):sys.argv=[sys.argv[0],str(path)]+(['--replay']if operation=='--replay'else[])
    elif operation=='--export':sys.argv=[sys.argv[0],spec['snapshot_descriptor']['path'],spec['dataset']]
    runpy.run_module(routes[operation],run_name='__main__')


if __name__=='__main__':main()
