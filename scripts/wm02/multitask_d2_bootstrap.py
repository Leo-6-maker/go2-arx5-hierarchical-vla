"""Load the sealed PR 411 runtime overlay over the previously reviewed stack."""

import hashlib
import json
from pathlib import Path
import sys


def main():
    operation, path, *args = sys.argv[1:]
    spec = json.loads(Path(path).read_bytes())
    source = Path(spec['source_root'])
    parent = spec['trusted_source_parent']
    baseline = Path(parent['source_root'])
    scopes = {'SLA_D2_MFREE_NATIVE_MUJOCO': '2352547871684c7617c2c89754fbf9c70a534abb',
              'SLA_D3_MFREE_NATIVE_MUJOCO': '323551a76e4462436d4ed13f1cbb205156d7ce99'}
    if (spec['plan_commit'] != scopes.get(spec['scope'])
            or Path(__file__).resolve() != source/'scripts/wm02/multitask_d2_bootstrap.py'):
        raise PermissionError('D2_EXACT_SEALED_BOOTSTRAP')
    for name, sha in spec['source_files'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != sha:
            raise PermissionError('D2_SOURCE_MEMBER_CHANGED:'+name)
    if hashlib.sha256(Path(parent['spec_path']).read_bytes()).hexdigest() != parent['spec_sha256']:
        raise PermissionError('D2_TRUSTED_RUNTIME_PARENT_CHANGED')
    sys.path[:0] = [str(source),str(source/'src'),str(baseline),str(baseline/'src')]
    import go2arx5_vla.world_model, go2arx5_vla.sla_adapter, scripts.wm02, scripts.sla, scripts.pi0294

    for module, rel in ((go2arx5_vla.world_model,'src/go2arx5_vla/world_model'),
                        (go2arx5_vla.sla_adapter,'src/go2arx5_vla/sla_adapter'),
                        (scripts.wm02,'scripts/wm02'), (scripts.sla,'scripts/sla'),
                        (scripts.pi0294,'scripts/pi0294')):
        module.__path__ = [str(source/rel), *module.__path__]
    if operation=='--case':
        from scripts.wm02.mobile_boundary_runner_v1 import run_case

        fd, = args
        return run_case(path, int(fd))
    if operation=='--campaign':
        from scripts.wm02.multitask_d2_campaign import run

        return run(path, spec)
    if operation=='--audit-g7':
        import os
        import time
        from scripts.wm02.multitask_d2_campaign import verify_g7_capture
        from go2arx5_vla.world_model.multitask_scope_v6 import read_pin

        sha, = args
        root = Path(spec['registered_root'])
        lease = json.loads((root/'control/LEASE.json').read_bytes())
        if (os.environ.get('CUDA_VISIBLE_DEVICES') != ''
                or os.getuid() != spec['uid'] or os.uname().nodename != spec['host']
                or spec['d2_stage'] not in ('D','R1','R2','Q') or time.time() >= spec['deadline_epoch']
                or lease['status'] != 'CLOSED' or lease['spec_sha256'] != hashlib.sha256(Path(path).read_bytes()).hexdigest()):
            raise PermissionError('D2_G7_REQUIRES_EXACT_COLD_NATIVE_CASE_AND_CPU_ENVIRONMENT')
        native = read_pin(dict(path=str(root/'RESULT.private.json'),sha256=sha))
        print(json.dumps(verify_g7_capture(spec,native),allow_nan=False))
        return 0
    raise ValueError('D2_DECLARED_EXECUTION_ROUTE_REQUIRED')


if __name__ == '__main__':
    raise SystemExit(main())
