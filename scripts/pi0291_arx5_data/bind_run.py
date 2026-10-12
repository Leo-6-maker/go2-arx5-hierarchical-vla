"""Register this immutable code snapshot and private host-aware run paths."""
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import time

from go2arx5_vla.task_data.io import file_hash, write
from .config import AUTHORITY, PARAMETERS, COMMISSIONING, PILOTS, TRAIN, DEV, layout, canonical_hash


def bind(root, source):
    root = Path(root)
    code = root / 'code'
    before = resource.getrusage(resource.RUSAGE_SELF)
    files = {p.relative_to(code).as_posix(): file_hash(p)
             for tree in ('scripts/pi0291_arx5_data', 'src/go2arx5_vla')
             for p in (code/tree).rglob('*.py')}
    seal_path=root/'CODE_SEAL.json'
    if seal_path.exists():
        prior=json.loads(seal_path.read_text())
        write(root/'evidence'/('CODE_SEAL_'+prior['source_sha']+'.json'),prior)
    write(seal_path, dict(source_sha=source, files=files))
    private = json.loads((root/'PRIVATE_ASSET_BINDING.json').read_text())
    native = Path(private['native_asset_root']) / 'go2_x5.xml'
    assert file_hash(native) == 'b306e99adfb4f5e8af73649df2818ac14472395dadd1ac0b1db1c8473bbe8501'
    usage = shutil.disk_usage(root)
    assert usage.free >= 50*1024**3
    probe = root / 'allocation-proof.tmp'
    with probe.open('xb') as handle:
        handle.write(os.urandom(1024*1024))
        handle.flush()
        os.fsync(handle.fileno())
    st = probe.stat()
    assert st.st_blocks*512 >= 1024*1024
    probe.unlink()
    index = dict(experiment_id='PI0291', run_id=root.name, executor_node='corelab-4080s',
                 work_root=str(root.resolve()), output_root=str(root/'episodes'),
                 private_source=str(native), runtime=private['runtime_python'],
                 source_sha=source, authority=AUTHORITY, personal_quota='UNKNOWN',
                 allocated_write_proof_bytes=st.st_blocks*512, df_free=usage.free,
                 fixed_asset_preserved=True, parameter_sha256=canonical_hash(PARAMETERS))
    write(root/'PRIVATE_PATH_INDEX.json', index)
    write(root/'REGISTERED_LAYOUTS.json', {i:layout(i) for i in COMMISSIONING+PILOTS+TRAIN+DEV})
    write(root/'evidence'/('N0_BINDING_'+source+'.json'), dict(source_sha=source, native_sha256=file_hash(native),
          free_bytes=usage.free, quota='UNKNOWN', actual_write_proof_bytes=st.st_blocks*512,
          proof_removed=True, runtime_reused=True, new_physics=0, setup_steps_actual=0,
          registered_engineering_steps_max=8000, paired_static_diagnostic_count=1))
    cpu = resource.getrusage(resource.RUSAGE_SELF)
    print(json.dumps(dict(status='BOUND',source_sha=source,CPU_s=cpu.ru_utime+cpu.ru_stime-before.ru_utime-before.ru_stime)))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source-sha',required=True)
    a=p.parse_args();bind(a.root,a.source_sha)
