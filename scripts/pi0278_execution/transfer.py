"""Reuse qualified byte transfer/readback; no local weight payload or new transport."""
import argparse,ast,importlib.util,time
from pathlib import Path
source=Path(__file__).resolve().parents[1]/'pi0277_execution/r2/transfer.py'
spec=importlib.util.spec_from_file_location('qualified_transfer',source);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
old.HOP='/scratch/e1192548/Projects/go2-arx5/experiments/PI0278/run';old.RAW='/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0278/R2'
old.OPTIONS=[*old.OPTIONS,'-C']  # Standard lossless transport; full-file readback is unchanged.
node=next(n for n in ast.parse(source.read_text()).body if getattr(n,'name',None)=='copy');assert isinstance(node.body[0],ast.Assert)
node.body[0]=ast.parse("assert point in [f'COVER93{n}' for n in (512,1024,2048,4096)]").body[0]
exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),'exec'),vars(old))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('point',nargs='?');p.add_argument('--check',action='store_true');a=p.parse_args()
    if a.check:old.check()
    else:
        assert a.point;started=time.time();old.copy(a.point)
        receipt=dict(status='PASS_DIRECT_COMPRESSED_SSH_FULL_FILE_READBACK',point=a.point,started_epoch_s=started,completed_epoch_s=time.time(),wall_s=time.time()-started,SSH_compression=True,scientific_tensor_bytes_changed=False)
        print(old.remote('core',f"import json\nfrom pathlib import Path\nr=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0278/R2/evidence');r.mkdir(exist_ok=True);p=r/{('TRANSFER_'+a.point+'.json')!r};p.write_text(json.dumps({receipt!r},sort_keys=True)+chr(10));print(p.read_text())"),flush=True)
