"""Qualified resumable SSH for exact replacement payloads, then full-file readback."""
import argparse,json,time
from pathlib import Path
import transfer
remote=transfer.old.remote

def copy(point):
    assert point in ['COVER931024','COVER932048','COVER934096']
    started=time.time();source=Path(__file__).with_name('overlay.py').read_text()
    print(remote('hop',f'''
import sys,json
from pathlib import Path
root=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0278/run');code=root.parent/'transport-code';code.mkdir(exist_ok=True)
p=code/'overlay.py';text={source!r}
if p.exists():assert p.read_text()==text
else:p.write_text(text)
sys.path.insert(0,str(code));import overlay
v=overlay.pack(root.parents[1]/'PI0277/R2_FOCUS/FOCUS/main/training/checkpoint-4096',root/'snapshots'/{point!r},root/'overlay/snapshots'/{point!r})
print(json.dumps(dict(point=v['point'],tensors=len(v['tensors']),changed_tensors=v['changed_tensors'],replacement_bytes=v['replacement_bytes'],pack_wall_s=v['pack_wall_s'])))
'''),flush=True)
    transfer.old.HOP='/scratch/e1192548/Projects/go2-arx5/experiments/PI0278/run/overlay'
    transfer.old.RAW='/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0278/R2/overlay'
    transfer.old.copy(point)
    print(remote('core',f'''
import sys,json
from pathlib import Path
sys.path.insert(0,'/home/liuyu-nfs/Projects/go2-arx5/PI0278/code');import overlay
root=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0278/R2');dest=root/'models'/{point!r};source=root/'overlay/models'/{point!r}
if (dest/'READY.json').exists():
 ready=json.loads((dest/'READY.json').read_text());assert ready==json.loads((source/'OVERLAY.json').read_text())['source_ready']
 for row in ready['files']:assert (dest/row['path']).stat().st_size==row['bytes'] and overlay.sha(dest/row['path'])==row['sha256']
 result=dict(status='PASS_REUSED_FULL_RECONSTRUCTED_FILES_READBACK',point={point!r})
else:result=overlay.restore(Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0277/R2/models/FOCUS4096'),source,dest)
result.update(total_wall_s=__import__('time').time()-{started!r},overlay_only_transport=True,optimizer_transferred=False)
p=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0278/R2/evidence')/('TRANSFER_'+{point!r}+'.json');p.write_text(json.dumps(result,sort_keys=True)+chr(10));print(p.read_text())
'''),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('point');copy(p.parse_args().point)
