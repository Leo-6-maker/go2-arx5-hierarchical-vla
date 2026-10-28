"""Retire one consumed immutable export duplicate; never the full Trainer."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi0276_execution'))
from coordinator import remote
POINTS=dict(CTRL1024='CTRL20481024',CTRL2048='CTRL20482048',LR5X1024='LR5X20481024',LR5X2048='LR5X20482048')

def run(point):
    proof=json.loads(remote('core',f"from pathlib import Path\nprint((Path('/home/liuyu-nfs/Projects/go2-arx5/PI0284/models')/{point!r}/'MATERIALIZATION.json').read_text())\n"))
    proof={k:proof[k] for k in ('status','full_master_state_sha256','export_receipt_sha256')}
    return remote('hop',f"""
import json,hashlib
from pathlib import Path
r=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0284');folder=r/'delivery'/{POINTS[point]!r};proof={proof!r}
export=json.loads((folder/'EXPORT.json').read_text());ready=json.loads((folder/'READY.json').read_text());delta=folder/'exact_changed_tensors.safetensors';record=folder/'DUPLICATE_RETIREMENT.json'
assert ready['point']=={POINTS[point]!r} and proof['status']=='PASS_EXACT_TRANSFER_MASTER_READBACK_NATIVE_BF16_VIEW'
assert proof['full_master_state_sha256']==export['full_state_sha256']
assert hashlib.sha256((folder/'EXPORT.json').read_bytes()).hexdigest()==proof['export_receipt_sha256']
cp=Path(export['checkpoint']);arm={'CTRL2048' if point.startswith('CTRL') else 'LR5X2048'!r};step={1024 if point.endswith('1024') else 2048}
assert cp.resolve()==(r/'learning'/arm/'main'/f'checkpoint-{{step}}').resolve()
assert (cp/'optimizer.pt').is_file() and (cp/'PI0282_RECOVERY.json').is_file() and len([p for p in cp.iterdir() if p.is_file()])==14
if not delta.exists():
 v=json.loads(record.read_text());assert v['status']=='DELETED_CONSUMED_EXPORT_DUPLICATE_FULL_TRAINER_RETAINED' and v['sha256']==export['exact_delta_sha256'];print(json.dumps(v))
else:
 assert delta.resolve().is_relative_to((r/'delivery').resolve()) and not delta.is_symlink() and delta.stat().st_nlink==1
 h=hashlib.sha256()
 with delta.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 assert h.hexdigest()==export['exact_delta_sha256'] and delta.stat().st_size==export['exact_delta_bytes']
 v=dict(status='PREVIEW_CONSUMED_IMMUTABLE_EXPORT_DUPLICATE',path=str(delta),bytes=delta.stat().st_size,sha256=h.hexdigest(),protected_full_Trainer=str(cp),Core_master_sha256=proof['full_master_state_sha256'],consumer_closure='READY closes writer; exact transfer/materialization completed; evaluation reads Core view; final audits read EXPORT metadata/full Trainer, never this delta')
 record.write_text(json.dumps(v,indent=2));print(json.dumps(v),flush=True)
 delta.unlink();v['status']='DELETED_CONSUMED_EXPORT_DUPLICATE_FULL_TRAINER_RETAINED';record.write_text(json.dumps(v,indent=2));print(json.dumps(v))
""")

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('point',choices=tuple(POINTS));a=p.parse_args();print(run(a.point))
