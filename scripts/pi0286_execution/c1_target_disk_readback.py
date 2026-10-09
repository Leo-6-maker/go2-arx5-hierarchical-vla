"""C1-only full disk byte/header readback after exact relay; no model imports."""
import ast,hashlib,json,os,struct,time
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
ROOT=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0286')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def stamp(p):
    s=p.stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
def header(p):
    with p.open('rb') as f:
        assert f.read(6)==b'\x93NUMPY';version=tuple(f.read(2));size=struct.unpack('<H' if version==(1,0) else '<I',f.read(2 if version==(1,0) else 4))[0]
        return ast.literal_eval(f.read(size).decode('latin1' if version<(3,0) else 'utf-8'))
start=time.time();receipt=ROOT/'evidence/C1_TRANSFER_READBACK_NATIVE_V2_BUCKET_V3.json'
v=json.loads(receipt.read_text());assert v['valid_branches']==27 and v['PBS_submissions']==0
manifest=json.loads((ROOT/'correction/CORRECTION_TRANSFER_MANIFEST_NATIVE_V2_BUCKET_V3.json').read_text())
by={x['path']:x for x in manifest['files']};assert len(by)==162
checks=[]
type_names={'u':'uint','i':'int','f':'float','b':'bool'}
for relative,expected in sorted(v['manifest']['members'].items()):
    p=ROOT/relative;assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(ROOT.resolve())
    before=stamp(p);assert before[2]==expected['bytes'] and sha(p)==expected['sha256']
    item=dict(path=relative,sha256=expected['sha256'],bytes=expected['bytes'],local_stamp=before)
    if p.suffix=='.npy':
        d=header(p);e=by[str(p.relative_to(ROOT/'data'))]
        descr=d['descr'];kind=descr[1];width=int(descr[2:])*8
        dtype='bool' if kind=='b' else type_names[kind]+str(width)
        assert list(d['shape'])==e['shape'] and dtype==e['dtype']
        item.update(shape=list(d['shape']),dtype=dtype,fortran_order=d['fortran_order'])
    assert stamp(p)==before;checks.append(item)
card=Path(v['card_path']);assert sha(card)==v['card_sha256'];c=json.loads(card.read_text())
assert c['stage']=='C1_CORRECT1024' and c['launch_allowed'] and c['canary']=='NEW_REAL_CONTINUOUS8_VS_FRESH4_RESTORE4_REQUIRED'
assert c['correction_admission_sha256']=='f2c7b43af6edc42795ee0595ed0a9c9f05f440369945d5b43bfc38419e21cb8d'
assert c['plan_sha256']=='0194583c35e93000771aed598817091a6d670b9f1720d31da6a06d6b37934f0e'
result=dict(status='PASS_ACTUAL_HOPPER_C1_NATIVE_V2_BUCKET_V3_FULL_DISK_BYTES_HEADERS_AND_CARD',files=checks,
    admitted_branches=27,new_NPY_files=162,new_NPY_bytes=manifest['total_bytes'],array_tree_sha256=manifest['tree_sha256'],
    transfer_receipt_sha256=sha(receipt),card_path=str(card),card_sha256=sha(card),bank_sha256=c['correction_admission_sha256'],plan_sha256=c['plan_sha256'],
    actual_canary='UNOBSERVED_FRESH_C1_REQUIRED',model_calls=0,physics_calls=0,codec_calls=0,optimizer_updates=0,PBS_submissions=0,elapsed_seconds=time.time()-start)
out=ROOT/'evidence/C1_TARGET_DISK_READBACK_NATIVE_V2_BUCKET_V3.json'
assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:x for k,x in result.items() if k!='files'}))
