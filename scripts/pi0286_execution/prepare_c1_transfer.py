"""Owner-invoked exact CPU correction relay; no runtime code edit or qsub.

Runs only after actual Core correction bank and fixed sampler PASS. Streams
only named fresh array members and receipts through existing PC SSH identities.
Writes prospective and CPU-validated C1 cards; owner controls capacity and launch.
"""
import argparse,json,shlex,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi0276_execution'))
from coordinator import remote

CORE='/home/liuyu-nfs/Projects/go2-arx5/PI0286'
HOP='/scratch/e1192548/Projects/go2-arx5/experiments/PI0286'
OPTIONS=['-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=6']


def transfer():
    metadata=r'''
import hashlib,json
from pathlib import Path
r=Path(CORE)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
b=json.loads((r/'correction/BANK_NATIVE_CONTRACT_V2.json').read_text());a=json.loads((r/'correction/SAMPLER_BUCKET_CONTRACT_V3/CPU_PLAN_DRAFT.json').read_text())
assert b['status']=='PASS_ACTUAL_BRANCH_RESTORE_FRESH_CORRECTIONS_FIXED_CODEC' and b['requested_branches']==32 and b['valid_branches']>=24
assert not b['missing'] and not b['duplicates'] and not b['duplicate_states']
assert a['status']=='PASS_ACTUAL_BANK_ORIGINAL_BUCKET_CONDITIONAL_EQUALITY_CPU_PLAN_DRAFT' and a['rows']==8192 and a['source_counts']=={'expert':4096,'corrected_suffix':4096}
assert b['valid_branches']==27 and a['data_bank_sha256']=='f2c7b43af6edc42795ee0595ed0a9c9f05f440369945d5b43bfc38419e21cb8d'
assert a['plan_sha256']=='0194583c35e93000771aed598817091a6d670b9f1720d31da6a06d6b37934f0e'
readback=json.loads((r/'correction/CORRECTION_TRANSFER_MANIFEST_NATIVE_V2_BUCKET_V3.json').read_text())
assert readback['status']=='PASS_ACTUAL_CORRECTION_NUMPY_FULL_MEMBER_READBACK_NOT_TRAINING_ADMISSION'
members={}
for row in b['rows']:
 for name,h in row['source_identity'].items():
  assert name.split('/')[0] in ('policy_observation','action_target') and row['episode'] in b['valid_slots']
  rel='data/'+name;p=r/rel;assert p.resolve().is_relative_to(r.resolve()) and p.is_file()
  assert rel not in members or members[rel]['sha256']==h
  members[rel]=dict(bytes=p.stat().st_size,sha256=h)
for rel in ('correction/BANK_NATIVE_CONTRACT_V2.json','correction/SAMPLER_BUCKET_CONTRACT_V3/CPU_PLAN_DRAFT.json','correction/SAMPLER_BUCKET_CONTRACT_V3/C1_CORRECT1024_BUCKET_CONTRACT_V3.jsonl','correction/CORRECTION_TRANSFER_MANIFEST_NATIVE_V2_BUCKET_V3.json','correction/C1_DATA_ADMISSION_NATIVE_V2_BUCKET_V3.json'):
 p=r/rel;members[rel]=dict(bytes=p.stat().st_size,sha256=sha(p))
assert members['correction/SAMPLER_BUCKET_CONTRACT_V3/C1_CORRECT1024_BUCKET_CONTRACT_V3.jsonl']['sha256']==a['plan_sha256'] and members['correction/BANK_NATIVE_CONTRACT_V2.json']['sha256']==a['data_bank_sha256']
assert len([k for k in members if k.endswith('.npy')])==162
print(json.dumps(dict(status='ACTUAL_C1_TRANSFER_FIXED_FILELIST',members=members,valid_branches=b['valid_branches'])))
'''
    manifest=json.loads(remote('core','CORE='+repr(CORE)+'\n'+metadata))
    source=r'''
import sys,tarfile,hashlib
from pathlib import Path
r=Path(CORE)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as t:
 for rel,item in sorted(MANIFEST['members'].items()):
  p=r/rel;assert p.resolve().is_relative_to(r.resolve()) and p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],rel
  t.add(p,arcname=rel,recursive=False)
'''
    target=r'''
import sys,os,json,hashlib,tarfile
from pathlib import Path
r=Path(HOP);os.environ['PI0286_ROOT']=str(r);sys.path[:0]=[str(r/'code'),str(r.parent/'PI0283/code')]
from contract import guard,sha,write
guard();seen=set()
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as t:
 for entry in t:
  rel=entry.name;assert rel in MANIFEST['members'] and rel not in seen and entry.isfile(),rel
  item=MANIFEST['members'][rel];assert entry.size==item['bytes'];p=r/rel
  assert p.resolve().is_relative_to(r.resolve());p.parent.mkdir(parents=True,exist_ok=True)
  partial=p.with_name(p.name+'.relay.partial');assert not partial.is_symlink()
  h=hashlib.sha256();count=0
  with t.extractfile(entry) as src,partial.open('wb') as dst:
   for chunk in iter(lambda:src.read(8<<20),b''):dst.write(chunk);h.update(chunk);count+=len(chunk)
   dst.flush();os.fsync(dst.fileno())
  assert count==item['bytes'] and h.hexdigest()==item['sha256'],rel
  if p.exists():assert p.stat().st_size==count and sha(p)==item['sha256'];partial.unlink()
  else:partial.replace(p)
  seen.add(rel)
assert seen==set(MANIFEST['members'])
b=json.loads((r/'correction/BANK_NATIVE_CONTRACT_V2.json').read_text());a=json.loads((r/'correction/SAMPLER_BUCKET_CONTRACT_V3/CPU_PLAN_DRAFT.json').read_text())
assert b['status']=='PASS_ACTUAL_BRANCH_RESTORE_FRESH_CORRECTIONS_FIXED_CODEC' and b['requested_branches']==32 and b['valid_branches']>=24
assert a['source_counts']=={'expert':4096,'corrected_suffix':4096} and a['rows']==8192
rows=[json.loads(x) for x in (r/'correction/SAMPLER_BUCKET_CONTRACT_V3/C1_CORRECT1024_BUCKET_CONTRACT_V3.jsonl').read_text().splitlines()]
assert len(rows)==8192 and [v['index'] for v in rows]==list(range(8192))
for v in rows:
 if v['source']=='corrected_suffix':
  assert v['episode'] in b['valid_slots']
  for name,h in v['source_identity'].items():assert MANIFEST['members']['data/'+name]['sha256']==h
c0=r/'design/C0_REPLAY1024_CARD.json'
if not c0.exists():c0=r/'design/C0_REPLAY1024_CARD.prospective.json'
card=json.loads(c0.read_text());card.update(stage='C1_CORRECT1024',plan_path=str(r/'correction/SAMPLER_BUCKET_CONTRACT_V3/C1_CORRECT1024_BUCKET_CONTRACT_V3.jsonl'),plan_sha256=a['plan_sha256'],correction_admission=str(r/'correction/BANK_NATIVE_CONTRACT_V2.json'),correction_admission_sha256=a['data_bank_sha256'],launch_allowed=False,canary='NEW_REAL_CONTINUOUS8_VS_FRESH4_RESTORE4_REQUIRED')
p=r/'design/C1_CORRECT1024_CARD_NATIVE_V2_BUCKET_V3.prospective.json'
if p.exists():assert json.loads(p.read_text())==card
else:write(p,card)
# Validate the original admission checks before publishing the C1-only card.
probe=r/'design/C1_CARD_CPU_VALIDATION_NATIVE_V2_BUCKET_V3.json';write(probe,dict(card,launch_allowed=True))
try:
 from authority import stage_card
 stage_card(r,probe)
finally:probe.unlink()
active=r/'design/C1_CORRECT1024_CARD.json';active_card=dict(card,launch_allowed=True)
if active.exists():assert json.loads(active.read_text())==active_card
else:write(active,active_card)
value=dict(status='PASS_ACTUAL_C1_NATIVE_V2_BUCKET_V3_EXACT_TRANSFER_READBACK_AND_CARD',manifest=MANIFEST,card_sha256=sha(active),card_path=str(active),prospective_card_sha256=sha(p),valid_branches=b['valid_branches'],optimizer_updates=0,model_queries=0,PBS_submissions=0,launch_allowed=True,actual_canary='UNOBSERVED_FRESH_C1_REQUIRED')
write(r/'evidence/C1_TRANSFER_READBACK_NATIVE_V2_BUCKET_V3.json',value);print(json.dumps(value))
'''
    prefix='CORE='+repr(CORE)+'\nHOP='+repr(HOP)+'\nMANIFEST='+repr(manifest)+'\n'
    def stage_relay(host,root,name,body):
        path=root+'/correction/transfer_tools/'+name
        script='from pathlib import Path\np=Path('+repr(path)+')\np.parent.mkdir(parents=True,exist_ok=True)\nb='+repr(body.encode())+'\nif p.exists():assert p.read_bytes()==b\nelse:p.write_bytes(b)\n'
        remote(host,script);return path
    source_path=stage_relay('core',CORE,'SOURCE_NATIVE_V2_BUCKET_V3.py',prefix+source)
    target_path=stage_relay('hop',HOP,'TARGET_NATIVE_V2_BUCKET_V3.py',prefix+target)
    src_remote=shlex.join(['python3',source_path])
    src_cmd=['ssh',*OPTIONS,'corelab-amd-server',shlex.join(['ssh',*OPTIONS,'corelab-4080s',src_remote])]
    dst_cmd=['ssh',*OPTIONS,'nus-hopper',shlex.join(['python3',target_path])]
    with subprocess.Popen(src_cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE) as src:
        with subprocess.Popen(dst_cmd,stdin=src.stdout,stdout=subprocess.PIPE,stderr=subprocess.PIPE) as dst:
            src.stdout.close();out,error=dst.communicate();src_error=src.stderr.read();code=src.wait()
            assert dst.returncode==code==0,(code,dst.returncode,src_error[-3000:].decode(),error[-3000:].decode())
    value=json.loads(out);p=Path('reports/PI0286/preparation/C1_TRANSFER_READBACK_NATIVE_V2_BUCKET_V3.json');p.write_text(json.dumps(value,indent=2)+'\n');return value


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    value=transfer();print(json.dumps({k:v for k,v in value.items() if k!='manifest'}))
