"""Owner-invoked exact CPU correction relay; no runtime code edit or qsub.

Runs only after actual Core correction bank and fixed sampler PASS. Streams
only named fresh array members and receipts through existing PC SSH identities.
Writes a prospective C1 card; owner reserve/admission promotion remains explicit.
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
b=json.loads((r/'correction/CORRECTION_BANK.json').read_text());a=json.loads((r/'correction/C1_PLAN_AUDIT.json').read_text())
assert b['status']=='PASS_ACTUAL_BRANCH_RESTORE_FRESH_CORRECTIONS_FIXED_CODEC' and b['requested_branches']==32 and b['valid_branches']>=24
assert not b['missing'] and not b['duplicates'] and not b['duplicate_states']
assert a['status']=='PREPARED_C1_EQUAL_EXPERT_CORRECTION_PLAN_NOT_TRAINING_ADMISSION' and a['rows']==8192 and a['source_counts']=={'expert':4096,'corrected_suffix':4096}
members={}
for row in b['rows']:
 for name,h in row['source_identity'].items():
  assert name.split('/')[0] in ('policy_observation','action_target') and row['episode'] in b['valid_slots']
  rel='data/'+name;p=r/rel;assert p.resolve().is_relative_to(r.resolve()) and p.is_file()
  assert rel not in members or members[rel]['sha256']==h
  members[rel]=dict(bytes=p.stat().st_size,sha256=h)
for rel in ('correction/CORRECTION_BANK.json','correction/C1_PLAN_AUDIT.json','data/C1_CORRECT1024.jsonl'):
 p=r/rel;members[rel]=dict(bytes=p.stat().st_size,sha256=sha(p))
assert members['data/C1_CORRECT1024.jsonl']['sha256']==a['plan_sha256'] and members['correction/CORRECTION_BANK.json']['sha256']==a['correction_bank_sha256']
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
b=json.loads((r/'correction/CORRECTION_BANK.json').read_text());a=json.loads((r/'correction/C1_PLAN_AUDIT.json').read_text())
assert b['status']=='PASS_ACTUAL_BRANCH_RESTORE_FRESH_CORRECTIONS_FIXED_CODEC' and b['requested_branches']==32 and b['valid_branches']>=24
assert a['source_counts']=={'expert':4096,'corrected_suffix':4096} and a['rows']==8192
rows=[json.loads(x) for x in (r/'data/C1_CORRECT1024.jsonl').read_text().splitlines()]
assert len(rows)==8192 and [v['index'] for v in rows]==list(range(8192))
for v in rows:
 if v['source']=='corrected_suffix':
  assert v['episode'] in b['valid_slots']
  for name,h in v['source_identity'].items():assert MANIFEST['members']['data/'+name]['sha256']==h
card=json.loads((r/'design/C0_REPLAY1024_CARD.json').read_text());card.update(stage='C1_CORRECT1024',plan_path=str(r/'data/C1_CORRECT1024.jsonl'),plan_sha256=sha(r/'data/C1_CORRECT1024.jsonl'),correction_admission=str(r/'correction/CORRECTION_BANK.json'),correction_admission_sha256=sha(r/'correction/CORRECTION_BANK.json'),launch_allowed=False)
p=r/'design/C1_CORRECT1024_CARD.prospective.json'
if p.exists():assert json.loads(p.read_text())==card
else:write(p,card)
# Validate the same admission checks without publishing an active launch card.
probe=r/'design/C1_CARD_CPU_VALIDATION.json';write(probe,dict(card,launch_allowed=True))
try:
 from authority import stage_card
 stage_card(r,probe)
finally:probe.unlink()
value=dict(status='PASS_ACTUAL_C1_EXACT_TRANSFER_READBACK_PROSPECTIVE_CARD',manifest=MANIFEST,card_sha256=sha(p),valid_branches=b['valid_branches'],optimizer_updates=0,model_queries=0,PBS_submissions=0,launch_allowed=False)
write(r/'evidence/C1_TRANSFER_READBACK.json',value);print(json.dumps(value))
'''
    prefix='CORE='+repr(CORE)+'\nHOP='+repr(HOP)+'\nMANIFEST='+repr(manifest)+'\n'
    src_remote=shlex.join(['python3','-c',prefix+source])
    src_cmd=['ssh',*OPTIONS,'corelab-amd-server',shlex.join(['ssh',*OPTIONS,'corelab-4080s',src_remote])]
    dst_cmd=['ssh',*OPTIONS,'nus-hopper',shlex.join(['python3','-c',prefix+target])]
    with subprocess.Popen(src_cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE) as src:
        with subprocess.Popen(dst_cmd,stdin=src.stdout,stdout=subprocess.PIPE,stderr=subprocess.PIPE) as dst:
            src.stdout.close();out,error=dst.communicate();src_error=src.stderr.read();code=src.wait()
            assert dst.returncode==code==0,(code,dst.returncode,src_error[-3000:].decode(),error[-3000:].decode())
    value=json.loads(out);p=Path('reports/PI0286/preparation/C1_TRANSFER_READBACK.json');p.write_text(json.dumps(value,indent=2)+'\n');return value


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(transfer()))
