from pathlib import Path
from urllib.parse import quote
import hashlib,json,os,time
import requests
root=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0253/storage_readback')
source=root/'FULL_FAMILY_STREAM_INPUTS.json'
inputs=json.loads(source.read_text());assert inputs['schema']=='PI0253_STORAGE_EXACT_STREAM_INPUTS_V1'
assert [a['asset'] for a in inputs['assets']]==['G2000','FULL2500','FULL5000','FULL7500','FULL10000']
def save(path,value):
 text=json.dumps(value,indent=2,sort_keys=True)+'\n'
 if path.exists():assert path.read_text()==text
 else:
  with path.open('x') as f:f.write(text);f.flush();os.fsync(f.fileno())
for asset in inputs['assets']:
 assert asset['repo']=='leoyuer/pi0246-gate-pass-campaign' and len(asset['revision'])==40
 directory=root/asset['asset'];directory.mkdir(exist_ok=True)
 save(directory/'INPUT.json',asset)
 with requests.get('https://huggingface.co/api/models/'+asset['repo']+'/revision/'+asset['revision'],timeout=(30,180)) as response:
  response.raise_for_status();assert response.json()['private'] is False
 assert len({r['path'] for r in asset['files']})==len(asset['files'])
 rows=[]
 for index,row in enumerate(asset['files']):
  assert not row['path'].startswith('/') and '..' not in Path(row['path']).parts
  dest=directory/f'file-{index:02d}.json'
  if dest.exists():
   old=json.loads(dest.read_text());assert old['path']==row['path'] and old['bytes']==row['bytes'] and old['sha256']==row['sha256'];rows.append(old);continue
  url='https://huggingface.co/'+asset['repo']+'/resolve/'+asset['revision']+'/'+quote(row['path'],safe='/')+'?download=true'
  h=hashlib.sha256();size=0;started=time.time()
  print(json.dumps(dict(asset=asset['asset'],path=row['path'],status='ANONYMOUS_FULL_STREAM_STARTED')),flush=True)
  with requests.get(url,stream=True,timeout=(30,180)) as response:
   response.raise_for_status();assert response.status_code==200
   for block in response.iter_content(8<<20):h.update(block);size+=len(block)
  assert size==row['bytes'] and h.hexdigest()==row['sha256'],row['path']
  observed=dict(path=row['path'],bytes=size,sha256=h.hexdigest(),method='ANONYMOUS_FIXED_REVISION_COMPLETE_CONTENT_STREAM_SHA256',elapsed_s=time.time()-started,verified_epoch_s=time.time())
  save(dest,observed);rows.append(observed)
  print(json.dumps(dict(asset=asset['asset'],path=row['path'],bytes=size,status='PASS_ANONYMOUS_FULL_STREAM')),flush=True)
 receipt=directory/'FULL_STREAM_RECEIPT.json'
 result=dict(schema='PI0253_STORAGE_FULL_CONTENT_READBACK_V1',status='PASS_ANONYMOUS_FULL_CONTENT_READBACK',asset=asset['asset'],repo_id=asset['repo'],revision=asset['revision'],tree_sha256=asset['tree'],source_receipt_sha256=asset['source_receipt_sha256'],public=True,full_trainer=True,files=rows,total_bytes=sum(r['bytes'] for r in rows),new_model_queries=0,new_optimizer_updates=0,downloaded_payload_retained=False)
 save(receipt,result)
 print(json.dumps(dict(asset=asset['asset'],status=result['status'],total_bytes=result['total_bytes'])),flush=True)
