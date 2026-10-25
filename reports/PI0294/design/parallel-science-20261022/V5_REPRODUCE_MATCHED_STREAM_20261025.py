"""CPU-only paired stream on fixed TRAIN manifests; no images/model/transport."""
import argparse,base64,collections,hashlib,json,sys,time,gzip
from pathlib import Path
from types import SimpleNamespace
parser=argparse.ArgumentParser()
for name in ['source-materials','stream-output','output']:parser.add_argument('--'+name,required=True)
args=parser.parse_args();R=Path(__file__).resolve().parents[4];sys.path[:0]=[str(R),str(R/'src')]
from scripts.pi0294.pv1.mix1024 import *
from scripts.pi0294.pv1.sampler import MatchedSampler
materials=json.loads((Path(args.source_materials)).read_bytes())['files'];raws={k:base64.b64decode(v['base64']) for k,v in materials.items()}
# Source packet sizes and hashes were already bound; recheck the actual inputs used.
for k in ['original_manifest','C1_manifest','new_manifest','frozen_source_pair_recipe']:
 assert hashlib.sha256(raws[k]).hexdigest()==materials[k]['sha256']
rows=[]
for key,prefix in [('original_manifest','original_'),('C1_manifest','C1_'),('new_manifest','new_')]:
 for raw in json.loads(raws[key])['episodes']:
  if raw['split']!='TRAIN':continue
  row=dict(raw,PV1_lane=prefix+raw['task']);anchors=raw.get('legal_anchors',list(range(raw['windows'])))
  assert len(anchors)==raw['windows'] and len(set(anchors))==len(anchors)
  rows.extend((row,int(a)) for a in anchors)
manifest_sha=digest({k:materials[k]['sha256'] for k in ['original_manifest','C1_manifest','new_manifest']})
dataset=SimpleNamespace(rows=rows,manifest_sha=manifest_sha,PV1_route='COVERAGE_FIRST',PV1_arm='B_COVERAGE')
samplers={a:MixSampler(dataset,arm=a) for a in ARMS};legacy=MatchedSampler(dataset,stop=VISITS)
original=json.loads(raws['frozen_source_pair_recipe']);recipes={a:recipe(original,a) for a in ARMS};assert lr_schedule(recipes['C50'])==lr_schedule(recipes['D25'])
allrows=[];counts={a:collections.Counter() for a in ARMS};c1={a:collections.Counter() for a in ARMS};c1origins={a:collections.Counter() for a in ARMS};shared=0;C1positions=0
for cursor in range(VISITS):
 picks={a:s.selection_at(cursor) for a,s in samplers.items()};assert picks['C50']==legacy.selection_at(cursor)
 l,z=picks['C50'],picks['D25'];assert l['task']==z['task']
 if l['lane']==z['lane']:assert l==z;shared+=1
 if z['lane'].startswith('new_'):assert l==z
 if l['lane']=='C1_T1':
  assert l==z;C1positions+=1
  for a,p in picks.items():
   raw=rows[p['index']][0];c1[a][raw['id'],p['anchor']]+=1
   # Original FULL anchor0 is excluded by actual legal_anchors; preserve the original algorithm.
   origin='INTERVENTION_START' if raw['id'].endswith('SHORT') and p['anchor']==0 else 'EXPERT_CONTINUATION'
   c1origins[a][origin]+=1
 for a,p in picks.items():counts[a][p['lane']]+=1
 allrows.append(dict(cursor=cursor,**picks))
for batch in range(1024):
 for a in ARMS:
  c=collections.Counter(x[a]['lane'] for x in allrows[batch*32:(batch+1)*32])
  expected=dict(C1_T1=8,original_T1=4,new_T1=4,original_T2=8,new_T2=8) if a=='C50' else dict(C1_T1=8,original_T1=6,new_T1=2,original_T2=12,new_T2=4)
  assert c==expected
for a in ARMS:
 full=[x[a] for x in allrows]
 for step in [1,256,511,1023]:
  suffix=MixSampler(dataset,arm=a,start=step*32)
  assert full[step*32:]==[suffix.selection_at(c) for c in range(step*32,VISITS)]
assert c1['C50']==c1['D25']
raw=(json.dumps(allrows,sort_keys=True,separators=(',',':'))+'\n').encode();out=Path(args.stream_output);out.parent.mkdir(parents=True,exist_ok=True)
with out.open('wb') as f:
 with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as g:g.write(raw)
out.chmod(0o600)
result=dict(schema='V5_C50_D25_ACTUAL_CPU_STREAM_V1',status='PASS_ACTUAL_NEW_IMPLEMENTATION_ON_FROZEN_MANIFEST_SUPPORT',
 prepared_only_not_actual_training=True,native_dispatch_connected=False,all_positions_checked=VISITS,all_batches_checked=1024,
 input_manifest_sha256={k:materials[k]['sha256'] for k in ['original_manifest','C1_manifest','new_manifest']},
 reader_support_sha256=samplers['C50'].support_sha256,common_dataset_identity_sha256=manifest_sha,recipes=recipes,
 lane_counts={a:dict(v) for a,v in counts.items()},C1_positions=C1positions,C1_exact_per_cursor=True,C1_actual_origin_histogram={a:dict(c) for a,c in c1origins.items()},
 shared_lane_identical_positions=shared,D_new_is_C_new_subset=True,C50_equals_legacy_B_1024_sampling_prefix_not_LR_schedule=True,
 lr_1025_values={a:lr_schedule(r) for a,r in recipes.items()},resume_suffix_steps_checked=[1,256,511,1023],
 paired_stream_uncompressed_sha256=hashlib.sha256(raw).hexdigest(),paired_stream_archive_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
 source_implementation_sha256=hashlib.sha256((R/'scripts/pi0294/pv1/mix1024.py').read_bytes()).hexdigest(),new_native_models=0,new_training=0,new_physics=0)
result['preparer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
q=Path(args.output);q.parent.mkdir(parents=True,exist_ok=True);q.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['recipes','lr_1025_values']}))
