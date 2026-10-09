"""Pair stored TRAIN8 initial inputs and all640 S4 noise/time packets, no new query."""
import sys,json
from pathlib import Path
sys.path.insert(0,'scripts/pi0276_execution');from coordinator import remote
PROGRAM=r'''
import json,hashlib,time
from pathlib import Path
import numpy as np
r=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0286');points=['C0_REPLAY1024-1024','C1_CORRECT1024-1024'];panels={};proofs={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
for point in points:
 p=r/'source/evidence'/(point+'_TRAIN.json');c=r/'audit'/(point+'_TRAIN_CPU.json');v=json.loads(p.read_text());cv=json.loads(c.read_text());assert cv['source_sha256']==sha(p) and cv['status']=='PASS_INDEPENDENT_CPU_CONTACT_SCORE_CLOCK_INSTALLATION_AND_PAIRED_INITIAL_INPUTS';assert len(v['rows'])==8;panels[point]={x['layout']:x for x in v['rows']};proofs[point]=dict(source_sha256=sha(p),CPU_sha256=sha(c))
rows=[];noise_equal=0;time_equal=0;initial_equal=0;K_equal=0
for seed in range(285100,285108):
 a,b=[panels[p][seed] for p in points];assert a['task']==b['task']=='T1' and a['instruction']==b['instruction'];fa,fb=Path(a['folder']),Path(b['folder']);qa,qb=[json.loads((f/'QUERIES.json').read_text()) for f in (fa,fb)];assert len(qa)==len(qb)==80
 initial={};mismatch=[]
 with np.load(fa/'q000/input.npz',allow_pickle=False) as aa,np.load(fb/'q000/input.npz',allow_pickle=False) as bb:
  for key in ('actual_state','base_raw','wrist_raw','instruction'):
   x,y=aa[key],bb[key];equal=bool(np.array_equal(x,y));initial[key]=dict(exact=equal,shape=list(x.shape),dtype=str(x.dtype),C0_array_sha256=ah(x),C1_array_sha256=ah(y))
 initial_ok=all(x['exact'] for x in initial.values());initial_equal+=initial_ok
 ha,hb=[],[];ta,tb=[],[]
 for i,(x,y) in enumerate(zip(qa,qb)):
  assert x['index']==y['index']==i and x['task_time_s']==y['task_time_s']==i*.4
  with np.load(fa/f'q{i:03d}/output.npz',allow_pickle=False) as aa,np.load(fb/f'q{i:03d}/output.npz',allow_pickle=False) as bb:
   na,nb=aa['noise_uint16'],bb['noise_uint16'];xa,xb=aa['native_time_buckets'],bb['native_time_buckets'];assert na.shape==nb.shape==(1,40,132) and na.dtype==nb.dtype==np.uint16
   ne=bool(np.array_equal(na,nb));te=bool(np.array_equal(xa,xb));noise_equal+=ne;time_equal+=te;K_equal+=int(aa['K'])==int(bb['K'])==4
   assert str(aa['source_sha256'])==x['input_sha256'] and str(bb['source_sha256'])==y['input_sha256'];ha.append(ah(na));hb.append(ah(nb));ta.append(ah(xa));tb.append(ah(xb))
   if not ne or not te:mismatch.append(dict(index=i,noise_exact=ne,time_exact=te))
 ca=json.loads((fa/'EPISODE_CARD.json').read_text());cb=json.loads((fb/'EPISODE_CARD.json').read_text());scene_equal=ca['scene']==cb['scene']
 rows.append(dict(layout=seed,query_count=80,scene_exact=scene_equal,instruction_exact=True,initial_q0_actual_arrays=initial,initial_q0_exact=initial_ok,noise_packet_sequence_C0_sha256=hashlib.sha256(json.dumps(ha).encode()).hexdigest(),noise_packet_sequence_C1_sha256=hashlib.sha256(json.dumps(hb).encode()).hexdigest(),native_time_sequence_C0_sha256=hashlib.sha256(json.dumps(ta).encode()).hexdigest(),native_time_sequence_C1_sha256=hashlib.sha256(json.dumps(tb).encode()).hexdigest(),mismatches=mismatch))
for p,v in proofs.items():assert sha(r/'source/evidence'/(p+'_TRAIN.json'))==v['source_sha256'] and sha(r/'audit'/(p+'_TRAIN_CPU.json'))==v['CPU_sha256']
assert initial_equal==8 and noise_equal==time_equal==K_equal==640 and all(x['scene_exact'] for x in rows)
print(json.dumps(dict(status='PASS_ACTUAL_SAVED_TRAIN8_INITIAL_SCENE_RGB_STATE_LANGUAGE_AND_640_NATIVE_NOISE_TIME_K4_PAIRS',epoch=time.time(),points=points,source_and_CPU_before_after_identity=proofs,layouts=list(range(285100,285108)),matched_initial_inputs=initial_equal,matched_noise_packets=noise_equal,matched_time_packets=time_equal,matched_K4=K_equal,rows=rows,interpretation='Same released conditions and initial actual inputs plus common native noise/time. Later RGB/state can diverge through different policies; not required equal and not a broad causal or heldout-generalization theorem.',model_queries=0,new_codec_calls=0,new_gradient=0,physics=0,render=0)))
'''
if __name__=='__main__':
 wrapper='import os,subprocess\nenv=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",MKL_NUM_THREADS="4")\nr=subprocess.run(["/home/liuyu-nfs/Projects/go2-arx5/PI0250/runtime/venv/bin/python","-c",'+repr(PROGRAM)+'],env=env,capture_output=True,text=True)\nassert r.returncode==0,r.stderr\nprint(r.stdout)\n'
 v=json.loads(remote('core',wrapper));Path('reports/PI0286/final/TWO_ARM_TRAIN8_INITIAL_NOISE_PAIRING.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({k:v[k] for k in ('status','matched_initial_inputs','matched_noise_packets','matched_time_packets','matched_K4')}))
