"""Bind actual staged bytes and complete parent before the sole PBS request."""
import base64,json,sys
from pathlib import Path
from prepare import remote,ROOTS,REPORT


def run(commit):
    saved=json.loads((REPORT/'PARENT_AND_PRIOR_PROCESS_READBACK.json').read_text())['hop']['parent_inventory']
    files={Path(r['path']).name:dict(bytes=r['sealed_bytes'],sha256=r['sealed_sha256']) for r in saved}
    root=ROOTS['hop'];audit=json.loads((REPORT/'PLAN_AUDIT.json').read_text())
    card=dict(authority_release='508e3db7b63d3c47758794f3ffbad87b577e6e9d',canonical=6048886543,stage='MORE',launch_allowed=True,parent='/scratch/e1192548/Projects/go2-arx5/experiments/PI0282/learning/ACTION2048/main/checkpoint-2048',parent_export='/scratch/e1192548/Projects/go2-arx5/experiments/PI0282/learning/ACTION2048/snapshots/step-2048/EXPORT.json',parent_full_master_sha256='cd845d21b9a953a02d74dc49c1f15b5d5136cfce2e915c5f8539ae7eb6b8dc77',parent_phase_step=2048,parent_global_update=2048,parent_files=files,plan_path=root+'/data/MORE_PLAN.jsonl',plan_sha256=audit['plans']['MORE']['sha256'],updates=2048,warmup=32,peak_LRs=[5e-5,5e-6],only_scheduler_reset=True,selected='NONE',deployment='NONE')
    (REPORT/'MORE_CARD.json').write_text(json.dumps(card,indent=2)+'\n')
    for host,root in ROOTS.items():
        script=f'''import hashlib,json,os,subprocess,shutil
from pathlib import Path
r=Path({root!r});old=r.parent/'PI0282';assert not (r/'runtime/TRAINING_ALLOCATION.json').exists()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
card=json.loads((old/'design/SOURCE_CARD.json').read_text()) if (old/'design/SOURCE_CARD.json').exists() else {{}}
if {host!r}=='core':
 card['canonical']=6048886543;card['authority_release']='508e3db7b63d3c47758794f3ffbad87b577e6e9d';card['points']=['MORE1024','MORE2048','EVENT1024'];card['development_points']=[]
 for n in ('codec_command','policy_command'):card[n]=[s.replace('/PI0282','/PI0283') for s in card[n]]
 card['native_pythonpath']=card['native_pythonpath'].replace('/PI0282','/PI0283');card['code_sha256']={{n:sha(r/'code'/n) for n in card['code_sha256']}}
 card['physical_caps']=dict(starts=40,control=30000,physics=160000,rgb=30000,dispatches=60000,teacher=0);card['model_caps']=dict(requests=3456,heads=13824)
 (r/'design/SOURCE_CARD.json').write_text(json.dumps(card,indent=2)+'\\n')
 points=json.loads((old/'design/MODEL_POINTS.json').read_text());(r/'design/MODEL_POINTS.json').write_text(json.dumps({{'ACTION2048':points['ACTION2048']}},indent=2)+'\\n')
parent_card={json.loads((REPORT/'MORE_CARD.json').read_text())!r}
if {host!r}=='hop':
 (r/'design/MORE_CARD.json').write_text(json.dumps(parent_card,indent=2)+'\\n')
 for n,item in parent_card['parent_files'].items():assert sha(Path(parent_card['parent'])/n)==item['sha256'],n
else:
 for n,h in points['ACTION2048']['files'].items():assert sha(Path(points['ACTION2048']['model_root'])/n)==h,n
manifest=json.loads((old/'design/IMPLEMENTATION.json').read_text());manifest.update(canonical=6048886543,authority_release='508e3db7b63d3c47758794f3ffbad87b577e6e9d',published_commit={commit!r},files={{p.name:sha(p) for p in (r/'code').iterdir() if p.is_file()}},training_allowed=True,model_evaluation_allowed=True,expert_codec_replay_allowed=True)
manifest['quality_budget_limits']=dict(policy_requests=32 if {host!r}=='hop' else 3424,S4_heads=128 if {host!r}=='hop' else 13696)
if {host!r}=='core':manifest['source_card_sha256']=sha(r/'design/SOURCE_CARD.json')
(r/'design/IMPLEMENTATION.json').write_text(json.dumps(manifest,indent=2)+'\\n')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONPATH=str(r/'code'))
python='/scratch/e1192548/stage22_h200_runtime_deb920_20260902/runtime/venv/bin/python' if {host!r}=='hop' else '/home/liuyu-nfs/Projects/go2-arx5/PI0250/runtime/venv/bin/python'
checks="from authority import load;from pathlib import Path;r=Path("+repr(str(r))+");assert load(r)['formal_cap']==3072;import learning;caps=load(r);assert learning.completed_kind([], 'MORE','continuous',1,caps)=='canary';from plan import self_check;self_check();import py_compile;[py_compile.compile(str(p),doraise=True,cfile=str(r/'runtime'/(p.stem+'.pyc'))) for p in (r/'code').glob('*.py')];print('PASS_PINNED_STAGE_CPU_CAPS_AND_PARSE; model/optimizer/physics/PBS=0')"
subprocess.run([python,'-c',checks],env=env,check=True)
v=dict(status='PASS_ACTUAL_SEALED_RUNTIME_CPU_ADMISSION',host={host!r},published_commit={commit!r},files=manifest['files'],new_model_queries=0,new_optimizer_updates=0,new_physics=0)
(r/'evidence/RUNTIME_CPU_ADMISSION.json').write_text(json.dumps(v,indent=2)+'\\n');print(json.dumps(v))
'''
        out=remote(host,script);(REPORT/(host.upper()+'_RUNTIME_CPU_ADMISSION.txt')).write_text(out);print(out[-500:])


if __name__=='__main__':run(sys.argv[1])
