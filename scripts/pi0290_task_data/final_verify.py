"""Final bounded accounting/source/test audit. No new episode or render."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from go2arx5_vla.task_data.io import write,file_hash
from go2arx5_vla.task_data.resource import Budget,owned_bytes,LIMITS
from go2arx5_vla.task_data.config import COMMISSIONING,PILOTS

p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]));budget=Budget(root)
seal=json.loads((root/'TEACHER_AND_LAYOUT_SEAL.json').read_text());proof={}
for module in ['config','engine','scene','teacher','score','codec','io','resource']:
    name='src/go2arx5_vla/task_data/'+module+'.py';assert file_hash(root/'code'/name)==seal['code_files'][name]
    proof[name]=seal['code_files'][name]
for name in ['scripts/pi0290_task_data/run_episode.py','scripts/pi0290_task_data/video.py']:
    assert file_hash(root/'code'/name)==seal['code_files'][name];proof[name]=seal['code_files'][name]
starts=budget.value['starts'];assert [s['id'] for s in starts]==list(COMMISSIONING+PILOTS)
assert sum(s['steps'] for s in starts)==budget.value['physics_steps']==176194
assert sum(s['pairs'] for s in starts)==budget.value['pairs']==7059
assert all(s['status']!='STARTED' for s in starts)
assert budget.value['pairs']+budget.value['diagnostic_pairs_actual']==7093<=LIMITS['pairs']
for identifier in PILOTS:
    result=json.loads((root/'episodes'/identifier/'RESULT.json').read_text())
    assert result['source_sha']==seal['source_sha'] and result['status']=='VALID_COMPLETE' and result['score']['persistent_success']
test=subprocess.run([str(root/'runtime/venv/bin/python'),'-B','-m','pytest','-q',str(root/'code/tests/task_data')],capture_output=True,text=True,
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
assert test.returncode==0,test.stdout+test.stderr
query=subprocess.run(['pgrep','-u',str(os.getuid()),'-a','-f','[r]un_episode.py|[v]ideo.py|[a]rchive_private_hf.py|[e]xport_native_windows.py'],capture_output=True,text=True)
assert query.returncode==1 and not query.stdout.strip(),'Project instance still active'
budget.guard()
write(root/'evidence/FINAL_VERIFY.json',dict(status='PASS_REQUIREMENT_AUDIT',sealed_physical_code_unchanged=proof,
    physics_instances_live=0,new_integrations=0,formal_pairs=7059,diagnostic_pairs=34,total_pairs=7093,
    steps=176194,starts=12,no_quality_retries=True,pytest_stdout=test.stdout,pytest_exit_code=test.returncode,
    owned_output_bytes=owned_bytes(root),resource_guard_pass=True,GR00T=0,WBC=0,Hopper_new_PBS=0))
budget.value['total_pairs_including_diagnostics']=7093
budget.value['diagnostic_pairs_provenance']='2 zero-step software dual probes +10 first optics diagnostics +10 X-side optics diagnostics +12 fixed-layout reset preflight pairs'
budget.value['selected']='NONE';budget.value['deployment']='NONE'
budget.value['owned_output_bytes']=owned_bytes(root)
budget.finish_cpu('D4_FINAL_SOURCE_RESOURCE_AND_RUNTIME_TEST_AUDIT')
print('PASS: 12 terminal attempts, 176194 integrations, 7093 all paired renders, immutable physical code, no live simulation')
