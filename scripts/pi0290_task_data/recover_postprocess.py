"""Recover only recorded C2901 CPU postprocessing; never run physics."""
import argparse
import json
from pathlib import Path
from go2arx5_vla.task_data.io import write,manifest,file_hash
from go2arx5_vla.task_data.score import saved_score
from go2arx5_vla.task_data.resource import Budget

p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
folder=root/'episodes/C2901';budget=Budget(root)
event=next(x for x in budget.value['starts'] if x['id']=='C2901')
assert event['status']=='STARTED' and event['steps']==16000 and event['pairs']==641
result=json.loads((folder/'RESULT.json').read_text());assert result['status']=='VALID_COMPLETE'
original_hash=file_hash(folder/'RESULT.json');end=(folder/'RESULT.json').stat().st_mtime
upper=max(0,end-event['epoch']+10)*4
result['score']=saved_score(folder)
result['postprocess_recovery']=dict(physics_integrations=0,quality_retry=False,
    original_result_sha256=original_hash,error='generic T1 scorer rejected shifted empty-grasp fixture source-goal separation',
    correction='Separate contact-negative scoring; never admitted as T1 quality success',
    original_run_CPU_s_conservative_upper=upper,original_run_CPU_measured=False)
write(folder/'RESULT.json',result);write(folder/'FILE_MANIFEST.json',manifest(folder))
event.update(status='VALID_COMPLETE',CPU_s_upper=upper,CPU_s_measured=False,GPU_s=0)
budget.value['CPU_s']+=upper
budget.finish_cpu('C2901_POSTPROCESS_RECOVERY')
print(json.dumps(result))
