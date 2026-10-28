import argparse
import json
import os
from pathlib import Path
from go2arx5_vla.task_data.audit import audit_episode
from go2arx5_vla.task_data.config import COMMISSIONING,PILOTS
from go2arx5_vla.task_data.io import write,file_hash
from go2arx5_vla.task_data.resource import Budget

p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--tokenizer',required=True)
p.add_argument('--commissioning-only',action='store_true');a=p.parse_args();root=Path(a.root)
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]));budget=Budget(root)
records=[]
try:
    for identifier in COMMISSIONING if a.commissioning_only else COMMISSIONING+PILOTS:
        records.append(audit_episode(root/'episodes'/identifier))
    negative_gates={}
    for fixture in ['privileged_leak','sensor_time','future_history','codec_rotation']:
        try:audit_episode(root/'episodes/C2903',negative_fixture=fixture)
        except AssertionError:negative_gates[fixture]='PASS_REJECTED_IN_MEMORY_MUTATION_OF_ACTUAL_DATA'
        else:raise RuntimeError('Admission failed to reject '+fixture)
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(a.tokenizer)
    instruction=json.loads((root/'episodes/C2903/CARD.json').read_text())['instruction']
    tokens=tokenizer.encode(instruction,add_special_tokens=False)
    assert tokens.ids and tokenizer.decode(tokens.ids,skip_special_tokens=False)==instruction
    result=dict(status='PASS_STATIC_DATA_AND_NATIVE_CODEC_ONLY',schema='INDEPENDENT_DYNAMIC_TOOL_DATA_V1',episodes=records,
        tokenizer=dict(source='existing Cosmos-Reason2-2B fixed revision 9ce19a195e423419c349abfc86fd07178b230561',
            sha256=file_hash(a.tokenizer),ids=tokens.ids,decode_roundtrip=True,network_loads=0),
        official_GR00T_processor_collator='NOT_RUN',GR00T=0,training_statistics_fitted=False,
        negative_gates=negative_gates,contact_negatives='C2901 actual empty and C2902 actual one-sided commissioning',
        split='whole episode; seen/heldout families; no random overlapping-window split',
        physics_integrations=0)
    write(root/'evidence'/('D1_DATA_AUDIT.json' if a.commissioning_only else 'DATA_ADMISSION_AND_CODEC.json'),result)
    print(json.dumps(result))
finally:budget.finish_cpu('D1_AUDIT' if a.commissioning_only else 'D3_AUDIT')
