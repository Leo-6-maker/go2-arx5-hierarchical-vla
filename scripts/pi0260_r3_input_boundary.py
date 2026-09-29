"""Actual train/rollout condition equality and forbidden-field rejection."""
import json
from pathlib import Path

import numpy as np

from pi0260_r2 import ROOT,read,sha
from pi0260_r3_inputs import batch,build,identity,inventory

COSMOS=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0250/inputs/frozen_v1/cosmos_snapshot')


def main():
    processor=build(read(ROOT/'dataset/statistics.json'),COSMOS)
    selected=read(ROOT/'dataset/codec_receipt.json')['selected_post_codec']
    cases=[]
    for split in ('TRAIN','KIN-VAL'):
        for task in ('M1','M3'):
            family=selected[split][0]
            for command in (0,1):
                seq=f'{family}-{task}-c{command}'
                rec=read(ROOT/'dataset'/(seq+'.json'))
                with np.load(ROOT/'dataset'/(seq+'.npz')) as z:
                    state=z['state_v1'][0].copy();label=z['raw_action'][0].copy()
                with np.load(ROOT/rec['rows'][0]['frame']) as z:
                    packet=dict(body_rgb=z['body_rgb'],wrist_rgb=z['wrist_rgb'],state_v1=state,language=rec['sequence']['instruction'])
                free=batch(processor,packet)
                supervised=batch(processor,packet,label)
                assert set(supervised['inputs'])-set(free['inputs'])=={'action','action_mask'}
                condition={k:v for k,v in supervised['inputs'].items() if k not in ('action','action_mask')}
                assert identity(free['inputs'])==identity(condition),('TRAIN_ROLLOUT_CONDITION_DRIFT',seq)
                for bad in ('time','phase','source_id','target_xyz','future_action','segmentation','depth','success'):
                    try:batch(processor,dict(packet,**{bad:1}))
                    except ValueError:pass
                    else:raise AssertionError(('FORBIDDEN_FIELD_ADMITTED',bad))
                cases.append(dict(sequence=seq,split=split,task=task,condition_sha256=identity(free['inputs']),
                                  paths=[x['path'] for x in inventory(free['inputs'])]))
    assert len(cases)==8
    p=ROOT/'R3/input_boundary.json';assert not p.exists()
    p.write_text(json.dumps(dict(status='PASS_ACTUAL_TRAIN_FREE_CONDITION_EQUALITY_AND_ALLOWLIST',cases=cases,
                                 statistics_sha256=sha(ROOT/'dataset/statistics.json'),model_loads=0,model_queries=0),indent=2)+'\n')
    print(json.dumps(dict(status='PASS_ACTUAL_TRAIN_FREE_CONDITION_EQUALITY_AND_ALLOWLIST',cases=len(cases),sha256=sha(p))))


if __name__=='__main__':main()
