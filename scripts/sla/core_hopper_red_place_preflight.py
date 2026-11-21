"""Replay the four original first plans and check the changed pitch without running dynamics."""
import argparse
import json
import hashlib
from pathlib import Path

import mujoco
import numpy as np

from go2arx5_vla.trajectory.so3 import quat_to_matrix
from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
from go2arx5_vla.world_model.multitask_teacher_v6 import MultitaskTeacher
from go2arx5_vla.world_model.native_replay import forbid_integration
from scripts.sla.core_hopper_expert import repair_rows
from scripts.sla.core_hopper_red_approach import RedPlacePitch, PLACE_KIND


def base_hold(folder, reference):
    with (folder/'planner.jsonl').open() as log:
        for line in log:
            event=json.loads(line)
            if (event.get('kind')!='installed' or event.get('ref_id')!=reference['accepted_wire_sha256']
                    or event.get('observation_time_s')!=reference['receiver_observation_time_s']):
                continue
            k=reference['receiver_knot_index']
            pose=event['poses'][k]
            if (event['times'][k]!=reference['receiver_knot_time_s'] or len(pose)!=14
                    or pose[:7]!=reference['accepted_pose7']):
                raise ValueError('Saved first reference differs from the receiver knot')
            return pose[7:]
    raise ValueError('Initial constant BASE hold not recorded')


def check(config):
    repair=config['carry_repair']
    rows=repair_rows(read_pin(config['population']),repair)
    prior=Path(read_pin(repair['original_producer'])['registered_root'])
    records={r['condition']:r for r in json.loads((prior/'RECORDS.private.json').read_bytes())}
    audit=read_pin(repair['audit'])
    result=[]
    with forbid_integration(mujoco) as calls:
        for row in rows:
            if row['id'] in records or (prior/'cases'/row['id']).exists():
                raise ValueError('Already opened candidate TRAIN condition')
            side=row['instruction'].split('-')[-1]
            for witness in audit['measurements'][side]['rows']:
                eid=witness['condition']
                rec=records[eid]
                case=prior/'cases'/eid
                spec=json.loads((case/'EXECUTION_SPEC.json').read_bytes())
                output=case/'native-01'
                path=output/'LIVE_DATA/QUERIES.private.json'
                if (rec['split']!='TRAIN' or rec['requests']!=1 or not rec['hard_STOP']
                        or rec['post_STOP_ticks']!=0 or rec['first_failure']!='ILLEGAL_CONTACT'
                        or rec['result']['sha256']!=witness['native_result_sha256']
                        or hashlib.sha256(path.read_bytes()).hexdigest()!=witness['queries_sha256']):
                    raise ValueError('Original contact terminal or its saved queries differ')
                read_pin(rec['result'])
                q=json.loads(path.read_bytes())[0]
                sample=q['actual_state']
                ref=q['recent_reference']
                pose=ref['accepted_pose7']
                reference=np.eye(4)
                reference[:3,:3]=quat_to_matrix(np.asarray(pose[3:]))
                reference[:3,3]=pose[:3]
                base=base_hold(output,ref)
                args=(sample,spec['mobile_condition'],spec['multitask_geometry'],row['instruction'])
                old=MultitaskTeacher(*args)
                new=RedPlacePitch(*args,None)
                if old.landing is not None or new.landing is not None or q['native_step']!=5703:
                    raise ValueError('Unexpected F2 placement callback or first-query clock')
                a,_=old.decide(sample,reference,base)
                b,ev=new.decide(sample,reference,base)
                if (not np.array_equal(a.decoded_H40,q['nominal_world_H40'])
                        or not np.array_equal(a.reference_path,q['installed_reference_H40'])
                        or b.decoded_H40.shape!=(40,10) or b.reference_path.shape!=(40,10)
                        or ev['teacher_version']!=PLACE_KIND):
                    raise ValueError('Original plan reproduction or full candidate H40 contract failed')
                result.append(dict(prior=eid,candidate=row['id'],native_step=q['native_step'],
                    original_nominal_and_installed_exact=True,candidate_full40_checked=True,
                    first_query_changed=not np.array_equal(a.decoded_H40,b.decoded_H40),
                    nominal_sha256=hashlib.sha256(b.decoded_H40.tobytes()).hexdigest(),
                    installed_sha256=hashlib.sha256(b.reference_path.tobytes()).hexdigest()))
        if calls:
            raise RuntimeError('Unexpected native integration during offline check')
    return dict(source=config['source_sha'],status='FOUR_RECORDED_FIRST_PLANS_VERIFIED',cases=result,
                native_requests=0,native_ticks=0,model_generations=0,optimizer_updates=0,VAL_inputs=0)


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('config',type=Path)
    cli.add_argument('sha256')
    args=cli.parse_args()
    print(json.dumps(check(read_pin(dict(path=str(args.config),sha256=args.sha256)))))
