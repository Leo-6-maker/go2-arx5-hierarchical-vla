"""Replay fixed D1 intervention decisions and first Teacher prefixes without physics."""
import argparse
import copy
from pathlib import Path
from tempfile import mkdtemp

import mujoco
import numpy as np

from m38_gpu_preflight_v1 import pin,read,require,sha
from scripts.pi0292.durable import write
from go2arx5_vla.world_model.multitask_score_v6 import Score,Task
from go2arx5_vla.world_model.multitask_state_v6 import packets,score_through
from go2arx5_vla.world_model.native_replay import forbid_integration
from v64_recovery_release_prefix import BoundaryReader
from v64_recovery_early_preview import static_prefix
from v7_precontact_teacher import EarlyWarning,prepare_warning


def replay(root,stage):
    lease=read(root/'control/LEASE.json');spec=read(pin(root/'EXECUTION_SPEC.json',lease['spec_sha256']))
    terminal=read(root/'RESULT.private.json')
    require(lease['status']=='CLOSED' and spec['collection_split']=='TRAIN' and terminal['shutdown_clean'],
            'Replay only the closed original TRAIN archive')
    native=root/'native-01';journal=native/'state.bin.journal'
    model=mujoco.MjModel.from_binary_path(str(pin(spec['model_path'],spec['frozen_files'][spec['model_path']])))
    color=Task.parse(spec['task_instruction_id']).color
    reader=BoundaryReader(model,color)
    with forbid_integration(mujoco) as calls:
        initial=next(reader.update(packet) for packet in packets(journal) if packet['step']==5703)
    require(not calls,'Initial reconstruction integrated physics')
    score=Score(spec['task_instruction_id'],initial,spec['multitask_geometry'])
    warning=EarlyWarning(stage,model,color)
    with (native/'COMMANDS.private.jsonl').open() as stream:
        base=next(row['measured_BASE_pose_hold'] for row in map(__import__('json').loads,stream)
                  if row.get('measured_BASE_pose_hold') is not None)
    planned=[(p,read(p)) for p in sorted((native/'g7-queries').glob('*/QUERY_ACTUAL.private.json'))]
    non_full=len([r for _,r in planned if not r.get('full_P10',False)])
    planned=[(p,r) for p,r in planned if r.get('full_P10',False) and not r['mock_consumption']
             and r['terminal_native_step']<terminal['actual_native_integrations']]
    require(planned,'No genuine pre-STOP complete P10 boundaries')
    first=None;refused=[];negative=None;count=0
    with journal.open('rb') as stream:
        for path,consumed in planned:
            step=consumed['terminal_native_step']
            scored=score_through(stream,reader,score,step);sample=reader.latest
            try:
                event=warning.observe(sample,scored,consumed)
            except ValueError as error:
                refused.append(dict(query=consumed['query'],reason=str(error)));continue
            count+=1
            if event is None or first is not None:
                continue
            atom=consumed['confirmed_intervals'][-1]
            previous=dict(width_m=atom['emitted_gripper_width_m'],source_native_s=atom['emitted_gripper_source_native_s'],
                          query_native_s=sample['time_s'])
            first=dict(event=event,query=consumed['query'],consumed_sha256=sha(path),
                seconds_before_archived_terminal=.002*(terminal['actual_native_integrations']-step),
                target_grasp=sample['native_grasp'],maximum_lift_m=scored['maximum_lift_m'],
                previous_width_m=previous['width_m'])
            try:
                with forbid_integration(mujoco) as integrated:
                    teacher,program,evidence,margin=prepare_warning(initial,sample,scored,spec,consumed,previous,base,event)
                    geometry=static_prefix(model,sample,program,color,reader.tcp,sample['native_grasp'])
                require(not integrated,'Teacher preview integrated physics')
                first.update(preview_prepared=True,original_full_H40_margin=margin,static_prefix=geometry,
                    teacher_stage=evidence['stage'],teacher_first_width_m=float(.002+.04*program.decoded_H40[0,9]),
                    desired_P10_delta_m=(program.decoded_H40[10,:3]-program.decoded_H40[0,:3]).tolist())
            except ValueError as error:
                first.update(preview_prepared=False,reason=str(error))
            fixtures=[]
            for label,mutate in (
                ('post_stop',lambda s,c,r:s.update(safety_valid=False)),
                ('partial_P10',lambda s,c,r:r.update(full_P10=False)),
                ('unconfirmed_ticks',lambda s,c,r:r.update(unconfirmed_native_ticks=1)),
                ('score_clock_mismatch',lambda s,c,r:c.update(native_step=c['native_step']+1))):
                s,c,r=copy.deepcopy((sample,scored,consumed));mutate(s,c,r)
                try:EarlyWarning(stage,model,color).observe(s,c,r)
                except ValueError:fixtures.append(label)
                else:raise AssertionError('Invalid D1 boundary admitted: '+label)
            negative=fixtures
    return dict(case=root.name,stage=stage,terminal_sha256=sha(root/'RESULT.private.json'),
        total_complete_pre_stop_queries=len(planned),warning_eligible_queries=count,rejected_boundaries=refused,
        non_full_or_bootstrap_query_records=non_full,
        first_warning=first,invalid_boundary_fixtures_rejected=negative,
        retrospective_only=True,new_native_integrations=0,new_training_labels=0)


def run(site):
    cases=[(site/'V7-D0-EARLY-NATIVE-02/V7_D0_EARLY_TRAIN_14','PREGRASP'),
           (site/'V7-D0-EARLY-NATIVE-02/V7_D0_EARLY_TRAIN_13','LIFT'),
           (site/'RECOVERY-TRAIN-LIVE-05/V64_RECOVERY_TRAIN_09','LIFT')]
    rows=[replay(root,stage) for root,stage in cases]
    output=Path(mkdtemp(prefix='v7-d1-teacher-replay.',dir=site))/'ACTUAL.json'
    write(output,dict(scope='D1_ACTUAL_JOURNAL_PRECONTACT_AND_LIFT_STALL_REPLAY',cases=rows,
          new_native_integrations=0,new_model_forwards=0,new_optimizer_updates=0,
          new_training_labels=0,prospective_native_qualified=False))
    print(output,sha(output),flush=True)
    for row in rows:print(row,flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('site',type=Path)
    run(**vars(cli.parse_args()))
