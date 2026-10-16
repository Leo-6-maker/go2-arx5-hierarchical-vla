"""Original-prefix S1 on existing fixed probes, from exact command-history replay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from scripts.pi0292.durable import OwnerLock,read,write
from scripts.pi0292.eval_contract import matrix10,gate_action
from scripts.pi0294.episode import _robot_contact
from scripts.pi0294.scene import make_engine
from scripts.pi0294.teacher import make_teacher
from scripts.pi0294.s1_preflight import predict
from scripts.pi0294.native_dataset import sha
from scripts.pi0291_arx5_data.config import PARAMETERS as P
from go2arx5_vla.task_data.codec import pose10
from .scope import validate,Ledger,DATA
from .initial_history import qualify_recorded_initial
from .collision import reconstruct_case


def check(root,packet,label,kind,row,engine,ledger,physical,stage):
    identifier=kind+'-'+label+'-'+row['id']
    state=dict(tcp=engine.pose('tool'),gripper_fraction=float(np.mean(engine.data.qpos[engine.finger_q])/P['finger_travel']),time_s=float(engine.data.time))
    actual=pose10(state['tcp'],state['gripper_fraction']);physical=np.asarray(physical)
    if kind=='on_expert':
        with np.load(packet/'evidence'/label/'probes'/(row['id']+'.npz'),allow_pickle=False) as z:
            if np.max(np.abs(actual-z['state']))>1e-10:raise ValueError('E1_EXPERT_PROBE_REPLAY_STATE_MISMATCH')
    else:
        if np.max(np.abs(actual-np.asarray(row['payload']['state_eef10'])))>1e-10:raise ValueError('E1_STUDENT_PROBE_REPLAY_STATE_MISMATCH')
    reference=gate_action(physical,actual,state['time_s'],row['id'])
    forecast=None
    if reference['decision']=='PASS':
        with ledger.steps('shadow_integrations',200,'PROBE-'+identifier) as meter:
            def charge(count,*,invocation):
                if not invocation:meter['actual']+=count
            forecast,_=predict(engine,matrix10(physical[:8]),physical[:8,9],nonce=identifier,guard=ledger.guard,charge=charge)
    value=dict(id=row['id'],model=label,population=kind,task=row['task'],stage=stage,
        source_tick=engine.steps,reference_decision=reference['decision'],original_S1=forecast,
        prefix_executable=bool(forecast and forecast['decision']=='PASS'),
        live_actions_installed=0,shadow_not_closed_loop=True)
    path=root/'evidence/PROBE_S1_ROWS'/identifier;write(path.with_suffix('.json'),value)
    return value


def expert_replay(root,packet,manifest,probes,labels,ledger):
    binding=read(root/'plans/PROBE_REPLAY_BINDINGS.private.json');bank=Path(binding['original_v3_0_native'])
    if sha(bank/'DATASET_MANIFEST.json')!=DATA:raise ValueError('E1_ORIGINAL_PROBE_DATA_CHANGED')
    rows={r['id']:r for r in manifest['episodes']};records=[];replays=[]
    for episode_id in sorted({r['episode_id'] for r in probes}):
        row=rows[episode_id];folder=Path(row['raw_root']);card=read(folder/'CARD.json')
        if card['split']!='TRAIN' or sha(folder/'samples.json')!=row['samples_sha256'] or sha(folder/'physics.npz')!=row['physics_sha256']:
            raise ValueError('E1_ORIGINAL_EXPERT_REPLAY_RAW_CHANGED')
        engine=make_engine(card,read(root/'INPUT_BINDINGS.private.json')['asset_root']);first=engine.snapshot();teacher=make_teacher(card,first['object_pose'],engine)
        initial=read(folder/'INITIAL_STATE.json');samples=read(folder/'samples.json');selected={r['anchor']:r for r in probes if r['episode_id']==episode_id}
        maximum=0.
        try:
            if (hashlib.sha256(engine.xml.encode()).hexdigest()!=initial['model_xml_sha256']
                    or not np.array_equal(first['qpos'],initial['qpos']) or not np.array_equal(first['qvel'],initial['qvel'])):
                raise ValueError('E1_EXPERT_INITIAL_HISTORY_MISMATCH')
            with np.load(folder/'highrate.npz',allow_pickle=False) as z:recorded=z['rows'].copy()
            for anchor in range(max(selected)+1):
                ledger.guard();state=engine.snapshot();state['robot_object_contact']=_robot_contact(engine,state)
                if anchor==0:qualify_recorded_initial(engine,folder,ledger.guard)
                teacher.observe(state)
                if anchor in selected:
                    for label in labels:
                        probe=selected[anchor]
                        with np.load(packet/'evidence'/label/'probes'/(probe['id']+'.npz'),allow_pickle=False) as z:physical=z['physical_H40'].copy()
                        records.append(check(root,packet,label,'on_expert',probe,engine,ledger,physical,samples[anchor]['privileged']['teacher_stage']))
                if anchor==max(selected):break
                command=samples[anchor]['supervision']
                if not command['actually_committed'] or not command['valid']:raise ValueError('E1_EXPERT_ACTUAL_COMMAND_REQUIRED')
                target=np.asarray(command['committed']);opening=float(command['committed_gripper'])
                with ledger.steps('shadow_integrations',25,'PROBE-EXPERT-REPLAY-'+episode_id+'-'+str(anchor)) as meter:
                    for _ in range(25):
                        previous=engine.steps
                        try:engine.step(target,opening)
                        finally:meter['actual']+=engine.steps-previous
                        current=engine.snapshot();actual=np.r_[engine.data.time,engine.data.qpos,engine.data.qvel,engine.data.ctrl,engine.data.xfrc_applied[engine.ids['tool']],float(not current['safety_valid'])]
                        error=float(np.max(np.abs(actual-recorded[engine.steps-1])));maximum=max(maximum,error)
                        if error>1e-10 or engine.safety:raise ValueError('E1_EXPERT_HIGH_RATE_HISTORY_MISMATCH')
            replays.append(dict(episode_id=episode_id,maximum_actual_highrate_difference=maximum,replayed_integrations=engine.steps,
                initial_static_qualifier_replayed=True,raw_replayed_not_qpos_injection=True))
        finally:engine.close()
    return records,replays


def run(root):
    root=Path(root);validate(root,'4080-simulation',physical=True);ledger=Ledger(root,'core');ledger.guard()
    if read(ledger.path)['reservations']:raise PermissionError('E1_NO_PROBE_SHADOW_WITH_ACTIVE_PHYSICS')
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:3]));owner=OwnerLock(root/'E1_SOURCE_OWNER.lock');start=time.process_time()
    packet=root/'datasets/received-fixed-probes/native';receipt=read(packet.parent/'CONSUMPTION.private.json')
    if receipt['status']!='PASS_ACTUAL_FIXED_PROBE_ARRAY_CONSUMPTION':raise PermissionError('E1_ACTUAL_HF_PROBE_CONSUMPTION')
    labels=receipt['index']['available'];probes=read(packet/'plans/PROBE_MANIFEST.json')['windows'];rows=[]
    try:
        binding=read(root/'plans/PROBE_REPLAY_BINDINGS.private.json');bank=Path(binding['original_v3_0_native']);manifest=read(bank/'DATASET_MANIFEST.json')
        rows,replays=expert_replay(root,packet,manifest,probes,labels,ledger)
        old=Path(read(root/'E1_INPUTS.private.json')['E0_root']);registry=read(old/'evidence/model-M2048/REGISTRY.json')
        student=read(packet/'plans/STUDENT_PROBES.private.json')['states']
        for case in sorted([c for c in registry['cases'].values() if c['card']['split']=='TRAIN'],key=lambda c:c['panel_index']):
            replay,anchors=reconstruct_case(root,case,ledger,save=False,diagnose=False)
            try:
                for row in student:
                    if row['original_episode_uuid']!=case['episode_uuid']:continue
                    engine,teacher=anchors[row['sequence']]
                    for label in labels:
                        physical=row['M0_physical_H40'] if label=='M0' else next(r['physical_H40'] for r in read(packet/'evidence'/label/'probes/ON_STUDENT_RESULTS.private.json')['records'] if r['id']==row['id'])
                        rows.append(check(root,packet,label,'on_student',row,engine,ledger,physical,teacher.stage))
            finally:
                for engine,_ in anchors.values():engine.close()
        value=dict(status='ACTUAL_FIXED_PROBE_ORIGINAL_PREFIX_S1_SHADOW_ONLY',rows=rows,expert_replays=replays,
            labels=labels,new_model_queries=0,new_main_starts=0,new_recovery_forks=0,
            global_old_E0_RNG_unrecorded=True,live_closed_loop_result=False)
        write(root/'evidence/PROBE_PREFIX_S1_RESULTS.private.json',value);return value
    except Exception as exc:
        if any(x in str(exc) for x in ['MISMATCH','CHANGED']):ledger.stop_physics('PROBE_REPLAY_SOURCE_INTEGRITY',type(exc).__name__+': '+str(exc))
        write(root/'evidence/PROBE_PREFIX_S1_FAILURE.private.json',dict(error=type(exc).__name__+': '+str(exc),completed_rows=len(rows)));raise
    finally:ledger.cpu(time.process_time()-start);owner.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();v=run(a.root);print(json.dumps(dict(status=v['status'],rows=len(v['rows']))),flush=True)
