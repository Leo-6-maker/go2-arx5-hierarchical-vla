"""Saved target startup causality audit; no target/model/policy construction or stepping."""
from pathlib import Path
import json
import struct
import time
import numpy as np
from .storage import digest,write_json
from scripts.pi0294.task_state import parse_state


def changes(rows,key,count=12):
    segments=[]
    for row in rows:
        value=row[key][:count]
        if not segments or not np.array_equal(value,segments[-1]['value']):
            segments.append(dict(first_step=row['step'],last_step=row['step'],value=value.copy()))
        else:segments[-1]['last_step']=row['step']
    return [dict(first_step=x['first_step'],last_step=x['last_step'],ticks=x['last_step']-x['first_step']+1,
                 max_abs=float(np.max(np.abs(x['value']))),value=x['value'].tolist())for x in segments]


def first_difference(left,right,key):
    for a,b in zip(left,right):
        if a['step']!=b['step']:raise ValueError('Compare same native steps only')
        diff=np.abs(a[key]-b[key])
        if np.any(diff):return dict(step=a['step'],time_s=a['time_s'],max_abs=float(diff.max()))
    return None


def load_packets(path):
    rows=[]
    with Path(path).open('rb')as f:
        while True:
            size=f.read(4)
            if not size:break
            if len(size)!=4:raise ValueError('Incomplete native journal size')
            size,=struct.unpack('<I',size)
            if not 40<=size<=2*1024**2:raise ValueError('Native journal packet size')
            row=parse_state(f.read(size));rows.append(row)
            if row['step']!=len(rows)or abs(row['time_s']-len(rows)*.002)>1e-9:
                raise ValueError('Native packet clock/continuity differs')
    return rows


def audit(manifest_path,output):
    manifest=json.loads(Path(manifest_path).read_text());began=time.monotonic();cpu=time.process_time();all_rows={};result={}
    for name,entry in manifest['episodes'].items():
        folder=Path(entry['root'])
        for filename,pin in entry['files'].items():
            if digest(folder/filename)!=pin:raise ValueError('Original saved evidence changed:'+filename)
        rows=load_packets(folder/'state.bin.journal');all_rows[name]=rows
        initial=json.loads((folder/'initialization.jsonl').read_text().splitlines()[0])
        before=json.loads((folder/'BEFORE_FIRST_INTEGRATION.json').read_text())
        control=[json.loads(x)for x in (folder/'control.jsonl').read_text().splitlines()]
        physics=[json.loads(x)for x in (folder/'physics.jsonl').read_text().splitlines()]
        model=next(x for x in physics if x.get('kind')=='model')
        stops=[x for x in physics if x.get('kind')=='stop']
        native_stops=[x for x in stops if x.get('reason')=='REGISTERED_JOINT_SAFETY']
        first_nonzero=next((x for x in rows if np.any(np.abs(x['ctrl'][:12])>1e-12)),None)
        fingers=[j for j in model['joints']if j['name']in ('task_left_slide','task_right_slide')]
        previous=dict(qpos=np.array(initial['qpos']),qvel=np.array(initial['qvel']));errors=[];violations=[]
        for row in rows[:100]:
            for i,j in enumerate(fingers):
                # Exact frozen wrapper law with held initial target before task command at t5.
                target=initial['qpos'][j['qposadr']]
                expected=np.clip(400*(target-previous['qpos'][j['qposadr']])-3*previous['qvel'][j['dofadr']],-8,8)
                errors.append(abs(float(row['ctrl'][18+i])-expected))
                q=row['qpos'][j['qposadr']]
                if q<j['range'][0]or q>j['range'][1]:violations.append(dict(step=row['step'],joint=j['name'],q=float(q),qd=float(row['qvel'][j['dofadr']]),ctrl=float(row['ctrl'][18+i])))
            previous=row
        result[name]=dict(evidence_sha256=entry['files'],packets=len(rows),initial_control_calls=before['control_calls'],
            initial_fsm=before['fsm'],initial_leg_gains=before['kp'][:12],initial_leg_targets=before['q_target'][:12],
            first_nonzero_leg_actuation=dict(step=first_nonzero['step'],time_s=first_nonzero['time_s'],ctrl=first_nonzero['ctrl'][:12].tolist())if first_nonzero else None,
            first27_leg_actuation_segments=changes(rows[:27],'ctrl'),
            first_dynamic_command=next((x for x in control if x['sim_time_s']>0),None),
            first_safety_stop_step=native_stops[0]['steps']if native_stops else None,
            packets_after_first_safety_stop=rows[-1]['step']-native_stops[0]['steps']if native_stops else 0,
            gripper_initial_target_m=[initial['qpos'][j['qposadr']]for j in fingers],
            first100_gripper_PD_reconstruction_max_error=float(max(errors)),gripper_raw_limit_violations=violations,
            native_stops=native_stops)
    a,b=all_rows['S1_RECOVERY01_T1'],all_rows['S1_RECOVERY01_T2']
    report=dict(status='SAVED_INITIALIZATION_CAUSALITY_AUDIT',episodes=result,
        first_difference={k:first_difference(a[:27],b[:27],k)for k in ('qpos','qvel','ctrl')},
        new_target_integrations=0,leg_policy_forwards=0,WM_updates=0,CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-began,
        inference_limit='Sparse command traces do not record every policy observation/history or inference completion. Timing association alone is not unique controller root-cause proof.')
    write_json(output,report);return report
