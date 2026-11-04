"""Saved first27 target ticks: exact observations plus explicitly unknown causal events."""
from pathlib import Path
import json,time
import numpy as np
from .native_replay import packets
from .storage import digest,write_json


def run(manifest,output):
    start=time.monotonic();cpu=time.process_time();root=Path(manifest['root']);out={};allrows={}
    for episode,pins in manifest['episodes'].items():
        folder=root/episode
        for n,h in pins.items():
            if digest(folder/n)!=h:raise ValueError('Original startup evidence identity differs:'+n)
        rows=packets(folder/'state.bin.journal');initial=json.loads((folder/'initialization.jsonl').read_text().splitlines()[0]);before=json.loads((folder/'BEFORE_FIRST_INTEGRATION.json').read_text());pd=[json.loads(x)for x in(folder/'control.jsonl').read_text().splitlines()];physics=[json.loads(x)for x in(folder/'physics.jsonl').read_text().splitlines()];model=next(x for x in physics if x['kind']=='model');stops=[x for x in physics if x['kind']=='stop'];timeline=[];previous=None
        for row in[dict(step=0,time_s=0.,qpos=np.array(initial['qpos']),qvel=np.array(initial['qvel']),ctrl=np.array(initial['ctrl']),contacts=[])]+rows[:27]:
            t=row['time_s'];known=[x for x in pd if x['sim_time_s']<=t+1e-12];last=known[-1]if known else None;violations=[]
            for j in model['joints']:
                if not j['limited']or j['type']not in(2,3):continue
                q=row['qpos'][j['qposadr']];over=max(0.,j['range'][0]-q,q-j['range'][1])
                if over>0:violations.append(dict(joint=j['name'],penetration=float(over),unit='rad'if j['type']==3 else'm',q=float(q)))
            ctrl=np.array(row['ctrl']);timeline.append(dict(step=row['step'],physics_t_s=t,qpos=row['qpos'].tolist(),qvel=row['qvel'].tolist(),actual_ctrl=ctrl.tolist(),leg_ctrl_changed_from_previous=bool(previous is None or not np.array_equal(ctrl[:12],previous[:12])),saved_PD_rows_at_this_time=[x for x in pd if abs(x['sim_time_s']-t)<1e-9],last_recorded_PD_time_s=last['sim_time_s']if last else None,last_recorded_control_calls=last['control_calls']if last else None,actual_PD_calls_at_this_tick='UNKNOWN_SPARSE_CONTROL_TRACE',policy_inference_begin='UNKNOWN_NOT_RECORDED',policy_observation_history='UNKNOWN_NOT_RECORDED',policy_end_or_output_version='UNKNOWN_NOT_RECORDED',position_velocity_consumed_versions='UNKNOWN_NOT_RECORDED',positive_joint_penetrations=violations,saved_STOP=[x for x in stops if x.get('steps')==row['step']]))
            previous=ctrl
        first=next((r for r in rows if np.any(r['ctrl'][:12]!=0)),None);firststop=stops[0]['steps']if stops else None
        # Explicitly conditional: PD with fixed first published command/zero leg
        # target velocity should change under measured changing q/qd. This is not
        # a reconstruction of unrecorded actual target versions or PD callbacks.
        firstpd=next((x for x in pd if x['sim_time_s']>0 and any(x['kp'][:12])),None);conditional=[]
        if firstpd:
            jointmap=manifest['joint_mapping'];joints=[j for j in model['joints']if j['type']==3][:18]
            for i,row in enumerate(rows[:27]):
                pre=rows[i-1]if i else dict(qpos=np.array(initial['qpos']),qvel=np.array(initial['qvel']))
                if row['time_s']<=firstpd['sim_time_s']or row['step']>(firststop or 27):continue
                native=np.zeros(18)
                for k in range(18):
                    hw=jointmap[k];j=joints[hw];native[hw]=firstpd['tau'][k]+firstpd['kp'][k]*(firstpd['q_target'][k]-pre['qpos'][j['qposadr']])-firstpd['kd'][k]*pre['qvel'][j['dofadr']]
                conditional.append(dict(step=row['step'],conditional_fixed_command_PD_leg_max_abs_difference=float(np.max(abs(native[:12]-row['ctrl'][:12])))))
        out[episode]=dict(input_sha256=pins,timeline=timeline,first_nonzero_leg_ctrl_step=first['step']if first else None,first_STOP_step=firststop,original_packets=len(rows),extra_integrations_after_first_STOP=max(0,rows[-1]['step']-firststop)if firststop is not None else None,initial_control_calls=before['control_calls'],conditional_fixed_first_command_zero_leg_target_velocity_PD=conditional,conditional_limit='Uses recorded first command and preceding generalized states under fixed-command hypothesis, not actual later commands/sensors/callback count; cannot prove unique mutex/queue cause')
        allrows[episode]=rows
    a,b=[allrows[n]for n in('S1_RECOVERY01_T1','S1_RECOVERY01_T2')];diff={}
    for key in('qpos','qvel','ctrl'):
        x=next(((u,v)for u,v in zip(a[:27],b[:27])if not np.array_equal(u[key],v[key])),None);diff[key]=dict(step=x[0]['step'],time_s=x[0]['time_s'],max_abs=float(np.max(abs(x[0][key]-x[1][key]))))if x else None
    result=dict(source_sha=manifest['source_sha'],status='SAVED_FIRST27_TIMELINE_COMPLETE_WITH_UNKNOWNS',episodes=out,first_causal_divergence=diff,unique_leg_root_cause='NOT_ESTABLISHED',new_native_integrations=0,policy_queries=0,CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start);write_json(output,result);return result
