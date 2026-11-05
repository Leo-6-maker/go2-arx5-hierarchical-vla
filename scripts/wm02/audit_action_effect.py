"""Actual matched-restart action responses; no exact-state counterfactual claim."""
def run(replay):
    import numpy as np
    from go2arx5_vla.world_model.target_replay import load
    from go2arx5_vla.world_model.target_stability import motion,rotation_wxyz
    from go2arx5_vla.world_model.storage import digest
    data,meta=load(replay)
    if meta.get('collection_protocol')!='FIXED_HOLD40_BURN8_DECIDE_RAMP8_V1':raise ValueError('NEXT1_STEADY_RESPONSE_DATA_REQUIRED')
    rows=[]
    for i,episode in enumerate(meta['episodes']):
        if episode['split']!='TRAIN':continue
        state=data['state'][i];p=data['tcp'][i,:,:3,3];base=state[:,:3];quat=state[:,3:7];decomp=motion(base,quat,p);r=decomp['relative_tcp'];R=rotation_wxyz(quat[8]);offset=data['goal'][i,:3]-data['history_goal'][i,0,:3]
        if np.count_nonzero(np.abs(offset)>1e-8)>1:raise ValueError('PREDECLARED_SINGLE_AXIS_ACTION_REQUIRED')
        label='HOLD'if np.linalg.norm(offset)<1e-8 else'PLUS_X'if offset[0]>.0009 else'MINUS_X'if offset[0]<-.0009 else'PLUS_Y'
        rows.append(dict(episode=episode['episode'],parent=episode['parent'],action=label,offset_xyz_m=offset.tolist(),responses={str(h):dict(world_TCP=(p[8+h]-p[8]).tolist(),relative_TCP_in_decision_world=(R@(r[8+h]-r[8])).tolist(),BASE=(base[8+h]-base[8]).tolist())for h in [1,8,16]}))
    parents=sorted({r['parent']for r in rows});result=[]
    for parent in parents:
        group=[r for r in rows if r['parent']==parent];holds=[r for r in group if r['action']=='HOLD'];actions=[r for r in group if r['action']!='HOLD']
        if len(holds)!=2 or len(actions)!=3 or {r['action']for r in actions}!={'PLUS_X','MINUS_X','PLUS_Y'}:raise ValueError('EXACT_FOUR_ACTIONS_AND_PREREGISTERED_HOLD_REPEAT_PER_PARENT_REQUIRED')
        for h in [1,8,16]:
            for frame in ['world_TCP','relative_TCP_in_decision_world','BASE']:
                hold=np.asarray([r['responses'][str(h)][frame]for r in holds]);mean=hold.mean(0);noise=hold.std(0,ddof=1)
                for action in actions:
                    offset=np.asarray(action['offset_xyz_m']);axis=int(np.argmax(np.abs(offset)));sign=np.sign(offset[axis]);effect=np.asarray(action['responses'][str(h)][frame])-mean;projected=float(effect[axis]*sign)
                    result.append(dict(parent=parent,horizon=h,frame=frame,action=action['action'],matched_effect_xyz_m=effect.tolist(),signed_projection_m=projected,HOLD_mean_xyz_m=mean.tolist(),HOLD_sample_std_xyz_m=noise.tolist(),HOLD_noise_floor_m=1e-6,signal_over_HOLD_std=abs(projected)/max(float(noise[axis]),1e-6),direction_correct=projected>0))
    gate={}
    for h in [8,16]:
        p=[]
        for parent in parents:
            selected=[r for r in result if r['parent']==parent and r['horizon']==h and r['frame']=='relative_TCP_in_decision_world']
            p.append(dict(parent=parent,all_three_directions_correct=all(r['direction_correct']for r in selected),median_SNR=float(np.median([r['signal_over_HOLD_std']for r in selected]))))
        count=sum(r['all_three_directions_correct']and r['median_SNR']>=3 for r in p)
        gate[str(h)]=dict(parents=p,parents_meeting_direction_and_SNR=count,qualified=count>=3)
    rng=np.random.default_rng(20261105);bootstrap=[]
    for h in [8,16]:
        for frame in ['world_TCP','relative_TCP_in_decision_world']:
            for action in ['PLUS_X','MINUS_X','PLUS_Y']:
                values=np.asarray([r['signed_projection_m']for r in result if r['horizon']==h and r['frame']==frame and r['action']==action]);means=values[rng.integers(len(values),size=(2000,len(values)))].mean(1)
                bootstrap.append(dict(horizon=h,frame=frame,action=action,physical_parents=len(values),mean_signed_effect_m=float(values.mean()),parent_bootstrap95_m=np.quantile(means,[.025,.975]).tolist()))
    passed=len(parents)==4 and all(g['qualified']for g in gate.values())
    return dict(schema='WM02_NEXT1_ACTION_EFFECT_AUDIT_V1',status='ACTUAL_ACTION_EFFECT_DIAGNOSTIC_PASSED'if passed else'ACTUAL_ACTION_EFFECT_DIAGNOSTIC_FAILED_PRESERVED',gate_passed=passed,gate_scope='Relative TCP increments in decision world axes under matched independent restarts; absolute world response also reported',replay_sha256=digest(replay),TRAIN_parents=len(parents),TRAIN_windows=len(rows),raw_responses=rows,matched_effects=result,horizon_gates=gate,parent_bootstrap=bootstrap,exact_counterfactual_restore=False,optimizer_updates=0,physics_integrations=0,Actor_updates=0)


def main(argv=None):
    import argparse,json
    from pathlib import Path
    from go2arx5_vla.world_model.storage import write_json
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--replay',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv);v=run(a.replay);write_json(a.output,v);print(json.dumps({k:v[k]for k in ['status','gate_passed','TRAIN_parents','TRAIN_windows','horizon_gates']}));return 0


if __name__=='__main__':raise SystemExit(main())
