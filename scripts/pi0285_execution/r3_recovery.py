"""Prospective fixed TRAIN32 recovery; importing this module does no physics work."""
import hashlib,json,time
from collections import Counter,deque
from pathlib import Path
import numpy as np
from contract import ROOT,INSTRUCTIONS,sha,write
from data import immutable,ordered,plan,quotas,WEIGHTS

TIMES=(2,6,10,16)
PREFIX_KEYS=('initial.npz','observations.npz','commands.npz','installed.npz','physics.npz','EPISODE_CARD.json')
FROZEN={'statistics.json':'c3ec9c6b797bb735e3ac38d611b409262de04bbb965cb68b7426df749ba7fb81',
        'processor_config.json':'fae35db9d84083cc1faac16442c062c74a22691fa3358cfc58efb759e01977d3',
        'embodiment_id.json':'fb47934b4bd4b5e11ec1c61ade0c5e03ca9110ec5f16abce9633b10cfd9edfbb'}


def slots():
    return [dict(slot=f'T1-TRAIN-{seed}-TAKEOVER-{seconds:02d}s',task='T1',split='TRAIN',layout=seed,
                 task_time_s=seconds,control_tick=20+seconds*20,instruction=INSTRUCTIONS['T1'])
            for seed in range(285100,285108) for seconds in TIMES]


def array_hash(value):
    value=np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()


def prefix(folder,slot):
    """Bind the actual4096 result and reject an unvisited branch, without replacements."""
    folder=Path(folder);result=json.loads((folder/'RESULT.json').read_text());card=json.loads((folder/'EPISODE_CARD.json').read_text())
    assert result['status'].startswith('VALID') and card['point']=='PLACE4096-4096'
    assert all(card[k]==slot[k] for k in ('task','layout','instruction')) and card['split']=='TRAIN'
    hashes={name:sha(folder/name) for name in PREFIX_KEYS}
    assert all(result['files'].get(name)==h for name,h in hashes.items()),'Partial or changed source prefix'
    tick=slot['control_tick']
    with np.load(folder/'installed.npz',allow_pickle=False) as f:actions=f['action'].copy()
    with np.load(folder/'commands.npz',allow_pickle=False) as f:commands=f['action'].copy()
    with np.load(folder/'observations.npz',allow_pickle=False) as f:observations={k:f[k].copy() for k in ('time_s','state_v1','base_camera_raw','hand_camera_raw')}
    with np.load(folder/'physics.npz',allow_pickle=False) as f:physics={k:f[k].copy() for k in f.files}
    assert len(actions)>=tick and len(commands)>=tick and len(physics['time_s'])>tick*5,'UNVISITED_STATE_COUNTS_IN_32'
    assert len(observations['time_s'])>tick,'MISSING_SAVED_BRANCH_RGB_COUNTS_IN_32'
    assert abs(float(observations['time_s'][tick])-tick*.05)<1e-8
    with np.load(folder/'initial.npz',allow_pickle=False) as f:initial={k:f[k].copy() for k in f.files}
    return dict(folder=str(folder),hashes=hashes,actions=actions[:tick],commands=commands[:tick],physics=physics,
                observations=observations,initial=initial,slot=slot,
                state_hash=array_hash(np.concatenate([physics[k][tick*5].ravel() for k in sorted(physics) if k!='time_s'])))


def match(actual,expected,label,tolerance=1e-6):
    actual=np.asarray(actual);expected=np.asarray(expected)
    assert actual.shape==expected.shape and actual.dtype==expected.dtype,(label,'shape/dtype')
    if actual.dtype.kind in 'fc':
        assert np.isfinite(actual).all() and np.isfinite(expected).all() and np.max(abs(actual-expected),initial=0)<=tolerance,(label,'restore mismatch')
    else:assert np.array_equal(actual,expected),(label,'restore mismatch')


def replay(ep,bound,deadline,flatten_initial=None):
    """Only called by a future admitted collection owner with an existing Episode."""
    if flatten_initial is None:
        from recorder import flatten as flatten_initial
    assert ep.control==0 and ep.seed==bound['slot']['layout']
    initial=flatten_initial(ep.initial);assert set(initial)==set(bound['initial'])
    for k in initial:match(initial[k],bound['initial'][k],k,0)
    for i,(action,command) in enumerate(zip(bound['actions'],bound['commands'])):
        assert time.time()<deadline,'45-minute collection boundary'
        ep.step(action,phase=-1,command=command,teacher=False)
        for n in range(i*5,(i+1)*5+1):
            assert set(ep.states[n])==set(bound['physics'])
            for k in ep.states[n]:match(ep.states[n][k],bound['physics'][k][n],f'{n}:{k}')
    observation=ep.observe();tick=ep.control
    for k in ('state_v1','base_camera_raw','hand_camera_raw'):match(observation[k],bound['observations'][k][tick],k,0)
    return dict(status='PASS_REPLAY_ALL_PHYSICS_SAMPLES_AND_EXACT_BRANCH_RGB_STATE',slot=bound['slot']['slot'],
                source_hashes=bound['hashes'],takeover_tick=tick,state_hash=bound['state_hash'],tolerance=1e-6,RGB_state_byte_exact=True)


def collect_episode(ep,bound,teacher_sha256,deadline):
    """Same teacher, fresh real observations through original task32s; no model calls."""
    from teacher import Teacher
    from source import step
    from gate import original_gate
    from scorer import score
    clock=json.loads((ROOT/'recovery/A/COLLECTION_CLOCK.json').read_text())
    assert deadline==clock['deadline_epoch'] and time.time()<deadline
    assert clock['decision_sha256']==sha(ROOT/'qualification/R2_A_SINGLE_BRANCH.json')
    panel=json.loads((ROOT/'source/evidence/PLACE4096-4096_TRAIN.json').read_text())
    original=next(row for row in panel['rows'] if row['layout']==bound['slot']['layout'] and row['task']=='T1')
    assert Path(original['folder']).resolve()==Path(bound['folder']).resolve()
    assert original==json.loads((Path(bound['folder'])/'RESULT.json').read_text())
    assert teacher_sha256==sha(ROOT/'code/teacher.py')
    restore=replay(ep,bound,deadline);slot=bound['slot'];start=ep.control
    write(ep.folder/'RECOVERY_RESTORE.json',restore)
    teacher=Teacher(ep,slot['task'],ep.env.layout_spec['zone_xy'])
    status='VALID_SOURCE_COMPLETE';gates=[]
    while ep.control<660:
        assert time.time()<deadline,'45-minute collection boundary'
        observation=ep.observe();command,phase=teacher.query((ep.control-start)*.05)
        gate=original_gate(np.tile(command,(40,1)),observation['state_v1'],ep.control*.05,slot['slot']+f'-{ep.control}')
        gates.append(dict(control=ep.control,decision=gate['decision']))
        if gate['decision']=='REJECT':status='VALID_SOURCE_GATE_REJECT';break
        step(ep,command,phase,True)
    physical={k:np.stack([v[k] for v in ep.states]) for k in ep.states[0]}
    scoring=score(physical,slot['task'],ep.env.layout_spec['zone_xy'],ep.states[0]['cubeA_pose'][:3],ep.states[0]['cubeB_pose'][:3],status,ep.control)
    write(ep.folder/'TASK_SCORE_CPU.json',scoring);write(ep.folder/'GATE_RECORDS.json',gates)
    write(ep.folder/'TEACHER_EVENTS.json',dict(events=teacher.events,teacher_time_origin_control=start,within_episode_recovery_attempts=teacher.attempts,teacher_failed=teacher.failed))
    write(ep.folder/'EPISODE_CARD.json',dict(**slot,point='PLACE4096-4096',teacher_sha256=teacher_sha256,
        restore=restore,takeover_tick=start,task_end_tick=660,source='corrected_suffix',truth_offline_only=True,
        scene=ep.env.layout_spec,reset_controls=[],code_seal_sha256=sha(ROOT/'design/CODE_SEAL.json')))
    result=ep.finish(status);result.update(task=slot['task'],split='TRAIN',layout=slot['layout'],instruction=slot['instruction'],
        folder=str(ep.folder),slot=slot['slot'],persistent_success=scoring['persistent_success'],task_score=scoring,
        source='corrected_suffix',model_queries=0,teacher_sha256=teacher_sha256,takeover_tick=start)
    write(ep.folder/'RESULT.json',result);return result


def suffix(root,folder,encoder,physical_errors):
    """Keep raw labels even on codec FAIL; return every real eligible H40 window."""
    root=Path(root);folder=Path(folder);result=json.loads((folder/'RESULT.json').read_text());card=json.loads((folder/'EPISODE_CARD.json').read_text())
    assert encoder.identity==FROZEN,'Recovery must use unchanged frozen processor/statistics/mapping'
    slot=next(s for s in slots() if s['slot']==result['slot']);assert all(card[k]==slot[k] for k in slot)
    assert result['status']=='VALID_SOURCE_COMPLETE' and result['persistent_success'] and result['control_calls']==660
    restore=json.loads((folder/'RECOVERY_RESTORE.json').read_text());assert restore==card['restore'] and restore['status'].startswith('PASS_REPLAY_')
    assert card['reset_controls']==[],'No H40 windows may cross a reset'
    for name,h in result['files'].items():assert sha(folder/name)==h
    tick=slot['control_tick'];episode=slot['slot'];members={}
    with np.load(folder/'observations.npz',allow_pickle=False) as f:obs={k:f[k][tick:660].copy() for k in ('base_camera_raw','hand_camera_raw','state_v1','time_s')}
    with np.load(folder/'commands.npz',allow_pickle=False) as f:actions=f['action'][tick:660].copy();phases=f['phase'][tick:660].copy()
    with np.load(folder/'physics.npz',allow_pickle=False) as f:grasp=f['is_cubeA_grasped'][tick*5:660*5:5].copy()
    count=660-tick
    assert actions.shape==obs['state_v1'].shape==(count,10) and phases.shape==grasp.shape==(count,)
    assert all(len(v)==count for v in obs.values()) and np.isfinite(actions).all() and np.isfinite(obs['state_v1']).all()
    assert np.allclose(obs['time_s'],np.arange(tick,660)*.05,rtol=0,atol=1e-8)
    assert np.isin(phases,range(5)).all() and all(obs[k].dtype==np.uint8 and obs[k].shape==(count,128,128,3) for k in ('base_camera_raw','hand_camera_raw'))
    for namespace,arrays in [('policy_observation',obs),('action_target',dict(action=actions,phase=phases))]:
        for k,v in arrays.items():
            path=root/namespace/episode/(k+'.npy');path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():assert np.array_equal(np.load(path,allow_pickle=False),v)
            else:np.save(path,v,allow_pickle=False)
            members[str(path.relative_to(root))]=sha(path)
    change=np.zeros(count,dtype=bool);change[1:]=(phases[1:]!=phases[:-1])|(grasp[1:]!=grasp[:-1])|(abs(np.diff(actions[:,-1]))>.01)
    rows=[];failures=[];maxima=dict(XYZ_max_m=0.,SO3_max_rad=0.,gripper_max_fraction=0.)
    for a in range(count-39):
        target=actions[a:a+40];state=obs['state_v1'][a]
        try:
            err=physical_errors(target,encoder.decode_actions(encoder.encode_actions(target,state),state))
            valid=not err['nonfinite'] and not err['decoded_rot6d_validity']['degenerate_count'] and not err['command_rot6d_validity']['degenerate_count']
            valid=valid and all(err[k]<=t for k,t in zip(maxima,(1e-5,1e-4,1e-5)))
            for k in maxima:maxima[k]=max(maxima[k],err[k])
        except (ValueError,AssertionError,FloatingPointError) as exc:valid=False;err=dict(error=str(exc))
        if not valid:failures.append(dict(anchor=a,errors=err))
        rows.append(dict(uid=f'{episode}-{a}',episode=episode,task='T1',split='TRAIN',layout=slot['layout'],
            instruction=INSTRUCTIONS['T1'],anchor=a,original_control_tick=tick+a,phase=int(phases[a]),source='corrected_suffix',
            event_first8=bool(change[a:a+8].any()),event_tail_only=bool(not change[a:a+8].any() and change[a+8:a+40].any()),
            source_identity=members,cache_identity=encoder.cache_key('T1',INSTRUCTIONS['T1'],members)))
    report=dict(status='FAIL_SUPERVISION_CODEC_PRESERVE_RAW_STOP_BRANCH' if failures else 'PASS_FRESH_SUFFIX_STRICT_CODEC',
        slot=episode,rows=rows,failures=failures,maxima=maxima,restore=restore,teacher_sha256=card['teacher_sha256'],
        result_sha256=sha(folder/'RESULT.json'),processor=encoder.identity,takeover_tick=tick,end_tick=660,windows_cross_takeover_or_reset=0)
    immutable(root/'suffixes'/episode/'CODEC.json',report);return report


def bank(reports,teacher_sha256):
    expected=slots();by={};duplicates=[];duplicate_states=[];states=set();valid=[];codec_failure=False
    for r in reports:
        assert r['slot'] in {s['slot'] for s in expected}
        if r['slot'] in by:duplicates.append(r['slot']);continue
        by[r['slot']]=r
        if r['status']=='FAIL_SUPERVISION_CODEC_PRESERVE_RAW_STOP_BRANCH':codec_failure=True
        if r['status']!='PASS_FRESH_SUFFIX_STRICT_CODEC':continue
        assert r['teacher_sha256']==teacher_sha256 and r['processor']==FROZEN
        assert r['rows'] and all(v['episode']==r['slot'] and v['source']=='corrected_suffix' and v['instruction']==INSTRUCTIONS['T1'] for v in r['rows'])
        assert len({v['uid'] for v in r['rows']})==len(r['rows'])
        key=r['restore']['state_hash']
        if key in states:duplicate_states.append(r['slot']);continue
        states.add(key);valid.append(r)
    passed=len(valid)>=24 and not codec_failure and not duplicates
    return dict(status='PASS_ACTUAL_BRANCH_RESTORE_FRESH_CORRECTIONS_FIXED_CODEC' if passed else 'FAIL_RECOVERY_BANK_NO_DOSE_SWITCH',
        requested_branches=32,valid_branches=len(valid),valid_per_task=dict(T1=len(valid)),duplicates=duplicates,
        missing=[s['slot'] for s in expected if s['slot'] not in by],duplicate_states=duplicate_states,codec_failure_stops_branch=codec_failure,
        valid_slots=[r['slot'] for r in valid],rows=[v for r in valid for v in r['rows']],teacher_sha256=teacher_sha256,processor=FROZEN)


def correction_plan(rows,total,bucket_counts):
    """Exact equal-branch counts and unchanged bucket totals; fail infeasible coverage."""
    episodes=sorted({r['episode'] for r in rows});counts=quotas(total,[1]*len(episodes));groups={}
    for r in rows:groups.setdefault((r['episode'],r['phase']),[]).append(r)
    # ponytail: small integral flow (at most32 branches x5 buckets), no solver dependency.
    capacity={};adj={};flow={}
    def edge(a,b,n):
        capacity[a,b]=n;capacity[b,a]=0;flow[a,b]=flow[b,a]=0
        adj.setdefault(a,[]).append(b);adj.setdefault(b,[]).append(a)
    for ep,n in zip(episodes,counts):
        edge('start',ep,n)
        for p in range(5):
            if (ep,p) in groups:edge(ep,f'phase{p}',total)
    for p,n in enumerate(bucket_counts):edge(f'phase{p}','end',n)
    sent=0
    while sent<total:
        parents={'start':None};q=deque(['start'])
        while q and 'end' not in parents:
            a=q.popleft()
            for b in adj.get(a,[]):
                if b not in parents and flow[a,b]<capacity[a,b]:parents[b]=a;q.append(b)
        assert 'end' in parents,'Fixed equal-branch/bucket quotas infeasible; no fallback weighting authorized'
        b='end';amount=total-sent
        while parents[b] is not None:a=parents[b];amount=min(amount,capacity[a,b]-flow[a,b]);b=a
        b='end'
        while parents[b] is not None:a=parents[b];flow[a,b]+=amount;flow[b,a]-=amount;b=a
        sent+=amount
    streams={ep:iter(ordered([(p,j) for p in range(5) for j in range(flow.get((ep,f'phase{p}'),0))],ep)) for ep in episodes}
    visits=Counter();poolvisits=Counter();out=[]
    for j in range(max(counts)):
        for ep,n in zip(episodes,counts):
            if j>=n:continue
            p,_=next(streams[ep]);group=groups[ep,p];event=[r for r in group if r['event_first8']]
            priority=p!=4 and visits[ep,p]%2==0;pool=event if priority and event else group;pool=sorted(pool,key=lambda r:r['anchor'])
            key=(ep,p,bool(priority and event));ordinal=poolvisits[key];strata=min(8,len(pool));stratum=ordinal%strata
            eligible=pool[stratum::strata];chosen=eligible[(ordinal//strata)%len(eligible)]
            out.append(dict(chosen,sampling_event_priority=priority,event_fallback=bool(priority and not event),stratum=stratum))
            visits[ep,p]+=1;poolvisits[key]+=1
    assert len(out)==total and [sum(r['phase']==p for r in out) for p in range(5)]==list(bucket_counts)
    assert Counter(r['episode'] for r in out)==dict(zip(episodes,counts));return out


def recovery_plan(experts,admission):
    assert admission['status']=='PASS_ACTUAL_BRANCH_RESTORE_FRESH_CORRECTIONS_FIXED_CODEC'
    original=plan(experts,['T1'],1024);target=quotas(16384,WEIGHTS)
    correction=correction_plan(admission['rows'],8192,[n-sum(r['phase']==p for r in original) for p,n in enumerate(target)])
    return [dict(r,index=i) for i,r in enumerate(v for pair in zip(original,correction) for v in pair)]


def dose_plan(existing,decision):
    assert decision['branch']=='DOSE2048' and decision['DOSE_allowed'] and not decision['RECOVERY_priority']
    assert 16384<=len(existing)<=32768 and all(r['task']=='T1' and r['split']=='TRAIN' and r['source']=='expert' for r in existing)
    # Retain the fixed bucket/event proportions using only actual existing draw positions.
    positions=[]
    for phase,count in enumerate(quotas(16384,WEIGHTS)):
        for priority,n in zip((True,False),quotas(count,[1,1]) if phase!=4 else (0,count)):
            pool=[i for i,r in enumerate(existing) if r['phase']==phase and bool(r['sampling_event_priority'])==priority]
            assert len(pool)>=n,'Existing continuation lacks fixed behavior/event quota'
            positions.extend(pool[(j*len(pool))//n] for j in range(n))
    positions=sorted(positions);assert len(positions)==len(set(positions))==16384
    return [dict(existing[j],index=i,existing_plan_position=j) for i,j in enumerate(positions)]


def collection_window(decision,now,measured_remaining_reserve_s):
    assert decision['branch']=='RECOVERY2048' and decision['RECOVERY_priority']
    assert now<=1791496800,'NOT_LAUNCHED_TIME: latest start06:00 SGT'
    assert measured_remaining_reserve_s>=45*60,'Confirmation reserve floor'
    assert now+45*60+measured_remaining_reserve_s<1791510300,'Collection plus real training/evaluation reserve exceeds09:45'
    return now+45*60


def authorize_collection(measured_remaining_reserve_s):
    """Future owner-only start: actual frozen4096 proof, single durable45-minute clock."""
    from r2 import admission
    admission();assert not (ROOT/'qualification/CONFIRM_A_CANDIDATE.json').exists()
    path=ROOT/'qualification/R2_A_SINGLE_BRANCH.json';decision=json.loads(path.read_text())
    assert decision['status']=='FROZEN_R2_SINGLE_BRANCH_DECISION_NOT_DATA_OR_TRAINING_ADMISSION'
    assert all(sha(ROOT/name)==h for name,h in decision['proofs'].items())
    route=json.loads((ROOT/'qualification/PLACE4096-4096_ROUTE.json').read_text())
    assert route['next_action']=='SELECT_R2_SINGLE_BRANCH_FROM_FROZEN_EVIDENCE'
    now=time.time();clock=ROOT/'recovery/A/COLLECTION_CLOCK.json'
    if clock.exists():
        value=json.loads(clock.read_text());assert value['decision_sha256']==sha(path)
        assert now<value['deadline_epoch'],'Original45-minute collection clock expired; no reset'
    else:
        deadline=collection_window(decision,now,measured_remaining_reserve_s)
        value=dict(status='ACTUAL_RECOVERY_COLLECTION_WINDOW_STARTED',decision_sha256=sha(path),started_epoch=now,
            deadline_epoch=deadline,requested_slots=slots(),measured_remaining_reserve_s=measured_remaining_reserve_s)
        immutable(clock,value)
    return value
