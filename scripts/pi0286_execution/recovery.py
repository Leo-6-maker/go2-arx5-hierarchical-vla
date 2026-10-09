"""PI0286 final DOSE TRAIN-state corrections; import has no model/physics calls."""
import argparse,hashlib,json,os,time
from collections import Counter,deque
from pathlib import Path
import numpy as np
from contract import ROOT,INSTRUCTIONS,sha,write
from data import immutable,ordered,plan,quotas,WEIGHTS

TIMES=(3,12)
PARENT='PARENT_DOSE6144'
PANEL='LEARNER_TRAIN16'
PREFIX_KEYS=('initial.npz','observations.npz','commands.npz','installed.npz','physics.npz','EPISODE_CARD.json','QUERIES.json')
FROZEN={'statistics.json':'c3ec9c6b797bb735e3ac38d611b409262de04bbb965cb68b7426df749ba7fb81',
        'processor_config.json':'fae35db9d84083cc1faac16442c062c74a22691fa3358cfc58efb759e01977d3',
        'embodiment_id.json':'fb47934b4bd4b5e11ec1c61ade0c5e03ca9110ec5f16abce9633b10cfd9edfbb'}


def slots():
    return [dict(slot=f'T1-TRAIN-{seed}-TAKEOVER-{seconds:02d}s',task='T1',split='TRAIN',layout=seed,
                 task_time_s=seconds,control_tick=20+seconds*20,instruction=INSTRUCTIONS['T1'])
            for seed in range(285108,285124) for seconds in TIMES]


def array_hash(value):
    value=np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()


def prefix(folder,slot):
    """Bind the actual frozen DOSE learner result; never invent an unvisited state."""
    folder=Path(folder);result=json.loads((folder/'RESULT.json').read_text());card=json.loads((folder/'EPISODE_CARD.json').read_text())
    assert result['status'].startswith('VALID') and card['point']==PARENT
    assert all(card[k]==slot[k] for k in ('task','layout','instruction')) and card['split']==PANEL
    hashes={name:sha(folder/name) for name in PREFIX_KEYS}
    assert all(result['files'].get(name)==h for name,h in hashes.items()),'Partial or changed source prefix'
    tick=slot['control_tick']
    with np.load(folder/'installed.npz',allow_pickle=False) as f:actions=f['action'].copy()
    with np.load(folder/'commands.npz',allow_pickle=False) as f:commands=f['action'].copy()
    with np.load(folder/'observations.npz',allow_pickle=False) as f:
        keys=['time_s','state_v1','base_camera_raw','hand_camera_raw']+[k for k in f.files if 'extrinsic' in k or 'intrinsic' in k]
        observations={k:f[k].copy() for k in keys}
    with np.load(folder/'physics.npz',allow_pickle=False) as f:physics={k:f[k].copy() for k in f.files}
    assert len(actions)>=tick and len(commands)>=tick and len(physics['time_s'])>tick*5,'UNVISITED_STATE_COUNTS_IN_32'
    assert len(observations['time_s'])>tick,'MISSING_SAVED_BRANCH_RGB_COUNTS_IN_32'
    assert abs(float(observations['time_s'][tick])-tick*.05)<1e-8
    query_checks=[]
    for q in read(folder/'QUERIES.json'):
        issued=20+8*q['index']
        if issued>=tick:break
        query_folder=folder/f"q{q['index']:03d}"
        for name,h in [('input.npz',q['input_sha256']),('output.npz',q['output_sha256'])]:assert sha(query_folder/name)==h
        with np.load(query_folder/'input.npz',allow_pickle=False) as f:
            assert str(f['instruction'])==slot['instruction']
            for key,source_key in [('actual_state','state_v1'),('base_raw','base_camera_raw'),('wrist_raw','hand_camera_raw')]:
                assert np.array_equal(f[key],observations[source_key][issued]),'Saved actual learner query input mismatch'
        with np.load(query_folder/'output.npz',allow_pickle=False) as f:
            assert str(f['source_sha256'])==q['input_sha256'] and int(f['K'])==4
            executed=min(8,tick-issued);assert np.array_equal(f['decoded'][:executed],commands[issued:issued+executed]),'Actual learner first8 prefix mismatch'
        query_checks.append(dict(query_index=q['index'],original_tick=issued,executed_prefix_controls=executed,input_sha256=q['input_sha256'],output_sha256=q['output_sha256']))
    assert sum(q['executed_prefix_controls'] for q in query_checks)==tick-20
    with np.load(folder/'initial.npz',allow_pickle=False) as f:initial={k:f[k].copy() for k in f.files}
    return dict(folder=str(folder),hashes=hashes,actions=actions[:tick],commands=commands[:tick],physics=physics,
                observations=observations,initial=initial,slot=slot,learner_query_checks=query_checks,
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
        assert time.time()<deadline,'PI0286 correction collection deadline'
        ep.step(action,phase=-1,command=command,teacher=False)
        for n in range(i*5,(i+1)*5+1):
            assert set(ep.states[n])==set(bound['physics'])
            for k in ep.states[n]:match(ep.states[n][k],bound['physics'][k][n],f'{n}:{k}')
    observation=ep.observe();tick=ep.control
    for k in bound['observations']:
        if k!='time_s':match(observation[k],bound['observations'][k][tick],k,0)
    return dict(status='PASS_REPLAY_ALL_PHYSICS_SAMPLES_AND_EXACT_BRANCH_RGB_STATE',slot=bound['slot']['slot'],
                source_hashes=bound['hashes'],takeover_tick=tick,state_hash=bound['state_hash'],tolerance=1e-6,RGB_state_byte_exact=True,
                learner_query_checks=bound['learner_query_checks'],actual_first8_prefix_exact=True)


def collect_episode(ep,bound,teacher_sha256,deadline):
    """Same teacher, fresh real observations through original task32s; no model calls."""
    from teacher import Teacher
    from source import step
    from gate import original_gate
    from scorer import score
    clock=json.loads((ROOT/'correction/COLLECTION_CLOCK.json').read_text())
    assert deadline==clock['deadline_epoch'] and time.time()<deadline
    panel_path=ROOT/'source/evidence'/f'{PARENT}_{PANEL}.json'
    assert clock['learner_panel_sha256']==sha(panel_path)
    panel=json.loads(panel_path.read_text())
    original=next(row for row in panel['rows'] if row['layout']==bound['slot']['layout'] and row['task']=='T1')
    assert Path(original['folder']).resolve()==Path(bound['folder']).resolve()
    assert original==json.loads((Path(bound['folder'])/'RESULT.json').read_text())
    assert teacher_sha256==sha(ROOT/'code/teacher.py')
    restore=replay(ep,bound,deadline);slot=bound['slot'];start=ep.control
    write(ep.folder/'RECOVERY_RESTORE.json',restore)
    teacher=Teacher(ep,slot['task'],ep.env.layout_spec['zone_xy'])
    # Protocol explicitly requires existing withdrawal/hold behavior at a placed state.
    # query(), speed limits, stage transitions, grasp planner and source remain unchanged.
    from scorer import predicates
    physical={k:np.stack([v[k] for v in ep.states]) for k in ep.states[0]}
    masks=predicates(physical,ep.env.layout_spec['zone_xy'],ep.states[0]['cubeA_pose'][:3],ep.states[0]['cubeB_pose'][:3])
    placed=all(bool(masks[k][-1]) for k in ('zone','table','static','not_grasped'))
    initialization='ORIGINAL_APPROACH'
    if placed:
        teacher.goal=ep.states[-1]['tcp'].copy();teacher.terminal=teacher.goal.copy()
        if bool(masks['TCP_clear'][-1]):
            teacher.stage='hold';initialization='EXISTING_HOLD_ALREADY_CLEAR'
        else:
            teacher.terminal[2,3]+=.10;teacher.stage='withdraw';initialization='EXISTING_WITHDRAW_PLACED'
    write(ep.folder/'TEACHER_INITIALIZATION.json',dict(status='PROTOCOL_DECLARED_EXISTING_STAGE_INITIALIZATION',mode=initialization,
        placed_predicates={k:bool(masks[k][-1]) for k in ('zone','table','static','not_grasped','TCP_clear')},teacher_code_changed=False))
    status='VALID_SOURCE_COMPLETE';gates=[]
    while ep.control<660:
        from budget import time_allowed
        if ep.control%8==0:time_allowed(ROOT)
        assert time.time()<deadline,'PI0286 correction collection deadline'
        observation=ep.observe();command,phase=teacher.query((ep.control-start)*.05)
        gate=original_gate(np.tile(command,(40,1)),observation['state_v1'],ep.control*.05,slot['slot']+f'-{ep.control}')
        gates.append(dict(control=ep.control,decision=gate['decision'],issues=gate.get('issues',[])))
        if gate['decision']=='REJECT':status='VALID_SOURCE_GATE_REJECT';break
        step(ep,command,phase,True)
    physical={k:np.stack([v[k] for v in ep.states]) for k in ep.states[0]}
    scoring=score(physical,slot['task'],ep.env.layout_spec['zone_xy'],ep.states[0]['cubeA_pose'][:3],ep.states[0]['cubeB_pose'][:3],status,ep.control)
    write(ep.folder/'TASK_SCORE_CPU.json',scoring);write(ep.folder/'GATE_RECORDS.json',gates)
    write(ep.folder/'TEACHER_EVENTS.json',dict(events=teacher.events,teacher_time_origin_control=start,within_episode_recovery_attempts=teacher.attempts,teacher_failed=teacher.failed))
    write(ep.folder/'EPISODE_CARD.json',dict(**slot,point=PARENT,teacher_sha256=teacher_sha256,
        restore=restore,takeover_tick=start,task_end_tick=660,source='corrected_suffix',truth_offline_only=True,
        teacher_initialization=initialization,scene=ep.env.layout_spec,reset_controls=[],code_seal_sha256=sha(ROOT/'design/CODE_SEAL.json')))
    result=ep.finish(status);result.update(task=slot['task'],split='TRAIN',layout=slot['layout'],instruction=slot['instruction'],
        folder=str(ep.folder),slot=slot['slot'],persistent_success=scoring['persistent_success'],task_score=scoring,
        source='corrected_suffix',model_queries=0,teacher_sha256=teacher_sha256,takeover_tick=start)
    write(ep.folder/'RESULT.json',result);return result


def suffix(root,folder,encoder,physical_errors):
    """Keep raw labels even on codec FAIL; return every real eligible H40 window."""
    root=Path(root);folder=Path(folder);result=json.loads((folder/'RESULT.json').read_text());card=json.loads((folder/'EPISODE_CARD.json').read_text())
    assert all((root/name).resolve().is_relative_to(ROOT.resolve()) for name in ('policy_observation','action_target')),'Correction writes require owned PI0286 namespaces'
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
            err['native_preclip_out_of_range_count']=encoder.action_audit(target,state)['native_preclip_out_of_range_count']
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
    missing=[s['slot'] for s in expected if s['slot'] not in by]
    passed=len(valid)>=24 and not duplicates and not missing and not duplicate_states
    return dict(status='PASS_ACTUAL_BRANCH_RESTORE_FRESH_CORRECTIONS_FIXED_CODEC' if passed else 'CORRECTION_NOT_ADMITTED',
        requested_branches=32,valid_branches=len(valid),valid_per_task=dict(T1=len(valid)),duplicates=duplicates,
        missing=missing,duplicate_states=duplicate_states,codec_failure_stops_branch=codec_failure,
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
    assert all(r['task']=='T1' and r['split']=='TRAIN' and 285100<=r['layout']<=285163 and r['source']=='expert' for r in experts)
    original=plan(experts,['T1'],512)
    correction=correction_plan(admission['rows'],4096,quotas(4096,WEIGHTS))
    return [dict(r,index=i) for i,r in enumerate(v for pair in zip(original,correction) for v in pair)]


def read(path):return json.loads(Path(path).read_text())


def collect(deadline=None):
    """Owner-only finite32-slot teacher collection after actual16-parent rollouts."""
    import fcntl
    from datetime import datetime
    from contract import guard
    from budget import process,time_allowed
    guard();assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
    panel_path=ROOT/'source/evidence'/f'{PARENT}_{PANEL}.json';panel=read(panel_path)
    assert len(panel['rows'])==16 and {r['layout'] for r in panel['rows']}==set(range(285108,285124))
    assert all(r['task']=='T1' and r['point']==PARENT and r['split']==PANEL for r in panel['rows'])
    audit=read(ROOT/'audit'/f'{PARENT}_{PANEL}_CPU.json')
    assert audit['status'].startswith('PASS_INDEPENDENT_CPU_') and audit['source_sha256']==sha(panel_path)
    now=time.time();latest=datetime.fromisoformat('2026-10-09T11:10:00+08:00').timestamp()
    stop=datetime.fromisoformat('2026-10-09T11:25:00+08:00').timestamp()
    clock_path=ROOT/'correction/COLLECTION_CLOCK.json'
    if clock_path.exists():clock=read(clock_path);assert clock['learner_panel_sha256']==sha(panel_path)
    else:
        assert now<=latest,'NOT_LAUNCHED_TIME_CORRECTIONS_AFTER_1110'
        deadline=min(now+40*60,stop) if deadline is None else deadline
        assert now<deadline<=stop
        clock=dict(status='ACTUAL_PI0286_CORRECTION_COLLECTION_STARTED',started_epoch=now,deadline_epoch=deadline,
            learner_panel_sha256=sha(panel_path),learner_CPU_sha256=sha(ROOT/'audit'/f'{PARENT}_{PANEL}_CPU.json'),
            slots=slots(),physical_passes_per_slot=1,quality_retries=0,maximum_authorized_passes=2)
        immutable(clock_path,clock)
    deadline=clock['deadline_epoch'];teacher_sha256=sha(ROOT/'code/teacher.py')
    import recorder,scene
    reports=[];actual=[]
    lock=(ROOT/'runtime/SOURCE_GPU0.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    began=process(ROOT,'start')
    try:
        for slot in slots():
            receipt=ROOT/'correction/slots'/slot['slot']/'COLLECTION.json'
            if receipt.exists():
                r=read(receipt);reports.append(r)
                if r.get('result'):actual.append(r['result'])
                continue
            ep=None
            try:
                assert time.time()<deadline,'COLLECTION_DEADLINE_NO_NEW_START'
                time_allowed(ROOT,30)
                row=next(r for r in panel['rows'] if r['layout']==slot['layout'])
                bound=prefix(row['folder'],slot)
                parent=recorder.RAW/'episodes'/slot['slot']
                assert not list(parent.glob('attempt-*')),'Preserve original attempt; owner audit required before technical restart'
                ep=recorder.Episode(slot['slot'],slot['layout'],'pd_ee_pose','PI0285PickPlaceStack-v1',dict(layout_id=slot['layout']))
                result=collect_episode(ep,bound,teacher_sha256,deadline)
                r=dict(slot=slot['slot'],status='ACTUAL_CORRECTION_PHYSICAL_CLOSED',result=result,
                    takeover_tick=slot['control_tick'],teacher_sha256=teacher_sha256,physical_passes=1)
                actual.append(result)
            except (AssertionError,ValueError,FloatingPointError) as exc:
                result=ep.finish('ENGINEERING_INCOMPLETE',repr(exc)) if ep is not None else None
                r=dict(slot=slot['slot'],status='NOT_ADMITTED_ORIGINAL_ATTEMPT_PRESERVED',error=repr(exc),
                    result=result,folder=str(ep.folder) if ep is not None else None,physical_passes=int(ep is not None))
            immutable(receipt,r);reports.append(r)
            write(ROOT/'correction/COLLECTION_PROGRESS.json',dict(requested=32,closed=len(reports),slots=reports))
        value=dict(status='ALL32_REQUESTED_CORRECTIONS_CLOSED_NOT_CODEC_ADMISSION',slots=reports,rows=actual,
            requested=32,teacher_sha256=teacher_sha256,learner_panel_sha256=sha(panel_path),new_model_queries=0)
        immutable(ROOT/'source/evidence/CORRECTIONS32.json',value)
        return value
    finally:
        process(ROOT,'end',began);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def audit_correction(folder):
    """Saved-array independent contact, clock, StateV1, gripper and task readback."""
    from audit_source import grasp_from_contacts
    from scorer import score
    folder=Path(folder);result=read(folder/'RESULT.json');card=read(folder/'EPISODE_CARD.json')
    for name,h in result['files'].items():assert sha(folder/name)==h,name
    with np.load(folder/'physics.npz',allow_pickle=False) as f:d={k:f[k].copy() for k in f.files}
    assert np.allclose(d['time_s'],np.arange(len(d['time_s']))*.01,rtol=0,atol=1e-8)
    assert np.array_equal(grasp_from_contacts(d),d['is_cubeA_grasped'])
    assert np.array_equal((np.linalg.norm(d['cubeA_velocity'],axis=1)<=.01)&(np.linalg.norm(d['cubeA_angular'],axis=1)<=.5),d['is_cubeA_static'])
    scored=score(d,'T1',card['scene']['zone_xy'],d['cubeA_pose'][0,:3],d['cubeB_pose'][0,:3],result['status'],result['control_calls'])
    assert scored==result['task_score']
    with np.load(folder/'observations.npz',allow_pickle=False) as f:s=f['state_v1'].copy();t=f['time_s'].copy()
    with np.load(folder/'commands.npz',allow_pickle=False) as f:c=f['action'].copy()
    with np.load(folder/'installed.npz',allow_pickle=False) as f:a=f['action'].copy()
    assert np.allclose(t,np.arange(len(t))*.05,rtol=0,atol=1e-8)
    pre=d['tcp'][::5][:len(s)];q=d['qpos'][::5][:len(s)]
    rebuilt=np.c_[pre[:,:3,3],pre[:,:2,:3].reshape(-1,6),(q[:,-2:].sum(1)-.00004)/.08]
    state_error=float(np.max(abs(rebuilt-s)));assert state_error<=1e-6
    polarity=float(np.max(abs((a[:,-1]+1)/2-np.clip(c[:,-1],0,1))));assert polarity<=1e-7
    return dict(status='PASS_INDEPENDENT_CPU_CORRECTION_CONTACT_CLOCK_STATE_GRIPPER_SCORE',source_result_sha256=sha(folder/'RESULT.json'),
        score=scored,state_max_abs=state_error,gripper_polarity_max_abs=polarity,model_queries=0,physics_calls=0)


def assemble(data_root):
    """CPU-only strict branch admission; all failed raw attempts remain excluded."""
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    import torch
    torch.set_num_threads(4)
    from data import codec
    from cpu_inputs import physical_errors
    data_root=Path(data_root);collection=read(ROOT/'source/evidence/CORRECTIONS32.json')
    assert collection['requested']==len(collection['slots'])==32
    encoder=codec(str(data_root));assert encoder.identity==FROZEN
    reports=[];cpu=[]
    for entry in collection['slots']:
        r=dict(slot=entry['slot'],status='NOT_ADMITTED_ORIGINAL_ATTEMPT_PRESERVED',collection=entry)
        result=entry.get('result')
        if result and result.get('status')=='VALID_SOURCE_COMPLETE' and result.get('persistent_success'):
            folder=result['folder'];audit=audit_correction(folder);cpu.append(dict(slot=entry['slot'],**audit))
            assert audit['score']['persistent_success']
            r=suffix(data_root,folder,encoder,physical_errors)
        elif result and result.get('task_score'):
            cpu.append(dict(slot=entry['slot'],**audit_correction(result['folder'])))
        reports.append(r)
    admission=bank(reports,collection['teacher_sha256'])
    admission.update(collection_sha256=sha(ROOT/'source/evidence/CORRECTIONS32.json'),CPU_readbacks=cpu,
        all_requested_slot_reports=reports,failed_prefix_labels=0,statistics_refit=0,new_model_queries=0,new_physics=0,
        parent_point='PI0285_REPAIR_A2048-2048',learner_point=PARENT,layouts=list(range(285108,285124)),branch_times_seconds=[3,12])
    immutable(ROOT/'correction/CORRECTION_BANK.json',admission)
    if admission['status'].startswith('PASS_'):
        experts=read(data_root/'WINDOW_INDEX.json')['rows']
        experts=[r for r in experts if r['task']=='T1' and r['split']=='TRAIN']
        try:
            rows=recovery_plan(experts,admission)
            payload=''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows)
            path=data_root/'C1_CORRECT1024.jsonl'
            if path.exists():assert path.read_text()==payload
            else:path.write_text(payload)
            audit=dict(status='PREPARED_C1_EQUAL_EXPERT_CORRECTION_PLAN_NOT_TRAINING_ADMISSION',rows=len(rows),
                source_counts=dict(Counter(r['source'] for r in rows)),phase_counts=dict(Counter(r['phase'] for r in rows)),
                plan_sha256=sha(path),correction_bank_sha256=sha(ROOT/'correction/CORRECTION_BANK.json'),optimizer_updates=1024,
                source_branch_weights='EQUAL_WITHIN_SOURCE_FIXED_BEHAVIOR_BUCKETS',failed_prefix_labels=0)
        except AssertionError as exc:
            phases={ep:{str(p):sum(r['episode']==ep and r['phase']==p for r in admission['rows']) for p in range(5)} for ep in admission['valid_slots']}
            audit=dict(status='CORRECTION_SAMPLER_NOT_ADMITTED_FIXED_BUCKETS_INFEASIBLE',error=repr(exc),optimizer_launch_allowed=False,
                required_phase_draw_counts=quotas(4096,WEIGHTS),equal_branch_draw_counts=dict(zip(sorted(phases),quotas(4096,[1]*len(phases)))),
                available_H40_windows_per_branch_phase=phases,branch_renormalization=0,bucket_changes=0)
        immutable(ROOT/'correction/C1_PLAN_AUDIT.json',audit)
    else:
        immutable(ROOT/'correction/C1_PLAN_AUDIT.json',dict(status='CORRECTION_NOT_ADMITTED',valid_branches=admission['valid_branches'],
            required=24,closed=len(reports),missing=admission['missing'],optimizer_launch_allowed=False))
    return admission


def self_check():
    expected=slots();assert len(expected)==len({s['slot'] for s in expected})==32
    assert {s['layout'] for s in expected}==set(range(285108,285124))
    assert {s['control_tick'] for s in expected}=={80,260}
    rows=[dict(uid=f"{s['slot']}-{p}-{a}",episode=s['slot'],phase=p,anchor=a,event_first8=a==0,task='T1',split='TRAIN',layout=s['layout'],
        source='corrected_suffix',instruction=INSTRUCTIONS['T1']) for s in expected[:24] for p in range(5) for a in range(2)]
    reports=[dict(slot=s['slot'],status='PASS_FRESH_SUFFIX_STRICT_CODEC',teacher_sha256='fixture',processor=FROZEN,
        restore={'state_hash':str(i)},rows=[r for r in rows if r['episode']==s['slot']]) if i<24 else
        dict(slot=s['slot'],status='FAIL_SUPERVISION_CODEC_PRESERVE_RAW_STOP_BRANCH') for i,s in enumerate(expected)]
    admitted=bank(reports,'fixture');assert admitted['status'].startswith('PASS_') and admitted['valid_branches']==24
    assert not bank(reports[:-1],'fixture')['status'].startswith('PASS_')
    assert not bank(reports+reports[:1],'fixture')['status'].startswith('PASS_')
    result=correction_plan(rows,4096,quotas(4096,WEIGHTS));assert len(result)==4096
    counts=Counter(r['episode'] for r in result);assert max(counts.values())-min(counts.values())<=1
    assert [sum(r['phase']==p for r in result) for p in range(5)]==quotas(4096,WEIGHTS)
    try:correction_plan([dict(r,phase=4) for r in rows],4096,quotas(4096,WEIGHTS))
    except AssertionError:pass
    else:raise AssertionError('Infeasible behavior coverage silently repaired')
    return dict(status='PASS_CPU_FIXED_SLOTS_CLOSED_BANK_EQUAL_BRANCH_FIXED_BUCKET_FIXTURES',requested=32,ticks=[80,260],
        codec_failure_excludes_branch_only=True,missing_or_duplicate_slot_blocks=True,equal_branch_weight=True,
        infeasible_bucket_fails=True,model_calls=0,physics_calls=0,teacher_calls=0,actual_restore='UNOBSERVED',actual_codec='UNOBSERVED')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('collect','assemble','slots','cpu-check'))
    p.add_argument('--data-root',default=str(ROOT/'data'));p.add_argument('--deadline',type=float)
    a=p.parse_args()
    value=collect(a.deadline) if a.action=='collect' else assemble(a.data_root) if a.action=='assemble' else self_check() if a.action=='cpu-check' else slots()
    print(json.dumps(value if a.action=='slots' else {k:v for k,v in value.items() if k not in ('rows','slots','all_requested_slot_reports','CPU_readbacks')}))

