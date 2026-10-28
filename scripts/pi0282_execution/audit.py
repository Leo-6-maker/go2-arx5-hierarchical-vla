"""PI0282 finite CPU evidence audit; no model/controller/physics imports."""
import argparse,hashlib,json,os,re,time
from pathlib import Path

ROOT=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0282')
BASE=ROOT.parent
SEEDS=set(range(30032,30048))
TOKEN=re.compile(r'(?<![A-Za-z0-9.])(3003[2-9]|3004[0-7])(?![A-Za-z0-9.])')
SEEDKEY=re.compile(r'(seed|condition|episode|slot|trial|case|uid|request|query|path|folder|id)',re.I)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def identity(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))

def save(folder,name,value):
    p=ROOT/folder/name;p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        h=p.parent/'history';h.mkdir(exist_ok=True);(h/(p.stem+'-'+sha(p)+p.suffix)).write_bytes(p.read_bytes())
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)
    return p

def seed_mentions(value,key='',location='$'):
    out=[]
    if isinstance(value,dict):
        for k,v in value.items():out.extend(seed_mentions(v,k,location+'.'+k))
    elif isinstance(value,list):
        for i,v in enumerate(value):out.extend(seed_mentions(v,key,location+f'[{i}]'))
    elif SEEDKEY.search(key):
        if isinstance(value,int) and not isinstance(value,bool) and value in SEEDS:out.append(dict(seed=value,field=location,value=value))
        elif isinstance(value,str):
            for m in TOKEN.finditer(value):out.append(dict(seed=int(m[1]),field=location,value=value[:600]))
    return out

def exposure():
    """Narrow metadata/journal/RESULT index inventory, excludes payloads and model trees."""
    roots=[]
    for n in range(276,283):
        r=BASE/f'PI0{n}'
        roots += [r/p for p in ('evidence','design','source','runtime') if (r/p).is_dir()]
        for phase in ('R2','R3'):
            roots += [r/phase/p for p in ('evidence','design','contracts','runtime') if (r/phase/p).is_dir()]
        r=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5')/f'PI0{n}'
        roots += [r/p for p in ('evidence','episodes','quality','source','fixed_inputs') if (r/p).is_dir()]
        for phase in ('R2','R3'):
            roots += [r/phase/p for p in ('evidence','episodes','quality') if (r/phase/p).is_dir()]
    skipped_names={'code','official','cache','hf_home','venv','ManiSkill','project-src','training','checkpoints','model_bindings','archive','audit','videos','assets'}
    files=[];matches=[];directories=[];errors=[]
    for root in roots:
        for path,dirs,names in os.walk(root):
            dirs[:]=[d for d in dirs if d not in skipped_names and not d.startswith('q0') and d!='history']
            for d in dirs:
                if TOKEN.search(d):directories.append(str(Path(path)/d))
            for name in names:
                p=Path(path)/name
                if p.suffix not in ('.json','.jsonl'):continue
                if name in ('rows.json','contacts.jsonl','actor_inputs.jsonl','actor_outputs.jsonl','GATES.json','TIMINGS.jsonl'):continue
                if p.stat().st_size>256<<20:errors.append(dict(path=str(p),reason='METADATA_FILE_EXCEEDS_256MiB'));continue
                record=identity(p);record['parsed_rows']=0;files.append(record)
                try:
                    values=(json.loads(line) for line in p.open() if line.strip()) if p.suffix=='.jsonl' else [json.loads(p.read_text())]
                    for row,value in enumerate(values):
                        record['parsed_rows']+=1
                        found=seed_mentions(value)
                        if found:
                            actual=(name in ('RESULT.json','QUERIES.json','START.json','PARENT.json') or any(x in name.lower() for x in ('model_attempt','model_head','model_budget','budget.jsonl')))
                            matches.append(dict(path=str(p),row=row,classification='EXECUTION_RECEIPT_OR_JOURNAL' if actual else 'CARD_OR_INDEX_METADATA_REQUIRES_CONTEXT',mentions=found))
                except (OSError,ValueError) as e:errors.append(dict(path=str(p),reason=str(e)))
    execution=[r for r in matches if r['classification']=='EXECUTION_RECEIPT_OR_JOURNAL']
    status='EXPOSURE_FOUND_PRESERVE_SEEDS' if execution or directories else ('NO_PRIOR_QUERY_FOUND_IN_SCOPED_RECEIPTS' if not errors else 'UNRESOLVED_INVENTORY_ERRORS')
    v=dict(schema='PI0282_CONFIRM_EXPOSURE_READBACK_V1',status=status,seeds=sorted(SEEDS),observed_unix_s=time.time(),roots=[str(p) for p in roots],
        confirm_allowed=status=='NO_PRIOR_QUERY_FOUND_IN_SCOPED_RECEIPTS',
        confirm_allowed_semantics='Exposure prerequisite only in the explicitly listed retained Corelab roots; selection and five-part candidate freeze must independently pass. This receipt does not unlock execution by itself.',
        files=files,matches=matches,seed_directories=directories,errors=errors,
        scope='Existing NFS PI0276..PI0282 source/evidence/design/runtime plus R2/R3 metadata; actual NVMe PI0276..PI0279 evidence/episodes/quality/source/fixed_inputs and R2/R3; semantic seed/id/path fields only. No model/array payload scan.',
        limitations='Absence in these retained receipts is not a proof about unrecorded execution or unrelated external machines; owner must bind historical source locations.',
        no_seed_replacement=True,new_model_calls=0,new_physics_steps=0)
    save('audit','CONFIRM_EXPOSURE.json',v)
    print(json.dumps(dict(status=status,files=len(files),execution_hits=len(execution),metadata_hits=len(matches)-len(execution),directory_hits=directories,errors=errors)))

def fixed_rules():
    v=dict(schema='PI0282_FIXED_AUDIT_AND_SELECTION_V1',development=dict(eligible_endpoints=['INTERFACE2048','ACTION2048'],TRAIN_denominator=4,TRAIN_min=3,DEV_denominator=8,DEV_min=6),
        ordering=['DEV persistent count descending','TRAIN persistent count descending','INTERFACE2048 preferred'],
        confirmation=dict(seeds=sorted(SEEDS),denominator=16,min_persistent=12,one_frozen_candidate_only=True,previously_unqueried_required=True,failed_confirm_no_second_candidate=True),
        metric='Full valid32s and unchanged on_top & static & not_grasped at every41 control samples28..30s; early valid failures retained in denominator, missing suffix never fabricated',
        identity_freeze=['model','processor','statistics','mapping','decode'],TRAIN_statistics_only=True,retention_not_selection=True,
        expected_rows=dict(index=10,scalars=10180740,slices=14,packed_scalars=325783680),training=dict(updates_each=2048,chunks_each=16384,phase_chunks_each=4096,effective_batch=8,visits_per_episode_phase=[44,45],episodes=93),
        required_actual_receipts=['original row10 slice hashes','frozen-row byte hashes','optimizer moments/grad masks/zero projector decay','actual sample cursor/order','actual full132 noise/time hash per forward','same initial/model snapshot both arms','canary fresh-process resume parity','TRAIN codec physical errors','cache actual parity'],
        completed_receipt_audits=[],source_candidate='NONE_UNTIL_FIXED_GATES',selected='NONE',deployment='NONE',new_model_calls=0,new_physics_steps=0)
    save('audit','FIXED_RULES.json',v);print('FIXED_RULES_READY_NOT_EXECUTED')

def plan_audit():
    import collections
    cardpath=BASE/'PI0276/design/DATASET_CARD.json';card=json.loads(cardpath.read_text())
    groups=collections.defaultdict(list)
    for r in card['rows']:
        if r['eligible'] and r['split']=='TRAIN':groups[(r['seed'],r['group'])].append(r)
    assert len(groups)==372 and len({s for s,p in groups})==93
    path=ROOT/'data/PLAN16384.jsonl';plan=[json.loads(x) for x in path.read_text().splitlines()]
    diagpath=ROOT/'data/DIAG48.jsonl';diag=[json.loads(x) for x in diagpath.read_text().splitlines()]
    visits=collections.Counter();phase=collections.Counter();observed=collections.defaultdict(list)
    for i,r in enumerate(plan):
        assert r['index']==i and r['split']=='TRAIN' and r['eligible'] and r['phase']==r['group'];visits[(r['seed'],r['phase'])]+=1;phase[r['phase']]+=1;observed[(r['seed'],r['phase'])].append(r)
    details=[]
    for key,rows in groups.items():
        rows.sort(key=lambda r:(r['anchor'],r['uid']));got=observed[key];assert len(got) in (44,45)
        assert got[0]['uid']==rows[0]['uid'] and got[1]['uid']==rows[-1]['uid']
        n=min(8,len(rows));small,extra=divmod(len(rows),n);strata=[];start=0
        for s in range(n):
            end=start+small+(s<extra);strata.append(rows[start:end]);start=end
        orders=[sorted(g,key=lambda r:hashlib.sha256(('PI0282-28203-'+r['uid']).encode()).hexdigest()) for g in strata]
        for j,r in enumerate(got[2:]):
            sid=j%n;wanted=orders[sid][(j//n)%len(orders[sid])];assert r['uid']==wanted['uid'] and r['stratum']==sid
        expected_uids={r['uid'] for r in rows};assert all(r['uid'] in expected_uids for r in got)
        assert {r['stratum'] for r in got}==set(range(n))
        details.append(dict(seed=key[0],phase=key[1],visits=len(got),eligible=len(rows),strata=n,unique_visited=len({r['uid'] for r in got}),first_last_exposed=True))
    assert len(plan)==16384 and phase=={0:4096,1:4096,2:4096,3:4096}
    assert len(diag)==48 and collections.Counter(r['split'] for r in diag)=={'TRAIN':24,'VAL':24}
    assert {r['seed'] for r in diag if r['split']=='TRAIN'}==set(range(10000,10004))
    v=dict(status='PASS_FIXED_PLAN_MEMBERSHIP_ORDER_AND_TIME_STRATA_CPU',inputs=[identity(cardpath),identity(path),identity(diagpath),identity(ROOT/'data/PLAN_AUDIT.json')],rows=len(plan),phase_counts=dict(phase),groups=details,
        min_visits=min(visits.values()),max_visits=max(visits.values()),unique_windows=len({r['uid'] for r in plan}),DIAG48=dict(TRAIN=24,VAL=24),
        semantic_limit='Metadata plan covers time strata, not all windows or independent demonstrations; no learned scores used.',new_model_calls=0,new_physics_steps=0)
    save('audit','PLAN_CPU_RECOMPUTE.json',v);print(json.dumps(dict(status=v['status'],rows=len(plan),groups=len(groups),unique_windows=v['unique_windows'])))

def expert_audit():
    import collections,sys
    import numpy as np
    from scipy.spatial.transform import Rotation
    panel=ROOT/'source/evidence/RESULT_EXPERT_TRAIN.json'
    if not panel.exists():
        print('WAIT_EXISTING_EXPERT_PANEL_NOT_CLOSED');return
    panelvalue=json.loads(panel.read_text());assert len(panelvalue['rows'])==4
    sys.path.insert(0,str(BASE/'PI0280/official'))
    from gr00t.data.state_action.state_action_processor import StateActionProcessor
    sys.path.insert(0,str(ROOT/'code'))
    from codec import modality,fields
    cfg=json.loads((ROOT/'data/processor_config.json').read_text())['processor_kwargs'];tag='new_embodiment'
    statistics=json.loads((ROOT/'data/statistics.json').read_text())
    proc=StateActionProcessor({tag:modality()},statistics={tag:statistics[tag]},use_percentiles=cfg['use_percentiles'],clip_outliers=cfg['clip_outliers'],use_relative_action=True)
    sys.path.insert(0,str(BASE/'PI0250/runtime/reference-317a/src'))
    from go2arx5_vla.contracts import TrajectoryChunkV1
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix,matrix_to_quat
    gate=TrajectorySafetyGateV1();cardpath=ROOT/'design/SOURCE_CARD.json';card=json.loads(cardpath.read_text());rows=[]
    for seed in range(10000,10004):
        d=ROOT/f'source/episodes/EXPERT-TRAIN-{seed}/attempt-000';result=json.loads((d/'RESULT.json').read_text());parent=Path(card['expert_parents'][str(seed)]['path'])
        for name,h in card['expert_parents'][str(seed)]['files'].items():assert sha(parent/name)==h
        for name,h in result['files'].items():assert sha(d/name)==h
        with np.load(parent/'commands.npz',allow_pickle=False) as f:reference=f['action']
        with np.load(parent/'initial.npz',allow_pickle=False) as a,np.load(d/'initial.npz',allow_pickle=False) as b:assert a.files==b.files and all(np.max(abs(a[k]-b[k]))<=1e-6 for k in a.files)
        with np.load(d/'physics.npz',allow_pickle=False) as f:p={k:f[k] for k in f.files}
        with np.load(d/'commands.npz',allow_pickle=False) as f:commands=f['action']
        with np.load(d/'installed.npz',allow_pickle=False) as f:installed={k:f[k] for k in f.files}
        with np.load(d/'observations.npz',allow_pickle=False) as f:obs={k:f[k] for k in f.files}
        assert result['control_calls']==660 and result['task_control_calls']==640 and len(p['success'])==3301 and result['physics_completed']==3300
        assert np.allclose(p['time_s'],np.arange(3301)*.01,atol=1e-6)
        predicate=p['is_cubeA_on_cubeB'] & p['is_cubeA_static'] & ~p['is_cubeA_grasped'];assert np.array_equal(predicate,p['success'])
        window=np.arange(2900,3101,5);persistent=bool(predicate[window].all());assert persistent==result['persistent_success']
        queries=json.loads((d/'QUERIES.json').read_text());assert len(queries)==80
        maxerr=dict(encoded=0.,decoded=0.,position_m=0.,rotation_rad=0.,gripper=0.,requested_to_command=0.,installed_pose=0.,actual_state=0.,gripper_actuator=0.)
        gates=collections.Counter();identities=[];clips=0
        for index,q in enumerate(queries):
            tick=20+index*8;folder=d/f'q{index:03d}';assert q['index']==index and abs(q['time_s']-index*.4)<1e-9
            assert sha(folder/'input.npz')==q['input_sha256'] and sha(folder/'output.npz')==q['output_sha256']
            with np.load(folder/'input.npz',allow_pickle=False) as f:a={k:f[k] for k in f.files}
            with np.load(folder/'output.npz',allow_pickle=False) as f:b={k:f[k] for k in f.files}
            assert str(b['source_sha256'])==q['input_sha256']
            codec_identity=json.loads(str(b['identity_json']))
            assert codec_identity=={name:sha(ROOT/'data'/name) for name in ('processor_config.json','statistics.json','embodiment_id.json')}
            wanted=reference[np.minimum(np.arange(index*8,index*8+40),639)];assert np.array_equal(a['commands'],wanted) and np.array_equal(b['commands'],wanted)
            assert np.array_equal(a['base_raw'],obs['base_camera_raw'][tick]) and np.array_equal(a['wrist_raw'],obs['hand_camera_raw'][tick])
            assert np.array_equal(a['actual_state'],obs['state_v1'][tick]) and np.array_equal(a['actual_state'],b['actual_state'])
            actual_pose=p['tcp'][tick*5];state=np.r_[actual_pose[:3,3],actual_pose[:2,:3].reshape(6),(p['qpos'][tick*5,-2:].sum()-.00004)/.08]
            maxerr['actual_state']=max(maxerr['actual_state'],float(np.max(abs(a['actual_state']-state))))
            encoded=proc.apply_action(fields(wanted.astype(np.float32)),tag,fields(a['actual_state'].reshape(1,10)));normalized=np.zeros((40,132),np.float32);normalized[:,:10]=np.c_[encoded['eef_9d'],encoded['gripper_position']]
            decodedfields=proc.unapply_action(fields(normalized[:,:10]),tag,fields(a['actual_state'].reshape(1,10)));decoded=np.c_[decodedfields['eef_9d'],decodedfields['gripper_position']].astype(np.float32)
            maxerr['encoded']=max(maxerr['encoded'],float(np.max(abs(normalized-b['normalized']))));maxerr['decoded']=max(maxerr['decoded'],float(np.max(abs(decoded-b['decoded']))))
            assert np.array_equal(normalized,b['normalized']) and np.array_equal(decoded,b['decoded'])
            maxerr['position_m']=max(maxerr['position_m'],float(np.linalg.norm(decoded[:,:3]-wanted[:,:3],axis=1).max()))
            maxerr['rotation_rad']=max(maxerr['rotation_rad'],float((Rotation.from_matrix(rot6d_rows_to_matrix(decoded[:,3:9]))*Rotation.from_matrix(rot6d_rows_to_matrix(wanted[:,3:9])).inv()).magnitude().max()))
            maxerr['gripper']=max(maxerr['gripper'],float(np.max(abs(decoded[:,-1]-wanted[:,-1]))))
            maxerr['requested_to_command']=max(maxerr['requested_to_command'],float(np.max(abs(commands[tick:tick+8]-decoded[:8]))))
            now=tick*.05;chunk=TrajectoryChunkV1(chunk_id=f'EXPERT-TRAIN-{seed}-{index}',source='PI0276',observation_timestamp=now,point_timestamps=now+np.arange(40)*.05,eef_position=decoded[:,:3],eef_orientation=matrix_to_quat(rot6d_rows_to_matrix(decoded[:,3:9])),source_frame='PI0276_MANISKILL_WORLD',tcp_semantic_id='PI0276_PANDA_HAND_TCP',validity_start=now,validity_end=now+1.95,gripper=decoded[:,-1])
            verdict=gate.evaluate(chunk,now=now,achieved_position=a['actual_state'][:3],achieved_orientation=matrix_to_quat(rot6d_rows_to_matrix(a['actual_state'][None,3:9]))[0]).as_dict();assert verdict['decision']==q['gate']['decision'];gates[verdict['decision']]+=1
            for key in ('linear_speed_mps','angular_speed_radps'):assert abs(verdict['metrics'][key]['max']-q['gate']['metrics'][key]['max'])<1e-6
            clip=json.loads(str(b['clip_audit_json']));clips+=clip['native_preclip_out_of_range_count'];identities.append(dict(input=identity(folder/'input.npz'),output=identity(folder/'output.npz'),codec_identity=json.loads(str(b['identity_json']))))
        with np.load(d/'initial.npz',allow_pickle=False) as f:base=f['articulations/panda_wristcam'][0,:7]
        root=np.eye(4);root[:3,:3]=Rotation.from_quat(base[[4,5,6,3]]).as_matrix();root[:3,3]=base[:3]
        target=np.tile(np.eye(4),(660,1,1));target[:,:3,3]=commands[:,:3];target[:,:3,:3]=rot6d_rows_to_matrix(commands[:,3:9])
        maxerr['installed_pose']=float(np.max(abs(installed['base_target']-np.linalg.inv(root)@target)))
        maxerr['gripper_actuator']=float(np.max(abs(installed['action'][:,-1]-(2*np.clip(commands[:,-1],0,1)-1))))
        assert maxerr['gripper_actuator']<1e-6
        assert maxerr['position_m']<=1e-5 and maxerr['rotation_rad']<=1e-4 and maxerr['gripper']<=1e-6 and maxerr['requested_to_command']==0 and maxerr['installed_pose']<=1e-5 and maxerr['actual_state']<=1e-6
        rows.append(dict(seed=seed,status=result['status'],science_pass=persistent,result=identity(d/'RESULT.json'),original_expert_files=[identity(parent/n) for n in card['expert_parents'][str(seed)]['files']],raw_replay_files=[identity(d/n) for n in ('physics.npz','QUERIES.json','installed.npz','commands.npz','observations.npz','initial.npz')],query_files=identities,controls=660,preroll_controls=20,task_controls=640,task_s=32.,preroll_s=1.,integrations=3300,physics_samples=3301,queries=80,neural_model_queries=0,window_physics_indices=window.tolist(),window_task_times_s=(p['time_s'][window]-1).tolist(),persistent_predicate_all41=predicate[window].tolist(),persistent_pass_samples=int(predicate[window].sum()),gate_decisions=dict(gates),max_errors=maxerr,native_preclip_out_of_range_values=clips,ik_holds=result['ik_holds']))
    v=dict(status='PASS_FOUR_ORIGINAL_EXPERT_CODEC_REPLAYS_CPU' if all(r['science_pass'] for r in rows) else 'SCIENTIFIC_FAIL_ORIGINAL_EXPERT_CODEC_REPLAY',rows=rows,persistent_count=sum(r['science_pass'] for r in rows),denominator=4,inputs=[identity(panel),identity(cardpath)]+[identity(ROOT/'data'/n) for n in ('processor_config.json','statistics.json','embodiment_id.json')],
        codec='Original official StateActionProcessor encode/decode recomputed without Gr00t model or image processor; same20Hz commanded references, actual anchors, stock inverse clipping',new_model_calls=0,new_actor_calls=0,new_physics_steps=0,new_teacher_calls=0,new_training_updates=0,selected='NONE',deployment='NONE')
    save('audit','EXPERT_CODEC_CPU_RECOMPUTE.json',v);print(json.dumps(dict(status=v['status'],persistent_count=v['persistent_count'],rows=[{k:r[k] for k in ('seed','gate_decisions','max_errors')} for r in rows])))

def droid_observation(card,row,language_key):
    """CPU construction matching saved PI0280 native.py; no future labels in query."""
    import numpy as np
    with np.load(row['input_path'],allow_pickle=False) as f:
        state=f['state'];observation=dict(video={k:f[k][None].copy() for k in card['video_key_binding']},
            state={k:state[a:b][None,None].copy() for k,(a,b) in card['state_slices'].items()},language={language_key:[[row['language']]]})
    return observation

def retention():
    import numpy as np
    old=BASE/'PI0280';cardpath=old/'design/NATIVE_CARD.json';card=json.loads(cardpath.read_text());tag=card['source_tag']
    bindingpath=old/'design/BASELINE_BINDING.json';binding=json.loads(bindingpath.read_text());modelroot=Path(binding['model_root'])
    pristine=json.loads((old/'evidence/PRISTINE_FILES.json').read_text());expected={p['path']:p['sha256'] for p in pristine['files']}
    assets={}
    for name in ('processor_config.json','statistics.json','embodiment_id.json','config.json','model.safetensors.index.json'):
        p=modelroot/name;assert sha(p)==expected[name];assets[name]=identity(p)
    processor=json.loads((modelroot/'processor_config.json').read_text());config=processor['processor_kwargs']['modality_configs'][tag]
    mapping=json.loads((modelroot/'embodiment_id.json').read_text());languagekey=config['language']['modality_keys'][0]
    assert tag=='oxe_droid_relative_eef_relative_joint' and card['history_frame_deltas']==[-15,0] and card['fps']==15 and len(card['rows'])==32
    prior=json.loads((old/'native/RESULT.json').read_text());assert prior['completed_pairs']==32 and prior['effective_weights_unchanged']
    rows=[];tuples=[]
    for row in card['rows']:
        inp=Path(row['input_path']);assert sha(inp)==row['input_sha256'];d=old/'native/outputs'/row['id']
        with np.load(d/'stock.npz',allow_pickle=False) as f:a={k:f[k] for k in f.files}
        with np.load(d/'thin_wrapper.npz',allow_pickle=False) as f:b={k:f[k] for k in f.files}
        noise=np.load(d/'actual_initial_noise.npy',allow_pickle=False);assert noise.shape==(1,40,132) and noise.dtype==np.uint16
        assert np.array_equal(noise,a['noise_uint16']) and np.array_equal(noise,b['noise_uint16'])
        for k in ('raw','decoded_eef_9d','decoded_gripper_position','decoded_joint_position'):assert np.array_equal(a[k],b[k])
        pair=json.loads((d/'PAIR.json').read_text());noisehash=hashlib.sha256(noise.tobytes()).hexdigest();assert noisehash==pair['noise_sha256']
        obs=droid_observation(card,row,languagekey)
        assert all(x.shape[0:2]==(1,2) and x.dtype==np.uint8 for x in obs['video'].values())
        assert {k:list(x.shape) for k,x in obs['state'].items()}=={'eef_9d':[1,1,9],'gripper_position':[1,1,1],'joint_position':[1,1,7]}
        with np.load(inp,allow_pickle=False) as f:
            assert f['actual_history_frame_indices'].tolist()==[row['frame']-15,row['frame']]
            assert np.allclose(f['actual_history_timestamps_s'],[(row['frame']-15)/15,row['frame']/15])
        token=(sha(inp),noisehash,assets['processor_config.json']['sha256'],assets['statistics.json']['sha256'],tag);tuples.append(token)
        rows.append(dict(id=row['id'],episode=row['episode'],frame=row['frame'],language=row['language'],input=identity(inp),
            actual_initial_noise=identity(d/'actual_initial_noise.npy'),actual_noise_array_sha256=noisehash,noise_dtype='uint16 exact BF16 storage bits',noise_shape=[1,40,132],
            pristine_output=identity(d/'stock.npz'),prior_wrapper_output=identity(d/'thin_wrapper.npz'),pair=identity(d/'PAIR.json'),
            observation_shapes={category:{k:list(v.shape) if hasattr(v,'shape') else [1,1] for k,v in values.items()} for category,values in obs.items()},
            raw_pristine_array_sha256=hashlib.sha256(a['raw'].tobytes()).hexdigest(),raw_shape=list(a['raw'].shape),
            prior_task_metric='NOT_MEASURED_OFFICIAL_OPEN_LOOP_DESCRIPTIVE_ONLY'))
    assert len(set(tuples))==32
    official=old/'official';sourcefiles=[old/'code/native.py',official/'gr00t/policy/gr00t_policy.py',official/'gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py',official/'gr00t/data/state_action/state_action_processor.py',official/'gr00t/model/gr00t_n1d7/action_head/flow_matching_action_head.py']
    v=dict(schema='PI0282_FIXED_ORIGINAL_DROID_RETENTION32_V1',status='FROZEN_32_EXISTING_TUPLES_CPU_NO_FRESH_QUERIES',rows=rows,source_tag=tag,projector_index=mapping[tag],original_modality_config=config,
        source_model_revision=card['model_revision'],official_source_revision=card['official_code'],assets=assets,
        prior_bindings=[identity(bindingpath),identity(cardpath),identity(old/'native/RESULT.json'),identity(old/'evidence/NATIVE_CPU_READBACK.json'),identity(old/'evidence/PRISTINE_FILES.json')],source_files=[identity(p) for p in sourcefiles if p.exists()],
        helper='scripts/pi0282_execution/audit.py::droid_observation reads actual old17D state and2 true historical RGB frames. Original future descriptive labels remain on disk, excluded from returned query.',
        owner_binding_required=dict(status='EVALUATION_ENTRY_EXTENSION_REQUIRED',current_entry=str(ROOT/'code/inference.py'),
            reason='Current inference.load/query constructs new EEF10 codec/current-frame10D observations; this entry cannot substitute for original DROID retention.',
            minimal_change='Reuse endpoint loaded model and quality ledger, construct separate ORIGINAL processor/statistics/tag mapping from indexed assets. Reuse official Gr00tPolicy observation→VLAStepData→processor→collator→BF16→model.get_action→original processor.decode_action path; use droid_observation helper and stored BF16 bits as sole initial native noise. No fresh pristine inference.',
            calls='Only each2048 endpoint32=64 fresh quality calls, original full S4/noise132 semantics; preserve comparison artifacts, INTERFACE exact route equality expected; ACTION changes descriptive.',
            invariants=['Original DROID17D state with seven real joints','Original [-15,0] real RGB/history,15Hz output and H40','Original serialized pretrain modality/statistics/mapping','Original stock image/token augmentation/eval behavior','Stock BF16 S4,FP32 decode','Stored full noise bits uint16→BF16 view, no cast/zero channels','No newEEF processor or source candidate selection use','Both loaded endpoint effective hashes bound before/after']),
        fixed_candidate_points=['ACTION2048','INTERFACE2048'],selection_metric=False,comparison='Saved pristine stock.npz at identical observation/noise/processor tuple',
        no_model_download=True,new_model_calls=0,new_actor_calls=0,new_physics_steps=0,new_render_calls=0)
    save('audit','RETENTION32.json',v);print(json.dumps(dict(status=v['status'],tuples=len(rows),projector=mapping[tag],owner_entry=v['owner_binding_required']['status'])))

if __name__=='__main__':
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only audit requires hidden CUDA'
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['exposure','rules','plan','expert','retention']);a=p.parse_args();{'exposure':exposure,'rules':fixed_rules,'plan':plan_audit,'expert':expert_audit,'retention':retention}[a.phase]()
