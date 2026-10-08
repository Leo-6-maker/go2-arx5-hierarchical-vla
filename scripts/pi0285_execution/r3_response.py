"""Finite R3 language response probes; preparation is CPU only, owner runs GPU modes."""
import argparse, hashlib, json, os, sys
from pathlib import Path
from contract import ROOT, INSTRUCTIONS, sha, write, guard
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_name]='4'


def noise_identity(uid, replica):
    key=json.dumps(dict(condition=dict(WP4_UID=uid,noise_replica=replica),query_index=0),sort_keys=True)
    return dict(condition=dict(WP4_UID=uid,noise_replica=replica),query_index=0,seed=int.from_bytes(hashlib.sha256(('PI0285:'+key).encode()).digest()[:8],'little')%(2**63-1),noise_file='runtime/noise/'+hashlib.sha256(key.encode()).hexdigest()+'.npy')


def manifest(root):
    import numpy as np
    root=Path(root);diag=root/'design/DIAG96_PER_TASK.json';plan=json.loads(diag.read_text());rows=[]
    bank=json.loads((root/'data/SOURCE_BANK.json').read_text())
    for layout in range(285100,285108):
        matches=[r for r in plan['rows'] if r['task']=='T1' and r['split']=='TRAIN' and r['layout']==layout and r['phase']==2]
        assert len(matches)==1;row=matches[0];source=Path(row['input']);assert sha(source)==row['input_sha256']
        with np.load(source) as f:
            base=f['base_raw'];wrist=f['wrist_raw'];state=f['actual_state'];assert str(f['instruction'])==INSTRUCTIONS['T1']
        episode=row['episode'];time_path=root/'data/policy_observation'/episode/'time_s.npy';time=np.load(time_path,mmap_mode='r')
        phase=np.load(root/'data/action_target'/episode/'phase.npy',mmap_mode='r');assert row['anchor']==np.flatnonzero(phase[20:621]==2)[0]+20
        original=next(e for e in bank['admitted_episodes'] if e['episode']==episode);folder=Path(original['folder']);card=json.loads((folder/'EPISODE_CARD.json').read_text())
        with np.load(folder/'physics.npz') as f:green=f['cubeB_pose'][row['anchor']*5,:3].copy()
        ah=lambda x:hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
        rows.append(dict(uid=row['uid'],task='T1',split='TRAIN',layout=layout,anchor=row['anchor'],phase_anchor_rule='FIRST_PHASE2_FIXED_DIAG_NO_REPLACEMENTS',time_s=float(time[row['anchor']]),time_source_sha256=sha(time_path),source_input=str(source),source_input_sha256=sha(source),base_RGB_sha256=ah(base),wrist_RGB_sha256=ah(wrist),actual_state_sha256=ah(state),RGB_shape=list(base.shape),state_shape=list(state.shape),declared_goal_geometry_evidence_only=dict(T1_blue_zone_xy=card['scene']['zone_xy'],T2_red_center_on_green_xyz=(green+np.asarray([0,0,.04])).tolist(),never_model_input=True,not_counterfactual_expert_labels=True),noise=[dict(noise_identity(row['uid'],n),packet_sha256='UNOBSERVED',status='NOT_FROZEN_REAL_GPU_BF16_PACKET') for n in range(2)]))
    return dict(status='FROZEN_REAL_INPUT_IDENTITY_CPU_ONLY_NO_NEURAL_RESULTS',owner='r3_response',root=str(root),DIAG_plan_sha256=sha(diag),processor={n:sha(root/'data'/n) for n in ('processor_config.json','statistics.json','embodiment_id.json')},rows=rows,instructions=INSTRUCTIONS,slots=['EARLIEST1024','FINAL_T1','FINAL_SHARED'],requests_per_distinct_point=32,max_distinct_points=3,max_language_requests=96,max_all_R3_descriptive_requests=128,max_all_R3_descriptive_K4_heads=512,noise_rule='ALL16_REAL_CUDA_BF16_1x40x132_PACKETS_FROZEN_BEFORE_ANY_OUTPUT; condition excludes instruction/model/point',noise_status='UNOBSERVED',query_results='UNOBSERVED',new_model_queries=0,new_physics=0,new_teacher=0,new_render=0,scientific_effect='DESCRIPTIVE_ONLY_NO_QUALIFICATION_SELECTION_OR_BRANCH_EFFECT',alternate_expert_labels=False)


def runtime_plan():
    guard();p=ROOT/'r3/WP4/FROZEN_INPUT_PLAN.json';v=json.loads(p.read_text());assert len(v['rows'])==8 and v['instructions']==INSTRUCTIONS
    assert v['rows']==manifest(ROOT)['rows'],'Frozen inputs/UIDs/time/noise conditions must equal the original fixed anchors'
    assert v['DIAG_plan_sha256']==sha(ROOT/'design/DIAG96_PER_TASK.json')
    for n,h in v['processor'].items():assert sha(ROOT/'data'/n)==h
    for row in v['rows']:assert sha(row['source_input'])==row['source_input_sha256']
    for p in (ROOT/'qualification').glob('*_ROUTE.json'):
        assert json.loads(p.read_text()).get('next_action')!='STOP_CONFIRMATION_VALID_QUALITY_FAIL_NO_SECOND_CANDIDATE','No probes after failed confirmation'
    for panel in ('CONFIRM_A','CONFIRM_B'):
        for p in (ROOT/'source/evidence').glob('*_'+panel+'.json'):
            actual=json.loads(p.read_text());assert all(t['requested']==32 and t['success']>=26 for t in actual['tasks'].values()),'Failed confirmation ends probes'
            if panel=='CONFIRM_B':assert actual['paired_both_success']>=24
    # Owner reaches a complete primary/frozen-candidate boundary before optional probes.
    assert (ROOT/'transfers/PLACE4096-4096/TRAINING_CPU.json').is_file() or (ROOT/'qualification/CONFIRM_A_CANDIDATE.json').is_file()
    return v


def freeze_noise():
    import fcntl, numpy as np, torch
    from budget import process,time_allowed
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0';torch.set_num_threads(4);v=runtime_plan();time_allowed(ROOT,2)
    out=ROOT/'r3/WP4';out.mkdir(parents=True,exist_ok=True)
    assert not list(out.glob('*/outputs/*.npz')),'All noise must precede every probe result'
    started=process(ROOT,'start');rows=[]
    try:
        with (ROOT/'runtime/MODEL_LEASE.lock').open('a') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for row in v['rows']:
                for n in row['noise']:
                    p=ROOT/n['noise_file'];p.parent.mkdir(parents=True,exist_ok=True)
                    if not p.exists():
                        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                            torch.manual_seed(n['seed']);torch.cuda.manual_seed_all(n['seed']);packet=torch.randn(size=(1,40,132),dtype=torch.bfloat16,device='cuda:0').cpu().view(torch.uint16).numpy()
                        temp=p.with_suffix('.tmp')
                        with temp.open('wb') as f:np.save(f,packet);f.flush();os.fsync(f.fileno())
                        temp.replace(p)
                    packet=np.load(p);assert packet.dtype==np.uint16 and packet.shape==(1,40,132)
                    rows.append(dict(uid=row['uid'],**n,packet_sha256=sha(p),status='FROZEN_NATIVE_CUDA_BF16_FULL_PACKET'))
            assert len(rows)==16 and len({r['noise_file'] for r in rows})==16
            write(out/'NOISE16_FROZEN.json',dict(status='ALL16_NATIVE_CUDA_BF16_PACKETS_FROZEN_BEFORE_PROBES',input_plan_sha256=sha(out/'FROZEN_INPUT_PLAN.json'),rows=rows,new_model_queries=0))
    finally:process(ROOT,'end',started)


def run(point,slot):
    import numpy as np
    from policy import resident_worker,reply,panel_gate
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0';v=runtime_plan();panel_gate(point,'TRAIN')
    if slot=='EARLIEST1024':assert point=='PLACE4096-1024'
    elif slot=='FINAL_SHARED':
        candidate=ROOT/'qualification/CONFIRM_B_CANDIDATE.json'
        assert point==json.loads(candidate.read_text())['point'] if candidate.exists() else point in ('SHARED4096-4096','REPAIR_B2048-2048')
    else:
        assert slot=='FINAL_T1';candidate=ROOT/'qualification/CONFIRM_A_CANDIDATE.json'
        assert point==json.loads(candidate.read_text())['point'] if candidate.exists() else point in ('PLACE4096-4096','REPAIR_A2048-2048')
    root=ROOT/'r3/WP4';freeze=json.loads((root/'NOISE16_FROZEN.json').read_text());assert freeze['input_plan_sha256']==sha(root/'FROZEN_INPUT_PLAN.json') and len(freeze['rows'])==16
    for n in freeze['rows']:assert sha(ROOT/n['noise_file'])==n['packet_sha256']
    registry=root/'POINTS.json';points=json.loads(registry.read_text()) if registry.exists() else {}
    binding=json.loads((ROOT/'design/MODEL_POINTS.json').read_text())[point];identity=hashlib.sha256(json.dumps(binding['effective_bf16_tensor_sha256'],sort_keys=True).encode()).hexdigest()
    matches=[r for r in points.values() if r['identity']==identity]
    if matches:
        value=matches[0];assert (root/value['point']/'RESULT.json').is_file();points[slot]=value;write(registry,points);return value
    assert slot not in points and len({r['identity'] for r in points.values()})<3
    folder=root/point;outputs=folder/'outputs';outputs.mkdir(parents=True,exist_ok=True);records=[]
    ledger=root/'DESCRIPTIVE_ATTEMPTS.jsonl'
    with resident_worker(point,folder/'worker.log') as server:
        for row in v['rows']:
            for task,text in INSTRUCTIONS.items():
                inp=root/'inputs'/row['uid']/(task+'.npz');inp.parent.mkdir(parents=True,exist_ok=True)
                with np.load(row['source_input']) as f:content={n:f[n].copy() for n in ('base_raw','wrist_raw','actual_state')}
                content['instruction']=np.asarray(text)
                if not inp.exists():np.savez_compressed(inp,**content)
                with np.load(inp) as f:assert set(f.files)==set(content) and all(np.array_equal(f[n],x) for n,x in content.items())
                for n in row['noise']:
                    replica=n['condition']['noise_replica'];output=outputs/(row['uid']+'-'+task+'-'+str(replica)+'.npz');request=dict(input=str(inp),output=str(output),condition=n['condition'],query_index=0)
                    if not output.exists():
                        attempts=[json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
                        assert not any(x['output']==str(output) for x in attempts),'Attempt without output needs owner engineering audit; no automatic retry'
                        assert len(attempts)<128 and sum(x.get('kind')=='language' for x in attempts)<96
                        with ledger.open('a') as f:f.write(json.dumps(dict(kind='language',output=str(output),point=point,slot=slot,identity=identity))+'\n');f.flush();os.fsync(f.fileno())
                        server.stdin.write(json.dumps(request)+'\n');server.stdin.flush();assert reply(server,'output')['output']==str(output)
                    with np.load(output) as f:
                        assert str(f['source_sha256'])==sha(inp) and str(f['instruction'])==text and int(f['K'])==4
                        assert np.array_equal(f['noise_uint16'],np.load(ROOT/n['noise_file'])) and np.array_equal(f['actual_state'],content['actual_state'])
                    records.append(dict(uid=row['uid'],task=task,replica=replica,input=str(inp),input_sha256=sha(inp),output=str(output),output_sha256=sha(output)))
    assert len(records)==32
    value=dict(point=point,slot=slot,identity=identity,status='COMPLETE32_DESCRIPTIVE_K4_LANGUAGE_PROBES',records=records,normalizer_changed=False,qualification_effect=False)
    write(folder/'RESULT.json',value);points[slot]={k:value[k] for k in ('point','slot','identity','status')};write(registry,points);return value


def compare(point):
    import numpy as np
    from diagnostics import metrics
    from policy import disposition
    from gate import original_gate
    folder=ROOT/'r3/WP4'/point;source=folder/'RESULT.json';v=json.loads(source.read_text());values={};outputs=[]
    for row in v['records']:
        assert sha(row['output'])==row['output_sha256'] and sha(row['input'])==row['input_sha256']
        with np.load(row['output']) as f:decoded=f['decoded'].copy();raw=f['raw'].copy();error=str(f['decode_error']);noise=f['noise_uint16'].copy();state=f['actual_state'].copy()
        rejected=disposition(decoded,error);values[row['uid'],row['task'],row['replica']]=(decoded,noise,rejected)
        gate=dict(decision='REJECT',reason=rejected) if rejected else original_gate(decoded,state,0.,'WP4-'+row['uid'])
        outputs.append(dict(uid=row['uid'],task=row['task'],replica=row['replica'],rejection=rejected,gate=gate,decode_error=error,raw_OOB_first8_by_channel=(abs(raw[0,:8,:10])>1).sum(0).tolist(),raw_OOB_H40_by_channel=(abs(raw[0,:,:10])>1).sum(0).tolist(),output_sha256=row['output_sha256']))
    differences=[]
    for uid in sorted({k[0] for k in values}):
        for n in range(2):
            a,noise_a,ra=values[uid,'T1',n];b,noise_b,rb=values[uid,'T2',n];assert np.array_equal(noise_a,noise_b)
            differences.append(dict(uid=uid,comparison='SAME_NOISE_DIFFERENT_TEXT',replica=n,first8=None if ra or rb else metrics(a[:8],b[:8],True),H40=None if ra or rb else metrics(a,b,True),rejections=[ra,rb]))
        for task in INSTRUCTIONS:
            a,_,ra=values[uid,task,0];b,_,rb=values[uid,task,1]
            differences.append(dict(uid=uid,comparison='SAME_TEXT_NOISE_SPREAD',task=task,first8=None if ra or rb else metrics(a[:8],b[:8],True),H40=None if ra or rb else metrics(a,b,True),rejections=[ra,rb]))
    result=dict(status='CPU_DESCRIPTIVE_LANGUAGE_AND_NOISE_DIFFERENCES_NOT_QUALIFICATION',point=point,source_sha256=sha(source),outputs=outputs,differences=differences,alternate_expert_labels=False,new_model_queries=0,new_physics=0,interpretation='Differences measure response only; shared grasp can legitimately match, T1 training need not master T2, invalid results are retained, no sensitivity threshold selects training or candidate.')
    write(folder/'RESPONSE_PROBES.json',result);return result


def self_check():
    import io,numpy as np
    from diagnostics import metrics
    from policy import disposition
    values=[noise_identity('T1-TRAIN-'+str(x)+'-100',n) for x in range(285100,285108) for n in range(2)]
    assert len({r['noise_file'] for r in values})==16 and values==[noise_identity('T1-TRAIN-'+str(x)+'-100',n) for x in range(285100,285108) for n in range(2)]
    assert all(set(r['condition'])=={'WP4_UID','noise_replica'} and r['query_index']==0 for r in values)
    assert 8*2*2==32 and 32*3==96 and 128*4==512
    content=dict(base_raw=np.zeros((128,128,3),dtype=np.uint8),wrist_raw=np.ones((128,128,3),dtype=np.uint8),actual_state=np.asarray([0,0,0,1,0,0,0,1,0,1],dtype=np.float32));buffers=[]
    for text in INSTRUCTIONS.values():
        stream=io.BytesIO();np.savez(stream,**content,instruction=np.asarray(text));stream.seek(0);buffers.append(dict(np.load(stream)))
    assert all(np.array_equal(buffers[0][n],buffers[1][n]) for n in content) and str(buffers[0]['instruction'])!=str(buffers[1]['instruction'])
    x=np.tile(content['actual_state'],(40,1));y=x.copy();y[:,0]=.01;y[:,-1]=.8;y[:,3:9]=[0,1,0,-1,0,0]
    for prefix in (8,40):
        result=metrics(x[:prefix],y[:prefix],True);assert np.isclose(result['XYZ_RMS_m'],.01) and np.isclose(result['SO3_RMS_rad'],np.pi/2) and np.isclose(result['gripper_MAE'],.2)
    assert disposition(x,'invalid')=='VALID_NATIVE_DECODE_REJECT';x[0,0]=np.nan;assert disposition(x,'')=='VALID_NONFINITE_REJECT'
    return dict(status='PASS_FINITE_NOISE_IDENTITIES_EXCLUDE_TEXT_POINT_INPUT_PAIRING_PREFIX_FULL_METRICS_AND_REJECTION',fixtures='SYNTHETIC_CPU_SCHEMA_ONLY_NOT_PHYSICAL_DATA',model=0,physics=0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('manifest','self-check','freeze-noise','run','cpu'));p.add_argument('--root',default=str(ROOT));p.add_argument('--out');p.add_argument('--point');p.add_argument('--slot',choices=('EARLIEST1024','FINAL_T1','FINAL_SHARED'));a=p.parse_args()
    if a.mode=='manifest':value=manifest(a.root);assert a.out;write(a.out,value)
    elif a.mode=='self-check':value=self_check()
    elif a.mode=='freeze-noise':freeze_noise();value={'status':'NOISE16_FROZEN'}
    elif a.mode=='run':value=run(a.point,a.slot)
    else:value=compare(a.point)
    print(json.dumps({k:v for k,v in value.items() if k not in ('rows','records','outputs','differences')}))
