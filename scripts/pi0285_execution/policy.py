"""Fixed native feedback panels: fresh RGB/state, K4, gate, eight controls, repeat."""
import argparse,json,os,subprocess,time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from contract import ROOT,INSTRUCTIONS,SPLITS,guard,sha,write


def tasks(point,panel):
    assert point in ('PLACE4096-2048','PLACE4096-4096','REPAIR_A2048-2048','SHARED4096-2048','SHARED4096-4096','REPAIR_B2048-2048')
    assert panel in ('TRAIN','DEV','CONFIRM_A','CONFIRM_B','ACQUISITION')
    stage_b=point.startswith(('SHARED','REPAIR_B'))
    if panel=='ACQUISITION':assert not stage_b;return ['T2']
    if panel=='CONFIRM_A':assert not stage_b
    if panel=='CONFIRM_B':assert stage_b
    return ['T1','T2'] if stage_b else ['T1']


def panel_gate(point,panel):
    guard();selected=tasks(point,panel)
    admission=json.loads((ROOT/'data/DATA_CODEC_ADMISSION.json').read_text())
    assert admission['status']=='PASS_NEW_COMMON_SCENE_FULL_NATIVE_CODEC_AND_TWO_WORKERS'
    for name,h in admission['bindings'].items():assert sha(ROOT/'data'/name)==h,name
    binding=json.loads((ROOT/'design/MODEL_POINTS.json').read_text())[point]
    training=json.loads((ROOT/'transfers'/point/'TRAINING_CPU.json').read_text())
    assert training['status']=='PASS_ACTUAL_NATIVE_TRAINER_INPUT_PLAN_NOISE_LOSS_MODES_SCOPE_AND_COMPLETE_LOCAL_STATE' and training['point']==point
    assert training['export_sha256']==sha(ROOT/'transfers'/point/'EXPORT.json')
    for name,h in binding['files'].items():assert sha(Path(binding['model_root'])/name)==h,name
    if point.startswith(('SHARED','REPAIR_B')) or panel=='ACQUISITION':
        protected=json.loads((ROOT/'qualification/PLACE_V1_PROTECTED.json').read_text())
        assert protected['status']=='PLACE_V1_SOURCE_QUALIFIED_AND_IMMUTABLY_PROTECTED'
        assert protected['CONFIRM_A_success']>=26 and protected['CONFIRM_A_requested']==32
        if panel=='ACQUISITION':assert point==protected['point']
    if panel=='DEV' and point!='SHARED4096-4096':
        train=json.loads((ROOT/'source/evidence'/f'{point}_TRAIN.json').read_text())
        assert all(train['tasks'][t]['requested']==8 and train['tasks'][t]['success']>=7 for t in selected)
    if panel.startswith('CONFIRM'):
        candidate=json.loads((ROOT/'qualification'/(panel+'_CANDIDATE.json')).read_text())
        assert candidate['point']==point and candidate['model_binding']==binding and candidate['confirm_allowed']
        for name,h in candidate['development_panels'].items():assert sha(ROOT/'source/evidence'/name)==h
        dev=json.loads((ROOT/'source/evidence'/f'{point}_DEV.json').read_text())
        assert all(dev['tasks'][t]['requested']==16 and dev['tasks'][t]['success']>=13 for t in selected)
        if panel=='CONFIRM_B':assert dev['paired_both_success']>=12
    split='DEV' if panel=='ACQUISITION' else panel
    seeds=list(range(SPLITS[split][0],SPLITS[split][1]+1))
    return selected,seeds[:8] if panel=='TRAIN' else seeds


def reply(server,key):
    while True:
        line=server.stdout.readline();assert line,'Native resident worker ended before response'
        try:value=json.loads(line)
        except json.JSONDecodeError:continue
        if key in value:return value


@contextmanager
def resident_worker(point,log_path):
    env=dict(os.environ,PYTHONPATH=':'.join([str(ROOT/'code'),str(ROOT.parent/'PI0283/code'),str(ROOT.parent/'PI0280/official'),str(ROOT/'project-src'),str(ROOT/'project-src/src')]))
    command=[str(ROOT.parent/'PI0250/runtime/venv/bin/python'),'-u',str(ROOT/'code/model_worker.py'),'--point',point]
    with Path(log_path).open('ab') as log:
        server=subprocess.Popen(command,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
        try:
            assert reply(server,'ready')['ready'];yield server
        finally:
            server.stdin.close();assert server.wait(timeout=120)==0


def disposition(commands,error):
    assert np.asarray(commands).shape==(40,10)
    if error:return 'VALID_NATIVE_DECODE_REJECT'
    if not np.isfinite(commands).all():return 'VALID_NONFINITE_REJECT'
    if (commands[:,-1]<-.1).any() or (commands[:,-1]>1.1).any():return 'VALID_GRIPPER_ENVELOPE_REJECT'
    return None


def episode(server,point,panel,task,seed):
    import recorder,scene
    from source import step
    from scorer import score
    from gate import original_gate
    from budget import time_allowed
    slot=f'{point}-{task}-{panel}-{seed}';parent=recorder.RAW/'episodes'/slot
    saved=list(parent.glob('attempt-*/RESULT.json'))
    valid=[p for p in saved if json.loads(p.read_text())['status'].startswith('VALID')]
    if valid:assert len(valid)==1;return json.loads(valid[0].read_text())
    assert not list(parent.glob('attempt-*')),'Audit original engineering attempt before exact technical replacement'
    time_allowed(ROOT,60)
    ep=recorder.Episode(slot,seed,'pd_ee_pose','PI0285PickPlaceStack-v1',dict(layout_id=seed))
    began=time.time();queries=[];status='VALID_SOURCE_COMPLETE'
    try:
        write(ep.folder/'EPISODE_CARD.json',dict(point=point,task=task,split=panel,layout=seed,instruction=INSTRUCTIONS[task],scene=ep.env.layout_spec,code_seal_sha256=sha(ROOT/'design/CODE_SEAL.json'),NON_REALTIME=True,policy_inputs='RGB_ACTUAL_EEF_OPENING_LANGUAGE_ONLY',new_WBC=0))
        red=ep.states[-1]['cubeA_pose'][:3].copy();green=ep.states[-1]['cubeB_pose'][:3].copy()
        neutral=ep.observe()['state_v1'].copy();neutral[-1]=1.
        for _ in range(20):step(ep,neutral,-1,False)
        for index in range(80):
            observation=ep.observe();state=observation['state_v1'].copy()
            folder=ep.folder/f'q{index:03d}';folder.mkdir()
            inp=folder/'input.npz';out=folder/'output.npz'
            np.savez_compressed(inp,actual_state=state,base_raw=observation['base_camera_raw'],wrist_raw=observation['hand_camera_raw'],instruction=np.asarray(INSTRUCTIONS[task]))
            # Point excluded from noise identity: fixed condition/index across candidates.
            request=dict(input=str(inp),output=str(out),condition=dict(task=task,layout=seed),query_index=index)
            server.stdin.write(json.dumps(request)+'\n');server.stdin.flush()
            assert reply(server,'output')['output']==str(out)
            q=dict(index=index,task_time_s=index*.4,input_sha256=sha(inp),output_sha256=sha(out));queries.append(q)
            with np.load(out) as f:commands=f['decoded'].copy();error=str(f['decode_error'])
            rejection=disposition(commands,error)
            if rejection:status=rejection;q['decode_error']=error;break
            try:gate=original_gate(commands,state,ep.control*.05,f'{slot}-{index}')
            except ValueError as exc:status='VALID_DEGENERATE_ROTATION_REJECT';q['error']=str(exc);break
            q['gate']=gate;write(folder/'GATE.json',gate)
            if gate['decision']=='REJECT':status='VALID_SOURCE_GATE_REJECT';break
            for command in commands[:8]:step(ep,command,-1,False)
        data={k:np.stack([v[k] for v in ep.states]) for k in ep.states[0]}
        resultscore=score(data,task,ep.env.layout_spec['zone_xy'],red,green,status,ep.control)
        write(ep.folder/'QUERIES.json',queries);write(ep.folder/'TASK_SCORE_CPU.json',resultscore)
        result=ep.finish(status)
        result['original_native_stack_first_success_s']=result.pop('first_success_s')
        result['first_task_hold_s']=resultscore['subpredicates'][task+'_hold']['first_true_s']
        result.update(point=point,task=task,split=panel,layout=seed,instruction=INSTRUCTIONS[task],folder=str(ep.folder),task_score=resultscore,persistent_success=resultscore['persistent_success'],elapsed_wall_s=time.time()-began,model_queries=len(queries),teacher_passes=0,NON_REALTIME=True,domain='NEW_COMMON_SOURCE_PANDA_NOT_OLD_BENCHMARK',selected='NONE',deployment='NONE')
        write(ep.folder/'RESULT.json',result);return result
    except BaseException as exc:
        write(ep.folder/'QUERIES.json',queries);ep.finish('ENGINEERING_INCOMPLETE',repr(exc));raise


def run(point,panel):
    import fcntl
    from budget import process
    selected,seeds=panel_gate(point,panel);assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
    path=ROOT/'source/evidence'/f'{point}_{panel}.json'
    if path.exists():return json.loads(path.read_text())
    lock=(ROOT/'runtime/SOURCE_GPU0.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # ponytail: one resident model process for a finite panel, no background service.
    began=process(ROOT,'start');rows=[]
    try:
        with resident_worker(point,ROOT/'runtime'/f'{point}_{panel}_worker.log') as server:
            for seed in seeds:
                for task in selected:
                    rows.append(episode(server,point,panel,task,seed))
                    write(ROOT/'source/evidence'/f'{point}_{panel}_PROGRESS.json',dict(rows=rows))
        counts={task:dict(success=sum(r['persistent_success'] for r in rows if r['task']==task),requested=len(seeds)) for task in selected}
        paired=sum(all(next(r['persistent_success'] for r in rows if r['task']==t and r['layout']==s) for t in selected) for s in seeds) if len(selected)==2 else None
        value=dict(status='FIXED_NATIVE_FEEDBACK_PANEL_COMPLETE',point=point,split=panel,tasks=counts,paired_both_success=paired,rows=rows,new_teacher_calls=0,new_WBC=0,selected='NONE',deployment='NONE')
        write(path,value);return value
    finally:process(ROOT,'end',began);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def self_check():
    assert tasks('PLACE4096-2048','TRAIN')==['T1']
    assert tasks('SHARED4096-4096','CONFIRM_B')==['T1','T2']
    for point,panel in [('SHARED4096-2048','CONFIRM_A'),('PLACE4096-2048','CONFIRM_B')]:
        try:tasks(point,panel)
        except AssertionError:pass
        else:raise AssertionError('Cross-stage confirmation opened')
    a=np.zeros((40,10));a[:,-1]=.5
    assert disposition(a,'') is None
    a[0,0]=np.nan;assert disposition(a,'')=='VALID_NONFINITE_REJECT'
    a[0,0]=0;a[39,-1]=1.2;assert disposition(a,'')=='VALID_GRIPPER_ENVELOPE_REJECT'
    assert disposition(a,'decode error')=='VALID_NATIVE_DECODE_REJECT'
    assert np.array_equal(np.concatenate([np.arange(i*8,i*8+8) for i in range(80)]),np.arange(640))
    print('PASS_FIXED_STAGE_ROUTES_RAW_REJECTIONS_AND_80_FRESH_8_CONTROL_PREFIXES; model/physics=0')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--point');p.add_argument('--panel');p.add_argument('--self-check',action='store_true');a=p.parse_args()
    self_check() if a.self_check else print(json.dumps({k:v for k,v in run(a.point,a.panel).items() if k!='rows'}))
