"""Fixed native feedback panels: fresh RGB/state, K4, gate, eight controls, repeat."""
import argparse,json,os,subprocess,time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from contract import ROOT,INSTRUCTIONS,SPLITS,guard,sha,write


PARENT='PARENT_DOSE6144'
ENDPOINTS=('C0_REPLAY1024-1024','C1_CORRECT1024-1024')
FIXED_FILES={'statistics.json':'c3ec9c6b797bb735e3ac38d611b409262de04bbb965cb68b7426df749ba7fb81','processor_config.json':'fae35db9d84083cc1faac16442c062c74a22691fa3358cfc58efb759e01977d3','embodiment_id.json':'fb47934b4bd4b5e11ec1c61ade0c5e03ca9110ec5f16abce9633b10cfd9edfbb'}


def tasks(point,panel):
    assert panel in ('LEARNER_TRAIN16','TRAIN','DEV','CONFIRM_C')
    assert (point==PARENT and panel=='LEARNER_TRAIN16') or (point in ENDPOINTS and panel!='LEARNER_TRAIN16')
    return ['T1']


def panel_seeds(point,panel):
    tasks(point,panel)
    return list(range(*{'LEARNER_TRAIN16':(285108,285124),'TRAIN':(285100,285108),'DEV':(285200,285216),'CONFIRM_C':(285500,285532)}[panel]))


def panel_gate(point,panel):
    guard();selected=tasks(point,panel)
    for name,h in FIXED_FILES.items():assert sha(ROOT/'data'/name)==h,name
    binding=json.loads((ROOT/'design/MODEL_POINTS.json').read_text())[point]
    for name,h in binding['files'].items():assert sha(Path(binding['model_root'])/name)==h,name
    if point!=PARENT:
        training=json.loads((ROOT/'transfers'/point/'TRAINING_CPU.json').read_text())
        assert training['status'].startswith('PASS_') and training['point']==point
        assert training['export_sha256']==sha(ROOT/'transfers'/point/'EXPORT.json')
    if panel=='DEV':
        from qualification import panel as checked_panel
        train,_=checked_panel(point,'TRAIN');assert train and train['T1']>=7
    if panel=='CONFIRM_C':
        from qualification import confirmation_gate
        confirmation_gate(point)
    return selected,panel_seeds(point,panel)


def reply(server,key):
    while True:
        line=server.stdout.readline();assert line,'Native resident worker ended before response'
        try:value=json.loads(line)
        except json.JSONDecodeError:continue
        if key in value:return value


@contextmanager
def resident_worker(point,log_path):
    env=dict(os.environ,HF_HOME=str(ROOT.parent/'PI0250/runtime/hf_home'),PYTHONPATH=':'.join([str(ROOT/'code'),str(ROOT.parent/'PI0283/code'),str(ROOT.parent/'PI0280/official'),str(ROOT/'project-src'),str(ROOT/'project-src/src')]))
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


def audited_replacement(parent):
    for attempt in parent.glob('attempt-*'):
        result=attempt/'RESULT.json';receipt=attempt/'ENGINEERING_REPLACEMENT.json'
        assert result.is_file() and receipt.is_file(),'Audit original engineering attempt before exact technical replacement'
        v=json.loads(receipt.read_text())
        assert v['status']=='OWNER_AUDITED_INCOMPLETE_SAME_CONDITION_REPLACEMENT'
        assert json.loads(result.read_text())['status']=='ENGINEERING_INCOMPLETE' and v['result_sha256']==sha(result)
        for name,h in v['original_files'].items():assert sha(attempt/name)==h,name


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
    audited_replacement(parent)
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
        value=dict(status='PI0286_FIXED_FIRST_VALID_NATIVE_PANEL_COMPLETE',point=point,split=panel,tasks=counts,paired_both_success=paired,rows=rows,new_teacher_calls=0,new_WBC=0,selected='NONE',deployment='NONE')
        write(path,value);return value
    finally:process(ROOT,'end',began);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def self_check():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        parent=Path(d);attempt=parent/'attempt-000';attempt.mkdir()
        try:audited_replacement(parent)
        except AssertionError:pass
        else:raise AssertionError('Unreviewed incomplete attempt accepted')
        write(attempt/'RESULT.json',dict(status='ENGINEERING_INCOMPLETE'))
        write(attempt/'ENGINEERING_REPLACEMENT.json',dict(status='OWNER_AUDITED_INCOMPLETE_SAME_CONDITION_REPLACEMENT',result_sha256=sha(attempt/'RESULT.json'),original_files={}))
        audited_replacement(parent)
        write(attempt/'RESULT.json',dict(status='VALID_SOURCE_COMPLETE'))
        try:audited_replacement(parent)
        except AssertionError:pass
        else:raise AssertionError('Valid scientific result admitted for replacement')
    assert tasks(PARENT,'LEARNER_TRAIN16')==['T1']
    assert panel_seeds(PARENT,'LEARNER_TRAIN16')==list(range(285108,285124))
    assert panel_seeds(ENDPOINTS[0],'CONFIRM_C')==list(range(285500,285532))
    for point,panel in [(PARENT,'CONFIRM_C'),(ENDPOINTS[0],'LEARNER_TRAIN16'),(ENDPOINTS[0],'CONFIRM_A')]:
        try:tasks(point,panel)
        except AssertionError:pass
        else:raise AssertionError('Old or cross-purpose panel opened')
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
