"""Fixed EEF10 source replay/evaluation; original physics, gate and task predicate."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np

BASE=Path('/home/liuyu-nfs/Projects/go2-arx5')
ROOT=BASE/'PI0282'
OUT=ROOT/'source'
RELEASE='8fdaab73b8b6686409317d9562833235c2839acc'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def guard(point,panel):
    card=json.loads((ROOT/'design/SOURCE_CARD.json').read_text())
    receipt=json.loads((ROOT/'design/IMPLEMENTATION.json').read_text())
    assert card['release']==RELEASE and receipt['release']==RELEASE
    assert receipt['source_card_sha256']==digest(ROOT/'design/SOURCE_CARD.json')
    for name,sha in card['code_sha256'].items():assert digest(ROOT/'code'/name)==sha,name
    for binding in card['predecessors'].values():assert digest(binding['path'])==binding['sha256']
    assert receipt['expert_codec_replay_allowed'] is True
    if point!='EXPERT':
        assert receipt['model_evaluation_allowed'] is True
        proof=OUT/'evidence/RESULT_EXPERT_TRAIN.json'
        assert receipt['expert_codec_result_sha256']==digest(proof)
        expert=json.loads(proof.read_text())
        assert {row['condition'] for row in expert['rows']}==set(range(10000,10004))
        assert len(expert['rows'])==4 and all(row['status']=='VALID_SOURCE_COMPLETE' and row['task_control_calls']==640 and row['persistent_success'] and row['persistent_pass_samples']==41 for row in expert['rows'])
    assert point in card['points'] and panel in card['panels']
    if point=='EXPERT':assert panel=='TRAIN'
    if panel=='DEV':assert point in card['development_points']
    if panel=='CONFIRM':
        frozen=json.loads((ROOT/'design/SOURCE_CANDIDATE.json').read_text())
        assert frozen['point']==point and frozen['confirm_allowed'] is True
        assert set(frozen['bindings'])=={'model','processor','statistics','mapping','decode'}
        assert frozen['bindings']==card['point_bindings'][point]
        for binding in frozen['bindings'].values():assert digest(binding['path'])==binding['sha256']
        exposure=json.loads(Path(frozen['exposure_audit_path']).read_text())
        assert digest(frozen['exposure_audit_path'])==frozen['exposure_audit_sha256']
        assert exposure['confirm_allowed'] is True
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    return card

def setup():
    from authority import load
    limits=load(ROOT)
    sys.path.insert(0,str(BASE/'PI0276/code'))
    import source_worker as sw
    sw.RAW=OUT;sw.LIMITS=dict(starts=100,control=72000,physics=360000,rgb=72000,teacher=8192);sw.BUDGET_COUNTS=None
    return sw

def step(sw,ep,command):
    from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix
    assert ep.control<660 and np.asarray(command).shape==(10,)
    target=np.eye(4);target[:3,3]=command[:3];target[:3,:3]=rot6d_rows_to_matrix(command[None,3:9])[0]
    action=sw.native_action(target,ep.root,float(command[-1]))
    ep.ledger.reserve('control',ep.slot);ep.ledger.sync();ep.observe()
    ep.commands.append(command.copy());ep.phase.append(-1)
    ep.env.step(action);ep.control+=1
    installed=dict(action=action,arm_target=sw.array(ep.arm._target_qpos)[0],base_target=sw.matrix(ep.arm._target_pose))
    q=sw.array(ep.env.agent.robot.get_qpos())[0];q[:7]=installed['arm_target'];installed['installed_target_fk']=ep.command_fk(q)
    assert np.max(abs(installed['base_target']-np.linalg.inv(ep.root)@target))<1e-5
    ep.installed.append(installed)

def reply(server,key):
    while True:
        line=server.stdout.readline();assert line,'Resident worker ended before response'
        try:value=json.loads(line)
        except json.JSONDecodeError:continue
        if key in value:return value

def episode(sw,server,card,point,panel,seed):
    slot=point+'-'+panel+'-'+str(seed)
    old=sw.retained(slot)
    if old:return old[1]
    from authority import allowed
    allowed(ROOT,'episode',reserve_s=card['conservative_episode_reserve_s'])
    attempts=list((OUT/'episodes'/slot).glob('attempt-*'))
    allowed=card.get('technical_replacements',{}).get(slot,[])
    assert len(attempts)==len(allowed),'Owner must audit an engineering attempt before a same-condition replacement'
    assert sum(len(v) for v in card.get('technical_replacements',{}).values())<=8
    for attempt,authorization in zip(sorted(attempts),allowed):
        proof=attempt/'RESULT.json';assert digest(proof)==authorization['result_sha256']
        assert json.loads(proof.read_text())['status']=='ENGINEERING_INCOMPLETE'
    ep=sw.Episode(slot,seed,'pd_ee_pose');queries=[];status='VALID_SOURCE_COMPLETE'
    try:
        reference=None
        if point=='EXPERT':
            parent=Path(card['expert_parents'][str(seed)]['path'])
            for name,sha in card['expert_parents'][str(seed)]['files'].items():assert digest(parent/name)==sha
            with np.load(parent/'commands.npz') as data:reference=data['action'].copy()
            assert reference.shape==(640,10)
            with np.load(parent/'initial.npz') as data:initial={k:data[k].copy() for k in data.files}
            actual=sw.flatten(ep.initial);assert initial.keys()==actual.keys()
            errors={k:float(np.max(abs(initial[k]-actual[k]))) for k in initial};assert max(errors.values())<=1e-6,errors
            write(ep.folder/'INITIAL_PAIRED_MATCH.json',errors)
        hold=ep.observe()['state_v1'].copy();hold[-1]=1.
        for _ in range(20):step(sw,ep,hold)
        for index in range(80):
            allowed(ROOT,'query',reserve_s=2)
            observation=ep.observe();state=observation['state_v1'].copy()
            folder=ep.folder/('q%03d'%index);folder.mkdir()
            inp=folder/'input.npz';out=folder/'output.npz'
            arrays=dict(actual_state=state,base_raw=observation['base_camera_raw'],wrist_raw=observation['hand_camera_raw'])
            if reference is not None:arrays['commands']=reference[np.minimum(np.arange(index*8,index*8+40),639)]
            np.savez_compressed(inp,**arrays)
            request=dict(input=str(inp),output=str(out),point=point,panel=panel,condition=seed,query_index=index)
            server.stdin.write(json.dumps(request)+'\n');server.stdin.flush();assert reply(server,'output')['output']==str(out)
            query=dict(index=index,time_s=index*.4,input_sha256=digest(inp),output_sha256=digest(out));queries.append(query)
            with np.load(out) as data:
                commands=data['decoded'].copy();decode_error=str(data['decode_error']) if 'decode_error' in data else ''
            assert commands.shape==(40,10)
            if decode_error:status='VALID_NATIVE_DECODE_REJECT';query['decode_error']=decode_error;break
            if not np.isfinite(commands).all():status='VALID_NONFINITE_REJECT';break
            try:gate=sw.pose_gate(commands,state,ep.control*.05,slot+'-'+str(index))
            except ValueError as exc:
                status='VALID_DEGENERATE_ROTATION_REJECT';query['rotation_error']=str(exc);break
            query['gate']=gate;write(folder/'GATE.json',gate)
            if gate['decision']=='REJECT':status='VALID_SOURCE_GATE_REJECT';break
            if (commands[:,-1]<-.1).any() or (commands[:,-1]>1.1).any():status='VALID_GRIPPER_ENVELOPE_REJECT';break
            for command in commands[:8]:step(sw,ep,command)
        success=np.asarray([x['success'] for x in ep.states])[::5][20:]
        persistent=bool(status=='VALID_SOURCE_COMPLETE' and ep.control==660 and len(success)==641 and success[560:601].all())
        write(ep.folder/'QUERIES.json',queries);result=ep.finish(status)
        result.update(point=point,panel=panel,condition=seed,folder=str(ep.folder),queries=len(queries),persistent_success=persistent,
            task_control_calls=max(0,ep.control-20),persistent_observed_samples=int(max(0,min(len(success),601)-560)),
            persistent_pass_samples=int(success[560:601].sum()),first_success_s=float(np.flatnonzero(success)[0]*.05) if success.any() else None,
            NON_REALTIME=True,domain='SOURCE_PANDA_EEF10_NOT_TARGET_WBC',selected='NONE',deployment='NONE')
        write(ep.folder/'RESULT.json',result);return result
    except BaseException as exc:
        write(ep.folder/'QUERIES.json',queries);ep.finish('ENGINEERING_INCOMPLETE',repr(exc));raise

def run(point,panel):
    card=guard(point,panel);sw=setup();rows=[]
    command=card['codec_command'] if point=='EXPERT' else card['policy_command']+['--point',point]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if point=='EXPERT' else '0',PYTHONPATH=card['native_pythonpath'],HF_HOME=card['hf_home'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    runtime=OUT/'runtime';runtime.mkdir(parents=True,exist_ok=True)
    with (runtime/(point+'-'+panel+'-worker.log')).open('ab') as log:
        server=subprocess.Popen(command,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
        try:
            assert reply(server,'ready')['ready']
            for seed in card['panels'][panel]:
                rows.append(episode(sw,server,card,point,panel,seed));write(OUT/'evidence'/('PROGRESS_'+point+'_'+panel+'.json'),dict(rows=rows))
        finally:
            server.stdin.close();code=server.wait(timeout=120);assert code==0
    write(OUT/'evidence'/('RESULT_'+point+'_'+panel+'.json'),dict(status='FIXED_PANEL_EXECUTED',rows=rows,persistent_count=sum(r['persistent_success'] for r in rows)))

def self_check():
    assert all(np.array_equal(np.arange(i*8,i*8+8),np.arange(640)[i*8:i*8+8]) for i in range(80))
    assert np.minimum(np.arange(632,672),639).tolist()==list(range(632,640))+[639]*32
    success=np.zeros(641,dtype=bool);success[560:601]=True
    assert success[560:601].all() and len(success[560:601])==41
    success[600]=False;assert not success[560:601].all()

if __name__=='__main__':
    if sys.argv[1]=='self-check':self_check();print('SOURCE_20HZ_PREFIX_PERSISTENT_CPU_PASS')
    else:assert sys.argv[1]=='run';run(sys.argv[2],sys.argv[3])
