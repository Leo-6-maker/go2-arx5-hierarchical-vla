"""R2 fixed local snapshot panels, unchanged native model/physics execution."""
import argparse,ast,hashlib,importlib.util,json,sys,types
from pathlib import Path
import numpy as np
from evaluation_card import COMMON,ROOT,OLD,RAW,OLD_RAW,read,sha,write,freeze,packet,RELEASE

def function(path,name,scope):
    text=path.read_text();node=next(n for n in ast.parse(text).body if getattr(n,'name',None)==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope);return scope[name]

def prepare():
    prior=read(COMMON/'design/EVALUATION_CARD.json');fit=[r for r in prior['fit'] if r['seed'] in range(10000,10004)]
    assert len(fit)==16 and all(sorted(r['group'] for r in fit if r['seed']==s)==[0,1,2,3] for s in range(10000,10004))
    old_noise=read(COMMON/'design/EVALUATION_NOISE.json')['rows'];uids={r['uid'] for r in fit}
    rows=[r for r in old_noise if (r['lane']=='FIT' and r['condition'] in uids and r['index']==0) or (r['lane']=='TRAIN_PHYSICS' and r['condition'] in range(10000,10004)) or (r['lane']=='TEST' and r['condition'] in range(30000,30008))]
    assert len(rows)==16+12*80
    freeze(ROOT/'design/FIT_CARD.json',dict(rows=fit))
    freeze(ROOT/'design/EVALUATION_CARD.json',dict(release=RELEASE,fit=fit,TRAIN4=list(range(10000,10004)),DEV8=list(range(30000,30008)),CONFIRM16=list(range(30032,30048)),
        points=[f'{a}{n}' for a in ('GRIP','FOCUS') for n in (512,1024,2048,4096)],physics_limits=prior['physics_limits'],
        noise_index=0,max_screens=8,max_DEV_candidates=2,confirm_candidates=1,native_episode_s=32))
    freeze(ROOT/'design/EVALUATION_NOISE.json',dict(release=RELEASE,rows=rows,old_card_sha256=sha(COMMON/'design/EVALUATION_NOISE.json'),confirmation='LOCKED_UNTIL_ONE_FROZEN_CANDIDATE'))
    (ROOT/'contracts/TASK/source/dataset').mkdir(parents=True,exist_ok=True)
    stats=ROOT/'contracts/TASK/source/dataset/statistics.json'
    if not stats.exists():stats.write_bytes((COMMON/'contracts/TASK/source/dataset/statistics.json').read_bytes())
    for name in ('ManiSkill','project-src'):
        p=COMMON/name
        if not p.exists():p.symlink_to(OLD/name,target_is_directory=True)
    p=COMMON/'design/SOURCE_FILESET.json'
    if not p.exists():p.write_bytes((OLD/'design/SOURCE_FILESET.json').read_bytes())
    (RAW/'evidence').mkdir(parents=True,exist_ok=True)
    shared=RAW.parent/'evidence/budget.jsonl';shared.parent.mkdir(parents=True,exist_ok=True);shared.touch(exist_ok=True)
    link=RAW/'evidence/budget.jsonl'
    if not link.exists():link.symlink_to(shared)
    assert link.resolve()==shared.resolve()
    collision=[str(p) for base in (OLD_RAW,RAW.parent) for p in (base/'episodes').glob('*') if any(f'-{seed}-' in p.name for seed in range(30032,30048))]
    freeze(ROOT/'evidence/PANEL_FREEZE.json',dict(status='PASS_FIXED_FIT16_TRAIN4_DEV8',card_sha256=sha(ROOT/'design/EVALUATION_CARD.json'),noise_sha256=sha(ROOT/'design/EVALUATION_NOISE.json'),
        inherited_quality_queries=128,inherited_quality_heads=512,confirmation_collisions=collision,new_model_queries=0,new_physics_steps=0))
    assert not collision,'Confirmation namespace is already used'

def admit(point):
    model=RAW/'models'/point;ready=read(model/'READY.json')
    assert point in read(ROOT/'design/EVALUATION_CARD.json')['points']
    assert ready['point']==point and ready['release']==RELEASE and not ready['resumable']
    for r in ready['files']:assert (model/r['path']).stat().st_size==r['bytes'] and sha(model/r['path'])==r['sha256']
    binding=dict(schema='PI0277_R2_LOCAL_MODEL_ONLY',point=point,view=str(model),view_type='model-only',source_revision='LOCAL@'+sha(model/'READY.json'),
        checkpoint_tree_sha256=hashlib.sha256(json.dumps(ready['files'],sort_keys=True).encode()).hexdigest(),effective_contract='GRIP',
        readout_contract_sha256=sha(model/'PI0277_READOUT_CONTRACT.json'),statistics_sha256=sha(ROOT/'contracts/TASK/source/dataset/statistics.json'),
        model_files_sha256={r['path']:r['sha256'] for r in ready['files'] if r['path'] in ('config.json','model.safetensors.index.json') or r['path'].endswith('.safetensors')})
    freeze(ROOT/'model_bindings'/f'{point}.json',binding)
    paths=list((ROOT/'code').glob('*.py'))+[ROOT/'design/EVALUATION_CARD.json',ROOT/'design/FIT_CARD.json',ROOT/'contracts/TASK/source/dataset/statistics.json',OLD/'code/source_worker.py',OLD/'code/evaluate.py',OLD/'code/replay.py']
    seal=ROOT/'design/EVALUATION_FILESET.json'
    if not seal.exists():freeze(seal,dict(files=[dict(path=str(p),sha256=sha(p)) for p in paths],algorithm='UNCHANGED_S4_BF16_FP32_DECODE_NATIVE_PHYSICS'))
    for r in read(seal)['files']:assert sha(r['path'])==r['sha256']
    freeze(ROOT/'evidence'/f'POLICY_ADMISSION_{point}.json',dict(status='PASS_PI0277_FIXED_POLICY_IDENTITY',point=point,
        model_binding_sha256=sha(ROOT/'model_bindings'/f'{point}.json'),evaluation_card_sha256=sha(ROOT/'design/EVALUATION_CARD.json'),archive='DEFERRED_USER_PRIORITY'))

def runtime(point,lane,seeds):
    run=types.ModuleType('run');run.ROOT=COMMON;run.sha=sha;run.write=write;sys.modules['run']=run
    spec=importlib.util.spec_from_file_location('pi0277_r2_native',OLD/'code/source_worker.py');source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
    source.ROOT=COMMON;source.RAW=RAW;source.LIMITS=read(ROOT/'design/EVALUATION_CARD.json')['physics_limits']
    base=source.Episode
    class Episode(base):
        def __init__(self,slot,seed,mode):
            assert seed in seeds and slot==f'{lane}-{seed}-{point}'
            starts=list((RAW/'episodes').glob('*/attempt-*/START.json'))
            assert len(starts)-len({p.parent.parent.name for p in starts})<8
            super().__init__(slot,seed,mode)
    def check_initial(ep,seed,lane):
        if lane=='CONFIRM':
            permit=read(ROOT/'evidence/FROZEN_CANDIDATE.json');assert permit['point']==point
            write(ep.folder/'INITIAL_SEEDED_NATIVE_RESET.json',dict(seed=seed,native_inventory_sha256=sha(ep.folder/'NATIVE_INVENTORY.json'),expert_not_required_by_R2=True));return
        slot=f'{"TEST" if lane=="TEST" else "TRAIN"}-{seed}-eef'
        candidates=[p for p in (OLD_RAW/'episodes'/slot).glob('attempt-*/RESULT.json') if read(p)['status'].startswith('VALID')];assert len(candidates)==1
        with np.load(candidates[0].with_name('initial.npz')) as z:initial={k:z[k] for k in z.files}
        actual=source.flatten(ep.initial);assert initial.keys()==actual.keys()
        differences={k:float(np.max(np.abs(initial[k]-actual[k]))) for k in actual};assert max(differences.values())<=1e-6
        write(ep.folder/'INITIAL_FIXED_PHYSICS_MATCH.json',dict(parent=str(candidates[0]),differences=differences))
    text=(OLD/'code/evaluate.py').read_text();node=next(n for n in ast.parse(text).body if getattr(n,'name',None)=='policy_episode')
    block=next(n for n in node.body if isinstance(n,ast.Try)).body
    assert isinstance(block[0],ast.Assign) and isinstance(block[1],ast.With) and isinstance(block[2],ast.Assert) and isinstance(block[3],ast.Assign) and isinstance(block[4],ast.Assert) and isinstance(block[5],ast.Expr)
    block[:6]=ast.parse('check_initial(ep,seed,lane)').body
    scope=dict(ROOT=ROOT,RAW=RAW,np=np,json=json,write=write,retained=source.retained,Episode=Episode,check_initial=check_initial,
        native_action=source.native_action,pose_gate=source.pose_gate,matrix=source.matrix,flatten=source.flatten)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(OLD/'code/evaluate.py'),'exec'),scope)
    return scope['policy_episode']

def scores(point,lane,seeds):
    reconstruct=function(OLD/'code/replay.py','reconstruct',dict(np=np));rows=[]
    for seed in seeds:
        candidates=[p for p in (RAW/'episodes'/f'{lane}-{seed}-{point}').glob('attempt-*/RESULT.json') if read(p)['status'].startswith('VALID')];assert len(candidates)==1
        folder=candidates[0].parent;rec=read(candidates[0])
        for name,h in rec['files'].items():assert sha(folder/name)==h
        with np.load(folder/'physics.npz') as z:
            native=reconstruct(z)
            for k,v in native.items():assert np.array_equal(v,z[k]),k
            success=native['success'][::5];persistent=bool(len(success)>=601 and success[560:601].all());assert persistent==rec['persistent_success']
            grasp=native['is_cubeA_grasped'];lift=z['cubeA_pose'][:,2]-z['cubeA_pose'][0,2]>=.02
            def runs(v):
                boundaries=np.diff(np.r_[False,v,False].astype(int));return (np.flatnonzero(boundaries==-1)-np.flatnonzero(boundaries==1))*.01
            first=lambda v:float(np.flatnonzero(v)[0]*.01) if v.any() else None
            full=rec['status']=='VALID_MODEL_PHYSICS_COMPLETE' and rec['control_calls']==640
            rows.append(dict(point=point,lane=lane,seed=seed,path=str(folder),status=rec['status'],interface_valid=full,
                persistent_success=persistent and full,first_success_s=rec['first_success_s'],first_grasp_s=first(grasp),longest_grasp_s=float(max(runs(grasp),default=0)),
                sustained_lift_2cm=bool(any(runs(lift)>=.3-1e-9)),first_lift_2cm_s=first(lift),on_top=bool(native['is_cubeA_on_cubeB'].any()),
                release_after_grasp=bool((grasp[:-1]&~grasp[1:]).any()),IK_holds=rec['ik_holds'],control_calls=rec['control_calls'],result_sha256=sha(candidates[0])))
    result=dict(point=point,lane=lane,rows=rows,denominator=len(rows),persistent_stacks=sum(r['persistent_success'] for r in rows),
        on_top_cases=sum(r['on_top'] for r in rows),sustained_lifts=sum(r['sustained_lift_2cm'] for r in rows),new_model_queries=0,new_physics_steps=0,
        score_definition='Original native predicate; all41control samples28..30s; full32s interface-valid execution',selected_checkpoint='NONE',deployment_selected_checkpoint='NONE')
    freeze(ROOT/'evidence'/f'PANEL_{point}_{lane}.json',result);return result

def panel(point,lane):
    from client import Server
    card=read(ROOT/'design/EVALUATION_CARD.json');seeds=card[{'TRAIN_PHYSICS':'TRAIN4','TEST':'DEV8','CONFIRM':'CONFIRM16'}[lane]]
    if lane=='TEST':
        assert read(ROOT/'evidence'/f'CPU_REPLAY_{point}_TRAIN_PHYSICS.json')['task']['persistent_stacks']>=3
        started=list((ROOT/'evidence').glob('DEV_ADMISSION_*.json'))
        existing=ROOT/'evidence'/f'DEV_ADMISSION_{point}.json'
        if not existing.exists():
            assert len(started)<2
            lineage='GRIP' if point.startswith('GRIP') else 'FOCUS'
            assert not any(read(p)['lineage']==lineage for p in started),'First qualifying candidate per lineage'
            freeze(existing,dict(point=point,lineage=lineage,TRAIN_CPU_receipt_sha256=sha(ROOT/'evidence'/f'CPU_REPLAY_{point}_TRAIN_PHYSICS.json')))
    if lane=='CONFIRM':assert read(ROOT/'evidence/FROZEN_CANDIDATE.json')['point']==point
    rows=read(ROOT/'design/EVALUATION_NOISE.json')['rows'];noises={(r['lane'],r['condition'],r['index']):r for r in rows}
    run=runtime(point,lane,seeds);server=Server(point)
    try:
        if lane=='TRAIN_PHYSICS':
            results=[]
            for row in card['fit']:
                folder=RAW/'quality'/point/'FIT'/row['uid']/'n0';server.query(packet(row),noises['FIT',row['uid'],0],folder)
                results.append(dict(uid=row['uid'],output=str(folder/'output.npz'),sha256=sha(folder/'output.npz')))
            freeze(ROOT/'evidence'/f'FIT16_{point}.json',dict(results=results,point=point))
        for seed in seeds:
            run(seed,lane,server,noises);scores_so_far=RAW/'episodes'/f'{lane}-{seed}-{point}'
            print('PHYSICAL_CASE_RETAINED',point,lane,seed,flush=True)
    finally:server.close()
    print(json.dumps(scores(point,lane,seeds)),flush=True)

if __name__=='__main__':
    import fcntl
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','admit','TRAIN_PHYSICS','TEST','CONFIRM'));p.add_argument('--point');a=p.parse_args()
    (COMMON/'runtime').mkdir(exist_ok=True)
    with (COMMON/'runtime/quality_owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.stage=='prepare':prepare()
        elif a.stage=='admit':admit(a.point)
        else:panel(a.point,a.stage)
