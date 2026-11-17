"""Bounded TRAIN/DEV source H50 preparation; TEST labels stay unread."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from go2arx5_vla.sla_adapter.source_native_v6 import load_numeric,SCHEMA,COORDINATES
from scripts.sla.robodojo_source_audit_v1 import read,write,sha

SEED=20261112


def visit_bank(episodes,total=32000):
    """Uniform task/episode slots; global dynamic/static quota exactly half.

    Forced strata are retained; eligible flexible slots fill the remaining
    quota by a fixed cursor hash. Impossible quotas are reported, not faked.
    """
    tasks=sorted({e['task']for e in episodes});by={t:[e for e in episodes if e['task']==t]for t in tasks}
    starts={};offset=0
    for e in episodes:starts[e['id']]=offset;offset+=e['windows']
    pools={e['id']:{p:np.flatnonzero(np.asarray(e['_dynamic'])==bool(p))for p in(0,1)}for e in episodes}
    chosen=[];forced_static=[];forced_dynamic=[];flexible=[]
    for cursor in range(total):
        task=tasks[cursor%len(tasks)];ep=by[task][(cursor//len(tasks))%len(by[task])];chosen.append(ep)
        available=[p for p in(0,1)if len(pools[ep['id']][p])]
        if available==[0]:forced_static.append(cursor)
        elif available==[1]:forced_dynamic.append(cursor)
        elif len(available)==2:flexible.append(cursor)
        else:raise ValueError('SOURCE_EPISODE_HAS_NO_REAL_ANCHORS')
    required=total//2-len(forced_static)
    if required<0 or required>len(flexible):
        return None,dict(ready=False,reason='GLOBAL_50_50_INCOMPATIBLE_WITH_UNIFORM_TASK_EPISODE_SLOTS',
            forced_static=len(forced_static),forced_dynamic=len(forced_dynamic),flexible=len(flexible))
    ranked=sorted(flexible,key=lambda i:hashlib.sha256((str(SEED)+'/stratum/'+str(i)).encode()).digest())
    static=set(forced_static+ranked[:required]);visits=[];rows=[]
    for cursor,e in enumerate(chosen):
        phase=0 if cursor in static else 1;pool=pools[e['id']][phase]
        rng=np.random.Generator(np.random.PCG64(SEED+8000000+cursor));anchor=int(pool[int(rng.integers(len(pool)))])
        visits.append(starts[e['id']]+anchor);rows.append(dict(cursor=cursor,task=e['task'],episode=e['id'],anchor=anchor,dynamic=bool(phase)))
    return np.asarray(visits,dtype='<i8'),dict(ready=True,rows=rows,dynamic=sum(x['dynamic']for x in rows),static=sum(not x['dynamic']for x in rows),
        distinct_windows=len({(x['episode'],x['anchor'])for x in rows}),by_episode={e['id']:sum(x['episode']==e['id']for x in rows)for e in episodes},
        by_task={t:sum(x['task']==t for x in rows)for t in tasks})


def evaluation_bank(episodes,split):
    tasks=sorted({e['task']for e in episodes});groups={}
    for task in tasks:
        ep=[e for e in episodes if e['task']==task];selected=[];used=set()
        for phase in(1,0):
            pool=[(e,i)for e in ep for i,v in enumerate(e['_dynamic'])if bool(v)==bool(phase)]
            pool.sort(key=lambda x:hashlib.sha256((str(SEED)+'/eval/'+split+'/'+x[0]['id']+'/'+str(x[1])).encode()).digest())
            for e,i in pool[:8]:selected.append(dict(episode=e['id'],anchor=i,dynamic=bool(phase),task=task));used.add((e['id'],i))
        if len(selected)<16:
            pool=[(e,i)for e in ep for i in range(e['windows'])if(e['id'],i)not in used]
            pool.sort(key=lambda x:hashlib.sha256((str(SEED)+'/fill/'+split+'/'+x[0]['id']+'/'+str(x[1])).encode()).digest())
            for e,i in pool[:16-len(selected)]:selected.append(dict(episode=e['id'],anchor=i,dynamic=bool(e['_dynamic'][i]),task=task))
        if len(selected)!=16:raise ValueError('DISTINCT_SOURCE16_PER_TASK_EVAL_REQUIRED')
        groups[task]=selected
    return [dict(groups[t][i],id=split+'-'+t+'-%02d'%i)for i in range(16)for t in tasks]


def prepare(spec):
    root=Path(spec['registered_root']);start=time.monotonic();cpu=time.process_time();pin=spec['source_map']
    if spec['plan_commit']!='ae3d3c0dceade437b0104a150cd0402fef650052'or spec['new_model_generations_budget']or spec['optimizer_updates']:
        raise PermissionError('V6_TYPED_SOURCE_CPU_PREPARATION_ONLY')
    if sha(pin['path'])!=pin['sha256']:raise ValueError('COMPLETE60_SOURCE_MAP_CHANGED')
    source=read(pin['path']);assert len(source['records'])==60
    output=root/'dataset';output.mkdir();(output/'labels').mkdir();episodes=[];test=[];fit={};failures=[]
    for i,record in enumerate(source['records']):
        row=record['source'];split=row['proposed_split']
        if split=='SOURCE_TEST':
            test.append(dict(source=row,raw_path=record['raw_path'],labels_read=False));continue
        if time.time()>=spec['deadline_epoch']-120 or(root/'STOP.json').exists():raise TimeoutError('FINITE_SOURCE_PREPARATION_DEADLINE_OR_STOP')
        ep=dict(id=row['task']+'-'+Path(row['path']).stem,task=row['task'],split=split,raw_path=record['raw_path'],raw_sha256=row['sha256'],valid=False)
        write(root/'PROGRESS.private.json',dict(epoch=time.time(),phase='NATIVE_TRAIN_DEV_NUMERIC_CONVERSION',completed=len(episodes),current=ep['id'],TEST_labels_read=False))
        try:
            if sha(record['raw_path'])!=row['sha256']:raise ValueError('SOURCE_BYTE_PIN_CHANGED')
            state,actions,text,audit=load_numeric(record['raw_path']);ep.update(valid=True,windows=len(state),instruction=text,audit=audit)
            path=output/'labels'/(ep['id']+'.npz');np.savez(path,state=state,actions=actions)
            ep['arrays']=dict(file=str(path.relative_to(output)),sha256=sha(path));fit[ep['id']]=(state,actions)
        except (ValueError,KeyError)as error:
            ep['invalid_reason']=str(error);failures.append(dict(episode=ep['id'],reason=str(error)))
        episodes.append(ep)
    train=[e for e in episodes if e['valid']and e['split']=='TRAIN'];dev=[e for e in episodes if e['valid']and e['split']=='SOURCE_DEV'];tasks=sorted({r['source']['task']for r in source['records']})
    counts={t:dict(TRAIN_episodes=sum(e['task']==t for e in train),TRAIN_windows=sum(e['windows']for e in train if e['task']==t),DEV_episodes=sum(e['task']==t for e in dev))for t in tasks}
    minimum=len(train)>=24 and len(tasks)==6 and all(v['TRAIN_episodes']>=4 and v['TRAIN_windows']>=64 and v['DEV_episodes']>=2 for v in counts.values())
    if not minimum:return dict(exit_code=0,status='SOURCE_DATA_COVERAGE_NOT_READY',ready=False,counts=counts,failures=failures,TEST_labels_read=False,new_updates=0,new_generations=0,new_native_requests=0)
    k=min(64,min(e['windows']for e in train));states=[];actions=[];ids=[]
    for e in train:
        selected=np.linspace(0,e['windows']-1,k,dtype=int);s,a=fit[e['id']];states.append(s[selected]);actions.append(a[selected].reshape(-1,14));ids.extend(dict(episode=e['id'],anchor=int(i))for i in selected)
    state_values=np.concatenate(states);action_values=np.concatenate(actions);stats={}
    for name,values in(('state',state_values),('actions',action_values)):
        quant=np.quantile(values,[.01,.25,.75,.99],axis=0);stats[name]=dict(q01=quant[0].tolist(),q99=quant[3].tolist(),IQR=(quant[2]-quant[1]).tolist(),
            dimensions=14,epsilon=1e-6,implicit_clip=False,units='SOURCE_RECORDED_NATIVE_NOT_HARDWARE_CALIBRATED')
    stats.update(schema=SCHEMA,coordinates=COORDINATES,fit_split='TRAIN_ONLY',fit_episodes=[e['id']for e in train],fit_windows=ids,equal_windows_per_episode=k,source_TEST_read=False)
    write(output/'STATISTICS.json',stats);threshold=.01*np.asarray(stats['actions']['IQR']);fingerprints={}
    for ep in episodes:
        if not ep['valid']:continue
        s,a=fit[ep['id']];baseline=np.zeros_like(s);baseline[:,6]=s[:,6];baseline[:,13]=s[:,13]
        dynamic=np.any(np.max(abs(a-baseline[:,None,:]),axis=1)>threshold,axis=1)
        path=output/ep['arrays']['file'];np.savez(path,state=s,actions=a,dynamic=dynamic);ep['arrays']['sha256']=sha(path);ep['dynamic_windows']=int(dynamic.sum());ep['static_windows']=int((~dynamic).sum());ep['_dynamic']=dynamic
        fp=hashlib.sha256(s.tobytes()+a.tobytes()).hexdigest();ep['numeric_fingerprint']=fp
        if fp in fingerprints and fingerprints[fp]['split']!=ep['split']:raise ValueError('EXACT_NUMERIC_TRAIN_DEV_DUPLICATE_NOT_ALLOWED')
        fingerprints[fp]=ep
    visits,visits_audit=visit_bank(train)
    if visits is None:
        for e in episodes:e.pop('_dynamic',None)
        write(root/'COVERAGE.private.json',dict(episodes=episodes,sampling=visits_audit,statistics_sha256=sha(output/'STATISTICS.json')))
        return dict(exit_code=0,status='SOURCE_STRATUM_DATA_COVERAGE_NOT_READY',ready=False,counts=counts,sampling=visits_audit,TEST_labels_read=False,new_updates=0,new_generations=0,new_native_requests=0)
    train_bank=evaluation_bank(train,'TRAIN');dev_bank=evaluation_bank(dev,'SOURCE_DEV');write(output/'EVAL_BANK.private.json',dict(TRAIN96=train_bank,SOURCE_DEV96=dev_bank,restore16=train_bank[:16],source_TEST_unopened=True))
    np.save(output/'VISITS32000.npy',visits,allow_pickle=False);write(output/'VISITS_AUDIT.private.json',visits_audit)
    for e in episodes:e.pop('_dynamic',None)
    manifest=dict(schema=SCHEMA,coordinates=COORDINATES,horizon=50,nominal_frequency_Hz=30,action_dimensions=14,padding_dimensions=18,label_kind='FUTURE_LOGGED_STATE_REFERENCE',
        current_observations_only=True,first_future_state_offset_frames=1,physical_calibration='UNKNOWN',source_map=pin,HF_revision=source['revision'],episodes=episodes,source_TEST=test,statistics=dict(file='STATISTICS.json',sha256=sha(output/'STATISTICS.json')),
        visit_map=dict(file='VISITS32000.npy',sha256=sha(output/'VISITS32000.npy')),evaluation_bank=dict(file='EVAL_BANK.private.json',sha256=sha(output/'EVAL_BANK.private.json')),
        dynamic_threshold_native=threshold.tolist(),dynamic_rule='ANY_FUTURE_DIM_CHANGE_FROM_CURRENT_EXCEEDS_1PCT_TRAIN_ACTION_IQR',
        split_claim='EPISODE_HOLDOUT_ONLY_UNKNOWN_SESSION_SCENE',TEST_labels_read=False)
    write(output/'MANIFEST.private.json',manifest)
    return dict(exit_code=0,status='TYPED_SOURCE_H50_READY_FOR_S1000',ready=True,plan_commit=spec['plan_commit'],source_sha=spec['source_sha'],dataset=dict(path=str(output),manifest_sha256=sha(output/'MANIFEST.private.json')),statistics_sha256=sha(output/'STATISTICS.json'),
        counts=counts,fit_episodes=len(train),dynamic_visits=visits_audit['dynamic'],static_visits=visits_audit['static'],distinct_training_windows=visits_audit['distinct_windows'],invalid_episodes=failures,
        TEST_labels_read=False,new_updates=0,new_generations=0,new_native_requests=0,wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu)


def main(spec):
    import traceback
    from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close
    try:result=prepare(spec)
    except BaseException:result=dict(exit_code=1,status='PRESERVED_SOURCE_PREP_TECHNICAL_FAILURE',failure=traceback.format_exc(),new_updates=0,new_generations=0,new_native_requests=0)
    write(Path(spec['registered_root'])/'RESULT.private.json',result);close(spec,result['exit_code']);return result['exit_code']
