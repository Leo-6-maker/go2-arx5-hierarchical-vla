"""Actual Core CPU two-process qualification of the declared conic decoder.

Consumes sealed ordinary model predictions. No policy model, expert labels,
GPU, optimizer or physics is loaded; this proof grants no native execution.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
from .storage import digest

PROFILE='TASK_LEARNING_CORE_CONIC_CPU_REPLAY_V3'
DATA='ff94fc77c5a8309dc1460032b308c2f213c96c088449ee1821b3e46e045107b0'
STATS='0faba237b8922ed9bbd53bd6732a9f449f46dcb6fc19017eae8f3cffa3afd9f2'
PACKAGES={
 'A':'9205afbb45328ee40255738e40ba193eee4760cb1de75422e7d1967d15598fe6',
 'B':'aa9e1517d432a0569fd3e7e7dcaff12c9aef197fc40a407c94dfb694d2ef5ff7',
 'C':'0b248af6ffe503c778c43e5f68c03eb9be6266b4c493ff82a9e1668dead8a8d3'}


def identity(array):
    a=np.ascontiguousarray(array)
    return dict(dtype=str(a.dtype),shape=list(a.shape),sha256=hashlib.sha256(a.tobytes()).hexdigest())


def once(spec,output,guard):
    from scripts.pi0292.durable import write
    from .task_learning_conic_decoder_v3 import fit,VERSION,SOLVER_VERSION
    from .task_learning_local_analysis_v1 import geometry
    root=Path(output);root.mkdir(exist_ok=False)
    start=time.monotonic();cpu=time.process_time();rows=[];models={}
    for method in('A','B','C'):
        descriptor=spec['matched_results'][method];file=Path(descriptor['path'])
        if digest(file)!=descriptor['sha256']:raise ValueError('Sealed complete matched result changed')
        source=json.loads(file.read_bytes())
        if not(source['method']==method and source['requested_predictions']==len(source['rows'])==180
            and source['dataset_sha256']==DATA and source['statistics_sha256']==STATS
            and source['last2_package']['sha256']==PACKAGES[method]):
            raise ValueError('Exact complete A/B/C populations and frozen model identities required')
        models[method]=source['last2_package']
        for index,row in enumerate(source['rows']):
            guard();sample=Path(row['sample']['path'])
            if digest(sample)!=row['sample']['sha256']:raise ValueError('Original prediction member changed')
            with np.load(sample,allow_pickle=False)as packet:
                raw=packet['physical_H40'].copy();state=packet['actual_state10'].copy()
            decoded,audit=fit(raw)
            item=dict(method=method,index=index,split=row['split'],phase=row['phase'],noise_group=row['noise_group'],
                episode=row['episode'],anchor=row['anchor'],seed=row['seed'],source_prediction_sha256=row['sample']['sha256'],
                raw_identity=identity(raw),fit=audit,valid=False)
            if decoded is not None:
                envelope=geometry(decoded,state);name=f'{method}-{index:03d}.private.npz'
                np.savez(root/name,raw=raw,decoded=decoded,state=state)
                item.update(envelope=envelope,decoded_identity=identity(decoded),output_file=name,
                    valid=bool(audit['solver_success']and envelope['original_envelope_pass']))
            rows.append(item)
            write(root/'PROGRESS.private.json',dict(completed=len(rows),requested=540,valid=sum(a['valid']for a in rows)))
    result=dict(schema='CORE_CONIC540_ONE_PROCESS_ACTUAL_V3',source_sha=spec['source_sha'],version=VERSION,
        solver_version=SOLVER_VERSION,requested=540,completed=len(rows),valid=sum(a['valid']for a in rows),
        rows=rows,model_identities=models,dataset_sha256=DATA,statistics_sha256=STATS,
        learning_updates=0,model_queries=0,model_loads=0,physics_integrations=0,
        solver_calls=2*len(rows),native_execution_admission=False,
        wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu)
    write(root/'RESULT.private.json',result);return result


def compare(first,second):
    first,second=Path(first),Path(second)
    a=json.loads((first/'RESULT.private.json').read_bytes());b=json.loads((second/'RESULT.private.json').read_bytes())
    if not(a['completed']==b['completed']==a['valid']==b['valid']==540
        and len(a['rows'])==len(b['rows'])==540
        and [(x['method'],x['index'])for x in a['rows']]==[(m,i)for m in('A','B','C')for i in range(180)]
        and a['model_identities']==b['model_identities']and a['source_sha']==b['source_sha']
        and a['solver_version']==b['solver_version']=='0.11.1'):
        raise ValueError('Complete540 valid per process required; failures never filtered')
    records=[]
    keys=('method','index','split','phase','noise_group','episode','anchor','seed','source_prediction_sha256','raw_identity','decoded_identity','valid')
    for x,y in zip(a['rows'],b['rows']):
        if any(x[k]!=y[k]for k in keys):raise ValueError('Different inputs or decoder outputs across fresh Core processes')
        with np.load(first/x['output_file'],allow_pickle=False)as one,np.load(second/y['output_file'],allow_pickle=False)as two:
            if not all(np.array_equal(one[k],two[k])for k in('raw','decoded','state')):
                raise ValueError('Actual saved Core arrays are not exactly reproducible')
        records.append(dict(method=x['method'],index=x['index'],split=x['split'],phase=x['phase'],noise_group=x['noise_group'],
            raw_sha256=x['raw_identity']['sha256'],decoded_sha256=x['decoded_identity']['sha256'],bit_exact=True))
    return dict(schema='CORE_CONIC540_TWO_PROCESS_REPLAY_ACTUAL_V3',status='PASS_ACTUAL_CORE_CPU540_TWICE_EXACT',
        requested_unique=540,actual_conic_fit_calls=1080,actual_solver_calls=2160,
        matched=540,all_bit_exact=True,records=records,source_sha=a['source_sha'],
        model_queries=0,model_loads=0,learning_updates=0,physics_integrations=0,
        native_execution_admission=False,original_cross_hardware_REJECT_preserved=True,
        original_DEV80_TEST=0,total_child_CPU_s=a['CPU_s']+b['CPU_s'])
