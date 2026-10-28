"""Saved-array fitting return, explicit deterministic/stochastic denominators."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
import argparse
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
from quality_worker import BASE,ROOT,RAW,read,sha,write,admitted
from condition_comparison import compare,summarize as compare_summary

sys.path.insert(0,str(BASE/'PI0271/code'))
spec=importlib.util.spec_from_file_location('pi0271_saved_report',BASE/'PI0271/code/report.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
from error_bank import decompose


def direct_replay(path,processor,codec):
    with np.load(path,allow_pickle=False) as z:
        raw=z['raw'];decoded=z['decoded'];q=z['zero_query']
        assert raw.shape==q.shape==(1,40,132) and np.count_nonzero(q)==0
        assert not any(k.startswith('trace_') for k in z.files)
        difference=None
        if np.isfinite(raw).all() and np.isfinite(decoded).all():
            rebuilt=codec.decode(processor,raw[0],z['state']).astype(np.float32)
            difference=float(np.max(abs(rebuilt-decoded[0])));assert difference<=1e-7
    return dict(file=str(path),sha256=sha(path),readout='DIRECT',codec_max_abs=difference,
        saved_zero_latent=True,Euler_steps=0,learned_network_recomputed=False)


def main():
    admitted()
    complete=read(ROOT/'evidence/QUERY_OUTPUTS_COMPLETE.json');assert complete['status']=='ALL_FIXED_PI0272_OUTPUTS_COMPLETE'
    import resource
    resource.setrlimit(resource.RLIMIT_AS,(16*2**30,16*2**30));os.nice(19)
    scope=old.metric_helpers()
    import torch
    import pi0260_r2_codec as codec
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    def forbidden(*a,**kw):raise RuntimeError('PI0272_SAVED_ARRAYS_ONLY_NO_NEURAL_QUERY')
    torch.nn.Module.__call__=forbidden;torch.optim.Optimizer.__init__=forbidden
    assert not torch.cuda.is_available()
    processor=codec.processor(read(ROOT/'contracts/ROT/source/dataset/statistics.json'))
    card=read(ROOT/'design/EVALUATION_CARD.json');support=read(ROOT/'design/FITTING_CARD.json')
    guard=read(BASE/'PI0270/design/NEW_FIT192.json')
    sources={panel:{r['input_id']:r for r in data['rows']} for panel,data in (('support',support),('early_guard',guard))}
    outputs={r['path']:r for r in complete['rows']};jobs=[];cpu=[]
    for q in card['rows']:
        p=Path(q['output']);assert sha(p)==outputs[str(p)]['sha256']
        cpu.append(scope['replay'](p,processor,torch,codec) if q['readout']=='S4FIT' else direct_replay(p,processor,codec))
        if not q['repeatability_only']:jobs.append((q['point'],q['panel'],q['input_id'],q['noise']['j'] if q['noise'] else None,p))
    previous=read(BASE/'PI0271/evidence/QUERY_OUTPUTS_COMPLETE.json')
    reused=[r for r in previous['rows'] if r['point']=='AGG4096' and r['panel'] in ('A1','A2')]
    assert len(reused)==1168
    for r in reused:
        p=Path(r['path']);assert sha(p)==r['sha256'];jobs.append(('AGG4096','support',r['input_id'],r['j'],p))
    for r in guard['rows']:
        record=read(RAW.parent/'PI0270/newfit/AGG4096'/r['input_id']/'record.json')
        assert record['row']==r and sha(record['tensor'])==record['tensor_sha256']
        jobs.append(('AGG4096','early_guard',r['input_id'],'ORIGINAL_NEWFIT',Path(record['tensor'])))
    rows=[];draws={}
    for point,panel,uid,j,path in jobs:
        source=sources[panel][uid];m=source['source_mapping'];assert sha(source['input'])==source['input_sha256']
        with np.load(source['input'],allow_pickle=False) as z:expert=z['raw_action'].copy()
        metrics,decoded,valid=old.measure(path,source['input'],expert,scope)
        result=dict(point=point,panel=panel,input_id=uid,j=j,condition_sha256=source['condition_sha256'],
            **{k:m[k] for k in ('task','command','family','bucket')},**metrics)
        result['prefix_xyz_mse_m2']=float(np.mean(np.sum((decoded[:9,:3]-expert[:9,:3])**2,axis=1))) if valid else None
        if valid:assert abs(result['prefix_xyz_mse_m2']-result['prefix_xyz_rmse_m']**2)<=1e-12
        rows.append(result)
        if panel=='support' and m['bucket']=='ORIGINAL_Q3' and point in ('AGG4096','S4FIT1024'):
            draws.setdefault((point,uid),{})[j]=(decoded,valid,expert)
    assert len(rows)==3504 and len(cpu)==2148
    index={(r['point'],r['panel'],r['input_id'],r['j']):r for r in rows};assert len(index)==len(rows)
    comparisons=[]
    for source in support['rows']:
        uid=source['input_id'];m=source['source_mapping'];direct=index['DIRECT1024','support',uid,None]
        for point in ('AGG4096','S4FIT1024'):
            for k in ((2,8) if m['bucket']=='ORIGINAL_Q3' else (2,)):
                result=compare(direct,[index[point,'support',uid,j] for j in range(k)])
                comparisons.append(dict(before=point,after='DIRECT1024',K=k,**{n:m[n] for n in ('task','command','family','bucket')},**result))
    groups=[]
    for fields in (('before','K','task','bucket'),('before','K','task','command','bucket'),('before','K','task','family','command','bucket')):
        for key in sorted({tuple(r[k] for k in fields) for r in comparisons},key=str):
            own=[r for r in comparisons if tuple(r[k] for k in fields)==key]
            groups.append(dict(grouping='/'.join(fields),**dict(zip(fields,key)),**compare_summary(own)))
    banks=[]
    for (point,uid),values in draws.items():
        assert set(values)==set(range(8)) and all(np.array_equal(v[2],values[0][2]) for v in values.values())
        banks.append(dict(point=point,input_id=uid,**decompose(np.stack([values[j][0][:,:3] for j in range(8)]),values[0][2][:,:3],np.array([values[j][1] for j in range(8)]))))
    assert len(banks)==192
    metrics=('xyz_rmse_m','so3_rmse_rad','prefix_xyz_rmse_m','prefix_so3_rmse_rad','suffix_xyz_rmse_m','suffix_so3_rmse_rad',
        'prefix_endpoint_error_m','first_actual_position_bias_m','direction_cosine','signed_teacher_direction_progress_m','gripper_mae')
    changes=[]
    for before,after in (('DIRECT0','DIRECT1024'),('AGG4096','S4FIT1024')):
        for a in rows:
            if a['point']!=before:continue
            b=index[after,a['panel'],a['input_id'],a['j']]
            assert a['condition_sha256']==b['condition_sha256']
            change=dict(before=before,after=after,**{k:a[k] for k in ('panel','input_id','j','task','command','family','bucket')},
                before_gate=a['gate'],after_gate=b['gate'])
            for k in metrics:
                x,y=a.get(k),b.get(k)
                change['delta_'+k]=y-x if x is not None and y is not None and np.isfinite(x) and np.isfinite(y) else None
            changes.append(change)
    out=RAW/'delivery';out.mkdir(parents=True,exist_ok=True)
    tables={'ALL_POINTS3504.csv':rows,'DIRECT_VS_STOCHASTIC_PER_CONDITION.csv':comparisons,'PER_CONDITION_STRATA.csv':groups,
        'MATCHED_WITHIN_READOUT_LEARNING.csv':changes,
        'QUALITY_STRATA.csv':old.summarize(rows,('point','panel','task','command','bucket'),metrics),
        'QUALITY_FAMILIES.csv':old.summarize(rows,('point','panel','family','task','command','bucket'),metrics)}
    for name,value in tables.items():scope['csv_write'](out/name,value)
    write(out/'K8_BANKS192.json',dict(banks=banks,population_bias_claim=False,reduced_K=False))
    primary=[]
    for point in ('AGG4096','S4FIT1024'):
        own=[r for r in comparisons if r['before']==point and r['K']==2 and r['bucket']=='M3_TERMINAL'];assert len(own)==24
        primary.append(dict(before=point,after='DIRECT1024',**compare_summary(own)))
    write(out/'FITTING_SUMMARY.json',dict(status='FIXED_FITTING_RESULTS_COMPLETE',primary_M3_terminal=primary,
        rows=3504,new_quality_queries=2148,reused_AGG_outputs=1360,direct_repeats=4,
        local_advantage_requires_SO3_gate_and_guard_comparison=True,new_task_qualification=False,blind_TEST=False,
        selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))
    write(out/'CPU_REPLAY.json',dict(status='PASS_PI0272_SAVED_READOUT_DECODE_AND_PHYSICAL_METRICS',predictions=cpu,
        tables={n:dict(rows=len(v),sha256=sha(out/n)) for n,v in tables.items()},shared_original_gate=True,
        independent_FP64_metrics_checked=True,source_files=[dict(path=str(p),sha256=sha(p)) for p in (
            ROOT/'design/EVALUATION_CARD.json',ROOT/'evidence/QUERY_OUTPUTS_COMPLETE.json',BASE/'PI0271/evidence/QUERY_OUTPUTS_COMPLETE.json',Path(__file__))],
        model_queries=0,gradient_queries=0,physics_steps=0,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))
    print('PASS_PI0272_SAVED_READOUT_DECODE_AND_PHYSICAL_METRICS',len(rows),len(cpu),flush=True)


if __name__=='__main__':main()
