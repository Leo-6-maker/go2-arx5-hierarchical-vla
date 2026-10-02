"""Cached TRAIN arithmetic and complete saved offline traces. No neural calls."""
from cpu_common import *
from audit_closed import matrices,angle
from collections import Counter
import io
import time
import resource
import numpy as np


def main():
    restrict();started=time.monotonic()
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    def forbidden(*a,**kw):raise RuntimeError('PI0268_NO_NEURAL_MODEL_OR_OPTIMIZER')
    torch.nn.Module.__call__=forbidden;torch.optim.Optimizer.__init__=forbidden
    import pi0260_r2_codec as codec
    from scipy.spatial.transform import Rotation
    stat=read(SOURCE/'actual_statistics.json');proc=codec.processor(stat)
    with np.load(PREP/'cache/train_numeric.npz') as z:s=z['state'];a=z['action'];rel=z['relative'];ns=z['normalized_state'];na=z['normalized_action']
    meta=read(OUT/'TRAIN_NUMERIC_INDEX.json');consumed=read(OUT/'CONSUMED_CONDITION_INDEX.json')
    cast=torch.from_numpy(ns).to(torch.bfloat16).float().numpy()
    sp=proc.norm_params['new_embodiment']['state']['eef_9d'];back=codec.unnormalize(cast[:,:9],sp)
    state_cast=dict(max_xyz_m=float(np.linalg.norm(back[:,:3]-s[:,:3],axis=1).max()),max_so3_rad=float(angle(matrices(back[:,3:9]),matrices(s[:,3:9])).max()))
    sensitivity=[]
    # Fixed metadata representatives: first source in each task/command/variant cell.
    chosen=[next(i for i,m in enumerate(meta) if (m['task'],m['command'],m['variant'])==(t,c,v)) for t in ('M1','M3') for c in (0,1) for v in range(7)]
    for i in chosen:
        for magnitude in (1e-7,1e-6):
            for kind in ('translation','world_left_rotation'):
                for axis in range(3):
                    changed=s[i].astype(float).copy();vec=np.eye(3)[axis]*magnitude
                    if kind=='translation':changed[:3]+=vec
                    else:changed[3:9]=(Rotation.from_rotvec(vec).as_matrix()@matrices(changed[3:9]))[:2].reshape(6)
                    # Held absolute future isolates anchor sensitivity; not a newly valid training example.
                    ss,aa,_=codec.encode(proc,changed,a[i]);base_s,base_a,_=codec.encode(proc,s[i].astype(float),a[i])
                    sensitivity.append(dict(family=meta[i]['family'],index=meta[i]['index'],task=meta[i]['task'],command=meta[i]['command'],variant=meta[i]['variant'],
                        kind=kind,axis=axis,magnitude=magnitude,state_normalized_change_max=float(abs(ss-base_s).max()),relative_target_change_max=float(abs(aa-base_a).max()),
                        input_floating_point='FLOAT64_PHYSICALLY_VALID_NUMERIC_PROBE',new_training_admission=False))
    csv_write('FIXED_CODEC_SENSITIVITY.csv',sensitivity)
    absolute=[]
    for i in range(10):
        params=stat['new_embodiment']['action']['eef_9d' if i<9 else 'gripper_position'];x=a[:,:,i]
        y=codec.normalize(a[:,:,:9] if i<9 else a[:,:,9:],params)[...,i if i<9 else 0]
        absolute.append(dict(feature=i,production_used=i==9,raw_min=float(x.min()),raw_max=float(x.max()),normalized_min=float(y.min()),normalized_max=float(y.max()),
            normalized_abs_p50=float(np.quantile(abs(y),.5)),normalized_abs_p95=float(np.quantile(abs(y),.95)),normalized_abs_p99=float(np.quantile(abs(y),.99)),outside1=float(np.mean(abs(y)>1)),outside10=float(np.mean(abs(y)>10)),outside1000=float(np.mean(abs(y)>1000))))
    csv_write('ABSOLUTE_ACTION_UNUSED_EEF.csv',absolute)
    ortho={name:dict(first_row_norm_max_error=float(abs(np.linalg.norm(x[...,:3],axis=-1)-1).max()),second_row_norm_max_error=float(abs(np.linalg.norm(x[...,3:],axis=-1)-1).max()),
        row_dot_max_abs=float(abs(np.sum(x[...,:3]*x[...,3:],axis=-1)).max()),component_max_abs=float(abs(x).max())) for name,x in [('state',s[:,3:9]),('absolute',a[:,:,3:9]),('relative',rel[:,:,3:9])]}
    write('NUMERIC_SUPPLEMENT.json',dict(state_BF16_cast_inverse=state_cast,rotation6D_orthonormal_checks=ortho,sensitivity_rows=len(sensitivity),
        native_loss_source_sha256=sha(BASE.parent/'isaac-gr00t/gr00t/model/gr00t_n1d7/gr00t_n1d7.py'),native_target='normalized_actions minus same-dtype Gaussian epsilon; actual epsilon values UNOBSERVED (hashes only)',**ZERO))
    exp=[]
    for arm in ('N','R'):
        own=[r for r in consumed if r['arm']==arm];counts=Counter(r['condition_sha256'] for r in own);seen={(r['family'],r['index']) for r in own}
        exp.append(dict(arm=arm,distinct_images=len({(r['body_rgb'],r['wrist_rgb']) for r in own}),distinct_states=len({r['raw_state_sha256'] for r in own}),
            distinct_labels=len({r['raw_label_sha256'] for r in own}),distinct_language=len({r['language'] for r in own}),repeat_histogram=dict(Counter(counts.values())),
            eligible_available=sum(m['eligible'] for m in meta),eligible_unconsumed=sum(m['eligible'] and (m['family'],m['index']) not in seen for m in meta),
            coverage=[dict(task=t,variant=v,available=sum(m['eligible'] and m['task']==t and m['variant']==v for m in meta),seen_sources=len({(r['family'],r['index']) for r in own if r['task']==t and r['variant']==v}),
                terminal_micro=sum(r['task']==t and r['variant']==v and r['anchor']==15 for r in own),near_terminal_micro=sum(r['task']==t and r['variant']==v and r['anchor']>=12 for r in own)) for t in ('M1','M3') for v in range(7)]))
    write('EXPOSURE_SUPPLEMENT.json',exp)
    opt=list(csv.DictReader((OUT/'OPTIMIZATION_JOIN.csv').open()));summaries=[]
    for arm in ('N','R'):
        for phase,lo,hi in [('early',1,1365),('middle',1366,2730),('late',2731,4096)]:
            for variant in range(7):
                own=[r for r in opt if r['arm']==arm and lo<=int(r['unit'])<=hi and int(r['variant'])==variant]
                if not own:continue
                row=dict(arm=arm,phase=phase,variant=variant,microbatches=len(own),clip_fraction=float(np.mean([r['gradient_clipped']=='True' for r in own])))
                for k in ('native_loss','position_loss','delta_loss','rotation_loss','speed_margin_loss','gradient_accumulated_norm','normalized_target_mse','normalized_state_max_abs'):
                    x=np.array([float(r[k]) for r in own]);row[k+'_median']=float(np.median(x));row[k+'_p95']=float(np.quantile(x,.95))
                summaries.append(row)
    csv_write('OPTIMIZATION_PHASE_VARIANT.csv',summaries)
    steps=[]
    for arm in ('N','R'):
        x=np.array([r['accumulated_norm'] for r in lines(SOURCE/arm/'update_gradient_norms.jsonl')]);steps.append(dict(arm=arm,updates=len(x),clipped_updates=int(np.sum(x>1)),median=float(np.median(x)),maximum=float(x.max()),median_absolute_step_change=float(np.median(abs(np.diff(x)))),p95_absolute_step_change=float(np.quantile(abs(np.diff(x)),.95))))
    write('OPTIMIZER_STEP_SUMMARY.json',steps)
    # All saved offline outputs, including REJECT and reused donor rows; source hashes remain separate.
    trace_rows=[];sources={};legacy_experts={};params=proc.norm_params['new_embodiment']['action']['eef_9d']
    for point in ('donor','N4096','R4096'):
        for domain,n in [('new',256),('legacy',440)]:
            paths=sorted((RAW/'offline'/point/domain).glob('*/record.json'));assert len(paths)==n
            for p in paths:
                h=sha(p);r=read(p);assert r['status']=='VALID';sources[str(p)]=h;payload=Path(r['tensor']).read_bytes();assert hashlib.sha256(payload).hexdigest()==r['tensor_sha256']
                sources[r['tensor']]=r['tensor_sha256']
                with np.load(io.BytesIO(payload)) as z:
                    raw=z['raw'][0,:40,:10].astype(float);dec=z['decoded'][0].astype(float);state=z['state'];inverse=codec.unnormalize(raw[:,:9],params)
                    row=dict(point=point,domain=domain,uid=r['row']['uid'],task=r['row'].get('task','LEGACY'),command=r['row'].get('command','LEGACY'),variant=r['row'].get('variant','LEGACY'),gate=r['gate']['decision'],
                        first_xyz_m=float(np.linalg.norm(dec[0,:3]-state[:3])),first_so3_rad=float(angle(matrices(dec[0,3:9]),matrices(state[3:9]))),
                        maximum_segment_m=float(np.linalg.norm(np.diff(dec[:,:3],axis=0),axis=1).max()),maximum_segment_so3_rad=float(angle(matrices(dec[1:,3:9]),matrices(dec[:-1,3:9])).max()),
                        normalized_state_max_abs=float(abs(codec.encode(proc,state)).max()),raw_noise_max_abs=float(abs(z['noise'][...,:10]).max()),
                        normalized_output_xyz_max_abs=float(abs(raw[:,:3]).max()),normalized_output_rot6d_max_abs=float(abs(raw[:,3:9]).max()),
                        inverse_relative_xyz_max_abs=float(abs(inverse[:,:3]).max()),inverse_relative_rot6d_max_abs=float(abs(inverse[:,3:9]).max()),
                        composed_xyz_max_abs=float(abs(dec[:,:3]).max()),inverse_relative_xyz_max_norm=float(np.linalg.norm(inverse[:,:3],axis=1).max()),
                        composed_displacement_max_norm=float(np.linalg.norm(dec[:,:3]-state[:3],axis=1).max()))
                    for k in range(4):
                        for name in ('trace_x','trace_v'):
                            row[name+str(k)+'_xyz_absmax']=float(abs(z[name][k,...,:3]).max());row[name+str(k)+'_rot_absmax']=float(abs(z[name][k,...,3:9]).max())
                    if 'expert' in z:
                        expert=z['expert']
                        if domain=='legacy':legacy_experts[r['row']['uid']]=(r['row']['identities']['reference_sha256'],expert.copy())
                    elif domain=='new':
                        ip=RAW/'offline/source'/r['row']['uid']/'input.npz';sources[str(ip)]=sha(ip)
                        with np.load(ip) as iz:expert=iz['expert']
                    else:
                        reference_hash,expert=legacy_experts[r['row']['uid']]
                        assert r['row']['identities']['reference_sha256']==reference_hash
                    row['prefix_xyz_rmse_m']=None if expert is None else float(np.sqrt(np.mean(np.sum((dec[:9,:3]-expert[:9,:3])**2,axis=1))))
                    row['whole_xyz_rmse_m']=r['metrics']['xyz_rmse_m'] if r['metrics'] else None
                    trace_rows.append(row)
                assert sha(p)==h
            print('TRACE_READ',point,domain,len(trace_rows),flush=True)
    assert len(trace_rows)==2088;csv_write('OFFLINE_TRACE_ROWS.csv',trace_rows);write('OFFLINE_TRACE_SOURCE_HASHES.json',sources)
    groups=[]
    for point in ('donor','N4096','R4096'):
        for variant in ['LEGACY']+list(range(7)):
            for task in (['LEGACY'] if variant=='LEGACY' else ['M1','M3']):
                own=[r for r in trace_rows if (r['point'],r['task'],r['variant'])==(point,task,variant)]
                g=dict(point=point,task=task,variant=variant,rows=len(own),REJECT=sum(r['gate']=='REJECT' for r in own))
                for name in trace_rows[0]:
                    if name in ('point','domain','uid','task','command','variant','gate'):continue
                    x=[r[name] for r in own if r[name] is not None]
                    g[name+'_median']=float(np.median(x)) if x else None;g[name+'_p95']=float(np.quantile(x,.95)) if x else None
                groups.append(g)
    write('OFFLINE_TRACE_SUMMARY.json',dict(groups=groups,representatives=[next(r for r in trace_rows if r['point']==p and r['task']==t and r['variant']==v) for p in ('donor','N4096','R4096') for t in ('M1','M3') for v in range(7)],
        note='Retrospective EVAL descriptive only. No bounds/statistics fit. Shared legacy expert is bound by original reference hash; all 2088 S4 traces present.',**ZERO))
    write('NUMERIC_AUDIT_RECEIPT.json',dict(status='COMPLETE',runtime_s=time.monotonic()-started,max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,**ZERO))
    print('COMPLETE_NUMERIC_CPU',flush=True)


def fill_legacy_prefix():
    """Finish initially unobserved legacy prefix columns; do not rerun other analyses."""
    restrict();rows=list(csv.DictReader((OUT/'OFFLINE_TRACE_ROWS.csv').open()));cache={};checks=[]
    for row in rows:
        if row['domain']!='legacy' or row['prefix_xyz_rmse_m'] not in ('','None'):continue
        uid=row['uid'];p=RAW/'offline'/row['point']/'legacy'/uid/'record.json';record=read(p)
        if uid not in cache:
            donor=read(RAW/'offline/donor/legacy'/uid/'record.json');assert sha(Path(donor['tensor']))==donor['tensor_sha256']
            with np.load(donor['tensor']) as z:cache[uid]=(donor['row']['identities']['reference_sha256'],z['expert'].copy())
        h,expert=cache[uid];assert record['row']['identities']['reference_sha256']==h and sha(Path(record['tensor']))==record['tensor_sha256']
        with np.load(record['tensor']) as z:decoded=z['decoded'][0]
        row['prefix_xyz_rmse_m']=float(np.sqrt(np.mean(np.sum((decoded[:9,:3].astype(float)-expert[:9,:3])**2,axis=1))))
        checks.append(dict(point=row['point'],uid=uid,reference_sha256=h,tensor_sha256=record['tensor_sha256']))
    assert len(checks)==880;csv_write('OFFLINE_TRACE_ROWS.csv',rows)
    summary=read(OUT/'OFFLINE_TRACE_SUMMARY.json')
    for group in summary['groups']:
        if group['variant']!='LEGACY':continue
        values=[float(r['prefix_xyz_rmse_m']) for r in rows if r['point']==group['point'] and r['domain']=='legacy']
        group['prefix_xyz_rmse_m_median']=float(np.median(values));group['prefix_xyz_rmse_m_p95']=float(np.quantile(values,.95))
    summary['note']='All 2088 prefix errors available; shared legacy expert verified by original reference hash. Retrospective only, no fitting.'
    write('OFFLINE_TRACE_SUMMARY.json',summary);write('LEGACY_PREFIX_COMPLETION.json',dict(rows=880,checks=checks,**ZERO));print('LEGACY_PREFIX_COMPLETE',flush=True)


if __name__=='__main__':
    fill_legacy_prefix() if '--fill-legacy-prefix' in sys.argv else main()
