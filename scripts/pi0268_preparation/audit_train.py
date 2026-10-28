"""One bounded TRAIN stream: actual codec scales, consumed exposure and unexecuted panels."""
from cpu_common import *
import io
import time
import resource
from collections import Counter,defaultdict
import numpy as np


def summary(x):
    x=np.asarray(x);a=np.abs(x)
    return dict(min=float(x.min()),max=float(x.max()),p50=float(np.quantile(x,.5)),
        abs_p95=float(np.quantile(a,.95)),abs_p99=float(np.quantile(a,.99)),
        outside_1=float(np.mean(a>1)),outside_10=float(np.mean(a>10)),outside_1000=float(np.mean(a>1000)),
        nonfinite=int(np.sum(~np.isfinite(x))))


def rotation_error(a,b):
    from scipy.spatial.transform import Rotation
    def matrices(x):
        u=x[...,:3].astype(float);u=u/np.linalg.norm(u,axis=-1,keepdims=True)
        v=x[...,3:].astype(float);v-=np.sum(u*v,axis=-1,keepdims=True)*u;v/=np.linalg.norm(v,axis=-1,keepdims=True)
        return np.stack((u,v,np.cross(u,v)),axis=-2)
    return Rotation.from_matrix(matrices(a)@matrices(b).swapaxes(-1,-2)).magnitude()


def main():
    restrict();started=time.monotonic()
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    def forbidden(*a,**kw):raise RuntimeError('PI0268_NO_NEURAL_MODEL_OR_OPTIMIZER')
    torch.nn.Module.__call__=forbidden;torch.optim.Optimizer.__init__=forbidden
    import pi0260_r2_codec as codec
    from gr00t.data.types import ActionType,ActionFormat
    statistics=read(SOURCE/'actual_statistics.json');proc=codec.processor(statistics)
    assert sha(SOURCE/'actual_statistics.json')==sha(R1/'design/statistics.json')=='4439e24348bc43f5c93d380c9ae3c25b4057dbd8111e1e9bf9563b868c53bc73'
    scene=read(BASE/'PI0267/design/SCENE_CARD.json')
    families=[f for f in scene['families'] if f['split']=='FEEDBACK_TRAIN'];assert len(families)==24
    sources={};metadata=[];states=[];actions=[];relative=[];encoded_s=[];encoded_a=[];roundtrip=[];gram=[]
    for family in families:
        fid=family['family_id'];folder=RAW/'S2'/fid;complete=read(folder/'COMPLETE.json');arrays={}
        for name in ('proposals.npz','policy_frames.npz','rows.json'):
            p=folder/name;before=p.stat();value=p.read_bytes();h=hashlib.sha256(value).hexdigest()
            assert h==complete['hashes'][name] and (before.st_size,before.st_mtime_ns)==(p.stat().st_size,p.stat().st_mtime_ns)
            sources[str(p)]=dict(sha256=h,bytes=len(value))
            if name.endswith('.npz'):
                with np.load(io.BytesIO(value)) as z:arrays[name]={k:z[k] for k in (('state_v1','raw_action','position_world','rotation_world') if name=='proposals.npz' else ('body_rgb','wrist_rgb'))}
            else:rows=json.loads(value)
        z=arrays['proposals.npz'];frames=arrays['policy_frames.npz'];frame_hash={}
        eligible=set(read(folder/'codec_admission.json')['eligible_indices'])
        for row in rows:
            i=row['index'];s=z['state_v1'][i];a=z['raw_action'][i];j=row['frame_index']
            if j not in frame_hash:frame_hash[j]={k:digest(frames[k][j]) for k in ('body_rgb','wrist_rgb')}
            rel=proc._convert_to_relative_action(a[:,:9],s[:9],ActionType.EEF,ActionFormat.XYZ_ROT6D)
            ns,na,mask=codec.encode(proc,s,a)
            assert ns.shape==(1,132) and na.shape==(40,132) and np.all(mask[:,:10]==1) and not mask[:,10:].any()
            assert not ns[:,10:].any() and not na[:,10:].any()
            assert np.allclose(na[:,:9],codec.normalize(rel,proc.norm_params['new_embodiment']['action']['eef_9d']).astype(np.float32),rtol=0,atol=0)
            out=dict(family=fid,split='TRAIN',index=i,task=row['task'],command=row['command'],variant=row['variant'],anchor=row['anchor_index'],quartile=row['quartile'],
                eligible=i in eligible,terminal=row['terminal'],frame_index=j,frame_key=row['frame_key'],raw_state_sha256=digest(s),raw_label_sha256=digest(a),
                proposal_sha256=complete['hashes']['proposals.npz'],frame_file_sha256=complete['hashes']['policy_frames.npz'],**frame_hash[j])
            metadata.append(out);states.append(s);actions.append(a);relative.append(rel);encoded_s.append(ns[0,:10]);encoded_a.append(na[:,:10])
            def norms(x):
                u=x[...,:3];v=x[...,3:];n=np.linalg.norm(u,axis=-1);u=u/n[...,None]
                return float(n.min()),float(np.linalg.norm(v-np.sum(u*v,axis=-1,keepdims=True)*u,axis=-1).min())
            gram.append(dict(family=fid,index=i,state=norms(s[3:9]),action=norms(a[:,3:9]),relative=norms(rel[:,3:9])))
            rt=dict(family=fid,index=i)
            for label,cast in (('FP32',na),('BF16_CPU_ARITHMETIC',torch.from_numpy(na).to(torch.bfloat16).float().numpy())):
                decoded=codec.decode(proc,cast,s)
                rt[label+'_xyz_m']=float(np.linalg.norm(decoded[:,:3]-a[:,:3],axis=1).max())
                rt[label+'_so3_rad']=float(rotation_error(decoded[:,3:9],a[:,3:9]).max())
            roundtrip.append(rt)
        print('TRAIN_FAMILY_CPU',fid,len(metadata),flush=True)
    assert len(metadata)==10752
    s=np.stack(states);a=np.stack(actions);rel=np.stack(relative);ns=np.stack(encoded_s);na=np.stack(encoded_a)
    OUT.mkdir(parents=True,exist_ok=True);cache=PREP/'cache';cache.mkdir(exist_ok=True)
    np.savez_compressed(cache/'train_numeric.npz',state=s,action=a,relative=rel,normalized_state=ns,normalized_action=na)
    write('TRAIN_NUMERIC_INDEX.json',metadata);write('TRAIN_SOURCE_IDENTITIES.json',sources)
    write('GRAM_AND_CAST.json',dict(gram=gram,roundtrip=roundtrip,arithmetic_only=True,**ZERO))
    bounds=[]
    for typ,features in statistics['new_embodiment'].items():
        for key,params in features.items():
            lo,w=codec.bounds(params);hi=np.asarray(params.get('q99',params['max']))
            for idx in np.ndindex(w.shape):
                get=lambda n:float(np.asarray(params[n])[idx])
                bounds.append(dict(block=typ,feature_group=key,horizon=idx[0] if len(idx)==2 else -1,feature=idx[-1],
                    q01=get('q01'),q99=get('q99'),minimum=get('min'),maximum=get('max'),width=float(w[idx]),
                    floor=bool(hi[idx]-lo[idx]<=1e-6),forward_gain=float(2/w[idx]),inverse_gain=float(w[idx]/2),
                    production_role='UNUSED_ABSOLUTE_EEF_STATS' if typ=='action' and key=='eef_9d' else 'USED'))
    csv_write('CODEC_FEATURE_BOUNDS.csv',bounds)
    data=[]
    groups={'ALL':np.arange(len(metadata))}
    for t in ('M1','M3'):
        for c in (0,1):
            for v in range(7):
                for q in range(4):
                    groups[f'{t}/c{c}/v{v}/quartile{q}']=np.array([i for i,m in enumerate(metadata) if (m['task'],m['command'],m['variant'],m['quartile'])==(t,c,v,q)])
    for group,indices in groups.items():
        for name,raw,norm in (('state',s[:,None,:],ns[:,None,:]),('relative_action',np.concatenate((rel,a[:,:,9:]),axis=2),na)):
            x,y=raw[indices],norm[indices]
            # Vectorized quantiles retain every feature/horizon without materializing per-row reports.
            quantiles=np.quantile(y,[.5,.95,.99],axis=0);absq=np.quantile(abs(y),[.5,.95,.99],axis=0)
            for h in range(y.shape[1]):
                for d in range(10):
                    z=y[:,h,d]
                    data.append(dict(group=group,rows=len(indices),block=name,horizon=h if name=='relative_action' else -1,feature=d,
                        raw_min=float(x[:,h,d].min()),raw_max=float(x[:,h,d].max()),normalized_min=float(z.min()),normalized_max=float(z.max()),
                        normalized_p50=float(quantiles[0,h,d]),normalized_abs_p50=float(absq[0,h,d]),normalized_abs_p95=float(absq[1,h,d]),normalized_abs_p99=float(absq[2,h,d]),
                        fraction_abs_gt1=float(np.mean(abs(z)>1)),fraction_abs_gt10=float(np.mean(abs(z)>10)),fraction_abs_gt1000=float(np.mean(abs(z)>1000)),
                        nonfinite=int(np.sum(~np.isfinite(z))),clipped=0))
    csv_write('CODEC_TRAIN_STRATA.csv',data)
    lookup={(m['family'],m['index']):i for i,m in enumerate(metadata)};consumed=[];optimization=[];actual={};first={}
    for arm in ('N','R'):
        logs=lines(SOURCE/arm/'sample_audit.jsonl');loss=lines(SOURCE/arm/'S4_audit.jsonl');full=lines(SOURCE/arm/'full_loss.jsonl');grad=lines(SOURCE/arm/'update_gradient_norms.jsonl')
        assert len(logs)==len(loss)==len(full)==8192 and len(grad)==4096
        actual[arm]=logs;first[arm]={}
        for k,row in enumerate(logs):
            assert (row['unit'],row['micro'])==(k//2+1,k%2) and loss[k]['forward_index']==full[k]['step']==k+1
            i=lookup[row['family'],row['index']];m=metadata[i];assert m['eligible'] and row['task']==m['task'] and row['command']==m['command'] and row['frame_key']==m['frame_key']
            first[arm].setdefault(row['condition_sha256'],row)
            consumed.append(dict(**m,arm=arm,unit=row['unit'],micro=row['micro'],template=row['template'],language=row['language'],condition_sha256=row['condition_sha256'],labels_masks_sha256=row['labels_masks_sha256'],nonlanguage_sha256=row['nonlanguage_sha256']))
            g=grad[k//2];assert g['unit']==row['unit']
            optimization.append(dict(arm=arm,unit=row['unit'],micro=row['micro'],task=m['task'],command=m['command'],family=m['family'],anchor=m['anchor'],quartile=m['quartile'],variant=m['variant'],
                native_loss=loss[k]['native_loss'],position_loss=loss[k]['physical_position_loss'],delta_loss=loss[k]['physical_delta_loss'],rotation_loss=full[k]['rotation'],
                speed_margin_loss=full[k]['speed_margin'],total_loss=full[k]['total'],gradient_accumulated_norm=g['accumulated_norm'],gradient_clipped=g['accumulated_norm']>g['clip_max_norm'],
                normalized_target_mse=float(np.mean(na[i]**2)),normalized_target_max_abs=float(np.max(abs(na[i]))),normalized_state_max_abs=float(np.max(abs(ns[i])))))
    write('CONSUMED_CONDITION_INDEX.json',consumed);csv_write('OPTIMIZATION_JOIN.csv',optimization)
    exposure=[]
    for arm in ('N','R'):
        own=[r for r in consumed if r['arm']==arm]
        for dims in (('task','command','variant'),('task','anchor','variant'),('family','task','variant'),('task','template','variant')):
            keys=sorted(set(tuple(r[k] for k in dims) for r in own))
            for key in keys:
                group=[r for r in own if tuple(r[k] for k in dims)==key]
                exposure.append(dict(arm=arm,stratum=dict(zip(dims,key)),microbatches=len(group),distinct_conditions=len({r['condition_sha256'] for r in group}),distinct_sources=len({(r['family'],r['index']) for r in group})))
    write('EXPOSURE.json',dict(groups=exposure,summary={arm:dict(consumed=8192,distinct_conditions=len(first[arm]),distinct_raw_sources=len({(r['family'],r['index']) for r in actual[arm]}),
        terminal_microbatches=sum(r['anchor']==15 for r in consumed if r['arm']==arm),near_terminal_microbatches=sum(r['anchor']>=12 for r in consumed if r['arm']==arm)) for arm in ('N','R')},
        unobserved=['per-module gradient attribution','activation values','training Euler/velocity tensor values (only hashes retained)','fresh seen-TRAIN model fit'],**ZERO))
    fit=[];missing=[]
    for task in ('M1','M3'):
        for command in (0,1):
            for variant in range(7):
                candidates=[r for r in consumed if r['arm']=='R' and (r['task'],r['command'],r['variant'])==(task,command,variant)]
                candidates=sorted(candidates,key=lambda r:(r['family'],r['anchor'],r['index'],r['language'],r['unit'],r['micro']))
                chosen=[];seen=set()
                for q in range(4):
                    rows=[r for r in candidates if r['quartile']==q and r['condition_sha256'] not in seen]
                    if rows:chosen.append(rows[0]);seen.add(rows[0]['condition_sha256'])
                for r in candidates:
                    if len(chosen)==4:break
                    if r['condition_sha256'] not in seen:chosen.append(r);seen.add(r['condition_sha256'])
                if len(chosen)!=4:missing.append(dict(task=task,command=command,variant=variant,available=len(chosen)))
                for r in chosen:
                    fit.append(dict(r,seen_by_N=r['condition_sha256'] in first['N'],seen_by_R=True,input_id=f'train-fit-{len(fit):03d}',
                        frame_source=str(RAW/'S2'/r['family']/'policy_frames.npz'),proposal_source=str(RAW/'S2'/r['family']/'proposals.npz')))
    assert not missing and len(fit)==112 and len({r['condition_sha256'] for r in fit})==112
    noises=[dict(path=str(RAW/'noise'/('legacy-'+b+'.npz')),key='float32',file_sha256=sha(RAW/'noise'/('legacy-'+b+'.npz'))) for b in ('B0','B1')]
    write('SEEN_TRAIN_112_MANIFEST.json',dict(status='DRY_RUN_ONLY',rows=fit,missing_strata=missing,models=['donor','N4096','R4096'],existing_noise=noises,proposed_queries=672,launch_allowed=False,approval_status='NOT_AUTHORIZED',**ZERO))
    factors=[]
    variants=[dict(kind='nominal',axis=None,sign=0,translation_m=0.,world_left_rotation_rad=0.)]
    variants += [dict(kind=kind,axis=axis,sign=sign,translation_m=.01*sign if kind!='rotation_only' else 0.,world_left_rotation_rad=.06*sign if kind!='position_only' else 0.)
                 for kind in ('position_only','rotation_only','combined') for axis in range(3) for sign in (1,-1)]
    for family in families[:2]:
        for task in ('M1','M3'):
            for command in (0,1):
                for anchor in (4,11):
                    m=next(r for r in metadata if (r['family'],r['task'],r['command'],r['anchor'],r['variant'])==(family['family_id'],task,command,anchor,0))
                    noise=RAW/'noise'/('WBC-'+family['family_id']+'.npz')
                    for variant in variants:
                        factors.append(dict(condition_id=f'factor-{len(factors):03d}',base=m,transform_spec_only=variant,
                            noise_file=str(noise),noise_file_sha256=sha(noise),noise_key=f'{task}_E0_{anchor:03d}',
                            matching_rgb='REQUIRES_FUTURE_AUTHORIZED_RENDER',state_correct_label='REQUIRES_FUTURE_AUTHORIZED_LABEL',invalid_conditions='KEEP_WITH_REASON_NO_REPLACEMENT'))
    assert len(factors)==304
    write('FACTOR_304_MANIFEST.json',dict(status='TRANSFORM_SPECS_ONLY_NO_CHANGED_POSES_OR_IMAGES',conditions=factors,proposed_queries=912,proposed_fresh_frame_pairs=304,
        models=['donor','N4096','R4096'],launch_allowed=False,approval_status='NOT_AUTHORIZED',**ZERO))
    write('TRAIN_AUDIT_RECEIPT.json',dict(status='COMPLETE_CPU_TRAIN_SCALE_EXPOSURE_AND_DRY_PANELS',source_statistics_sha256=sha(SOURCE/'actual_statistics.json'),
        archived_statistics_parsed_equal={arm:statistics==read(SOURCE/arm/'checkpoint-config/statistics.json') for arm in ('N','R')},
        numerical_rows=10752,source_families=24,consumed_rows=16384,state_normalized=summary(ns),target_normalized=summary(na),
        raw_dtype=str(s.dtype),relative_dtype=str(rel.dtype),packed_dtype=str(na.dtype),padding_dimensions=122,active_dimensions=10,
        max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,elapsed_s=time.monotonic()-started,
        clipping=False,processor_source_sha256=sha(Path(__import__('inspect').getfile(type(proc)))),codec_source_sha256=sha(Path(codec.__file__)),**ZERO))
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<16*2**30
    print('COMPLETE_CPU_TRAIN_AUDIT',flush=True)


if __name__=='__main__':main()
