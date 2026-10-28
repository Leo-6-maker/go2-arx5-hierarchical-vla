"""Seal exact fixed input/noise cards before any PI0268 quality query."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
import json
from pathlib import Path
import numpy as np
from freeze_contracts import ROOT,OLD,RAW,read,sha,write
import pi0260_r2_codec as codec
from scipy.spatial.transform import Rotation

def matrices(x):
    u=x[...,:3].astype(float);u/=np.linalg.norm(u,axis=-1,keepdims=True)
    v=x[...,3:].astype(float);v-=np.sum(u*v,axis=-1,keepdims=True)*u;v/=np.linalg.norm(v,axis=-1,keepdims=True)
    return np.stack((u,v,np.cross(u,v)),axis=-2)

def main():
    import torch
    torch.set_num_threads(4)
    def no_model(*a,**kw):raise RuntimeError('NO_MODEL_IN_CARD_FREEZE')
    torch.nn.Module.__call__=no_model;torch.optim.Optimizer.__init__=no_model
    assert not (ROOT/'evidence/model_attempts.jsonl').exists()
    root=RAW.parents[1]/'PI0268/R1';factors=root/'factors'
    final=read(factors/'COMPLETE.json');assert final['proposed']==304
    fit=read(ROOT/'design/FIT112_FIRST2048.json');prior=read(OLD/'R1/design/FIXED_OFFLINE_CARD.json')
    approved=read(ROOT/'design/APPROVED_RUN_CARD.json');procs={a:codec.processor(read(ROOT/'contracts'/a/'source/dataset/statistics.json')) for a in ('OLD','ROT')}
    fitrows=[];factorrows=[];noise_files=set();roundtrips=[]
    for row in fit['rows']:
        for noise in fit['noise']:
            fitrows.append(dict(uid=row['input_id']+'-'+Path(noise['path']).stem,kind='fit',input_id=row['input_id'],
                input=str(ROOT/'design/fit_fixtures'/row['fixture_file']),input_sha256=row['fixture_sha256'],language=row['language'],
                noise_file=noise['path'],noise_name=noise['key'],task=row['task'],command=row['command'],variant=row['variant'],
                family=row['family'],anchor=row['anchor'],quartile=row['quartile'],first_old_consumption=dict(unit=row['unit'],micro=row['micro']),
                processed=row['processed'],actual_new_seen='PENDING_TRAINING_AUDIT'))
            noise_files.add(noise['path'])
    for key,digest in final['records'].items():
        p=factors/key/'record.json';assert sha(p)==digest;record=read(p);ip=p.with_name('input.npz');assert sha(ip)==record['input_sha256']
        condition=record['condition'];base=condition['base'];reasons=list(record['reasons'])
        with np.load(ip) as z:s=z['state_v1'];a=z['raw_action']
        for arm,proc in procs.items():
            ps,pa,mask=codec.encode(proc,s,a);out=codec.decode(proc,pa,s)
            xyz=float(np.linalg.norm(out[:,:3]-a[:,:3],axis=-1).max())
            rot=float(Rotation.from_matrix(matrices(out[:,3:9])@matrices(a[:,3:9]).swapaxes(-1,-2)).magnitude().max())
            assert xyz<1e-5 and rot<1e-4 and np.isfinite(pa).all() and np.isfinite(ps).all()
            if arm=='ROT':assert max(abs(ps[:,3:9]).max(),abs(pa[:,3:9]).max())<=1+2e-6
            roundtrips.append(dict(condition_id=key,arm=arm,xyz_m=xyz,so3_rad=rot))
        factorrows.append(dict(uid=key,kind='factor',input=str(ip),input_sha256=sha(ip),language=record['language'],
            noise_file=condition['noise_file'],noise_name=condition['noise_key'],task=base['task'],command=base['command'],family=base['family'],
            anchor=base['anchor'],transform=condition['transform_spec_only'],admitted=not reasons,reasons=reasons,raw_record_sha256=digest))
        noise_files.add(condition['noise_file'])
    legacy=prior['legacy_rows'];assert len(legacy)==440
    noise_files.update(str(RAW/'noise'/('legacy-'+r['noise']+'.npz')) for r in legacy)
    noise=[]
    for path in sorted(noise_files):
        with np.load(path) as z:keys=[dict(name=k,noise_sha256=__import__('hashlib').sha256(z[k].tobytes()).hexdigest(),shape=list(z[k].shape),dtype=str(z[k].dtype)) for k in z.files]
        noise.append(dict(file=path,sha256=sha(path),keys=keys))
    points=['OLD0','ROT0','OLD2048','ROT2048']
    write(ROOT/'design/FIXED_OFFLINE_CARD.json',dict(schema='PI0268_R1_FIXED_OFFLINE_V1',release=approved['release'],model_points=points,
        fit_rows=fitrows,factor_rows=factorrows,legacy_rows=legacy,noise=noise,contracts=approved['contracts'],old_statistics_sha256=approved['contracts']['OLD'],
        rows_per_point=len(fitrows)+len(factorrows)+440,logical_rows=4*(len(fitrows)+len(factorrows)+440),source_legacy_card_sha256=sha(OLD/'R1/design/FIXED_OFFLINE_CARD.json')))
    virtual=read(OLD/'R1/design/EVALUATION_CARD.json')
    virtual.update(schema='PI0268_R1_FIXED_VIRTUAL_V1',model_points=points,primary_episodes=256,contracts=approved['contracts'],
        statistics_sha256=None,source_card_sha256=sha(OLD/'R1/design/EVALUATION_CARD.json'),OLD0_reuse=64,new_planned_episodes=192,
        reliability=dict(original_completion_by_s=35.,horizon_s=40.,hold_from_s=35.,xyz_m=.035,so3_rad=.10,all_saved_ticks=True,
            task_backend_min=13,command_min=6,zero_gate_refusals=True,zero_dangerous_collisions=True,independent_of_original_V3=True))
    virtual['caps']=dict(inference_attempts=24576,inference_heads=98304,new_virtual_starts=208,new_virtual_queries=19200,total_new_frame_pairs=20000)
    write(ROOT/'design/EVALUATION_CARD.json',virtual)
    write(ROOT/'evidence/FACTOR_CODEC_ADMISSION.json',dict(status='PASS_304_FIXED_FACTOR_INPUTS_AND_TWO_CODECS',proposed=304,admitted=sum(r['admitted'] for r in factorrows),
        roundtrips=roundtrips,frame_pairs_new=final['fresh_pairs_this_attempt'],frame_pairs_reused=final['reused_pairs_this_attempt'],model_queries=0))
    write(ROOT/'design/PAIRING_PLAN.json',read(OLD/'R1/design/PAIRING_PLAN.json'))
    print('SEALED',len(fitrows),len(factorrows),len(legacy),'NO_MODEL_QUERY',flush=True)

if __name__=='__main__':main()
