"""Finite CPU-only readback of one closed native WBC fixture; no actor or physics."""
import argparse,json,os,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from audit_saved import ROOT,OLD,sha,identity,save,stats

def pose(rotation,translation):
    p=np.eye(4);p[:3,:3]=rotation;p[:3,3]=translation;return p

def vec(e,key,default='0 0 0'):
    return np.fromstring(e.attrib.get(key,default) if e is not None else default,sep=' ')

def main(name):
    d=ROOT/'wbc'/name;result=json.loads((d/'RESULT.json').read_text());done=json.loads((d/'DONE.json').read_text())
    assert done.get('returncode')==0 and result['status'] in ('WBC_FIXTURE_EXECUTED','VALID_NATIVE_TERMINATION','VALID_GATE_REJECT')
    rows=json.loads((d/'rows.json').read_text());origin=result['initialization_s']
    binding=json.loads((OLD/'evidence/WBC_BINDING.json').read_text());bound=next(x for x in binding['files'] if x['name']=='native_urdf')
    urdf=Path(bound['path']);assert sha(urdf)==bound['sha256'];tree=ET.parse(urdf).getroot()
    names=list(rows[0]['hard_penetration']);joints={e.attrib['name']:e for e in tree.findall('joint')}
    limits=np.array([[float(joints[n].find('limit').attrib[k]) for k in ('lower','upper')] for n in names])
    bychild={e.find('child').attrib['link']:e for e in tree.findall('joint')};chain=[];link='x5_link6'
    while link in bychild:
        e=bychild[link];chain.append(e);link=e.find('parent').attrib['link']
    chain.reverse();assert [e.attrib['name'] for e in chain if e.attrib.get('type')!='fixed']==[f'x5_joint{i}' for i in range(1,7)]
    def fk(row):
        q=dict(zip(names,row['q']));root=row['root'];p=pose(Rotation.from_quat(root[3:7]).as_matrix(),root[:3])
        for e in chain:
            o=e.find('origin');p=p@pose(Rotation.from_euler('xyz',vec(o,'rpy')).as_matrix(),vec(o,'xyz'))
            if e.attrib.get('type')!='fixed':
                axis=vec(e.find('axis'),'xyz','0 0 1');axis=axis/np.linalg.norm(axis)
                p=p@pose(Rotation.from_rotvec(axis*q[e.attrib['name']]).as_matrix(),np.zeros(3))
        return p@pose(np.eye(3),[.1424,0,.0001057])
    fp=np.array([fk(row) for row in rows]);q=np.array([r['q'] for r in rows]);margin=np.minimum(q-limits[:,0],limits[:,1]-q)
    hard=np.maximum(0,-margin);saved_hard=np.array([[r['hard_penetration'][n] for n in names] for r in rows]);assert np.max(abs(hard-saved_hard))<1e-12
    def orientation(r):return (Rotation.from_quat(r['installed_quaternion'])*Rotation.from_quat(r['native_quaternion']).inv()).magnitude()
    tail=[r for r in rows if 8<=r['time']-origin<=10+1e-5]
    ep=[np.linalg.norm(np.array(r['native_TCP'])-r['installed']) for r in tail];er=[orientation(r) for r in tail]
    def observed(values): return stats(values) if len(values) else None
    soft=[x for r in rows for x in r['telemetry']['joint_limit_margin']]
    with np.load(d/'reference.npz',allow_pickle=False) as v:reference={k:v[k] for k in v.files}
    # Reconstruct the bound canonical curve on CPU; never interpolate a different
    # trajectory or equate actual progress with the intended time-law reference.
    from wbc_cpu import batch
    import torch
    curve=batch(reference['position'],reference['quaternion'],reference['anchor'])
    times=torch.tensor([max(0.,r['time']-origin) for r in rows],dtype=torch.float32)
    s=curve.s_ref(times[None])
    rawpos=curve.p_at(s).numpy()[0]
    rawrot=Rotation.from_quat(curve.quat_at(s).numpy()[0])
    layout=json.loads((d/'observation_layout.json').read_text());parts=layout['parts'];width=sum(parts.values());offset={};i=0
    for k,v in parts.items():offset[k]=i;i+=v
    errors={k:[] for k in ('preview_rot6d','preview_position','body_target_rot6d','body_target_quaternion','body_target_position','body_position_error','body_rotation_proxy','proxy_sin_halfangle','native_link6_orientation','actor_FK_orientation','history_shift')}
    previous=None;ticks=[];historywidth=[];initial_history=None
    with (d/'actor_inputs.jsonl').open() as f:
        for line in f:
            a=json.loads(line);r=a['state'];h=np.asarray(a['obs_history']);current=h[-width:];ticks.append(a['tick']);historywidth.append(len(h))
            if previous is not None:errors['history_shift'].append(np.max(abs(h[:-width]-previous[width:])))
            else:initial_history=h.reshape(-1,width)
            previous=h
            def part(k):return current[offset[k]:offset[k]+parts[k]]
            preview=part('traj_preview').reshape(-1,14);heading=Rotation.from_quat(r['heading_quaternion']);base=Rotation.from_quat(r['base_quaternion'])
            preview_rot=heading.inv()*Rotation.from_quat(a['preview_quaternion']);preview_pos=heading.inv().apply(np.asarray(a['preview_position'])-np.array(r['root'][:3]))
            errors['preview_rot6d'].append(np.max(abs(preview[:,3:9]-preview_rot.as_matrix()[:,:,:2].reshape(-1,6))))
            errors['preview_position'].append(np.max(abs(preview[:,:3]-preview_pos)))
            target=Rotation.from_quat(r['target_body_quaternion']);installed=Rotation.from_quat(r['installed_quaternion']);native=Rotation.from_quat(r['native_quaternion'])
            errors['body_target_rot6d'].append(np.max(abs(part('ee_target_rot6d_body')-target.as_matrix()[:,:2].reshape(6))))
            errors['body_target_quaternion'].append((target.inv()*base.inv()*installed).magnitude())
            errors['body_target_position'].append(np.max(abs(part('ee_target_pos_body')-base.inv().apply(np.array(r['installed'])-np.array(r['root'][:3])))))
            errors['body_position_error'].append(np.max(abs(part('ee_pos_err_body')-base.inv().apply(np.array(r['installed'])-np.array(r['native_TCP'])))))
            errors['body_rotation_proxy'].append(np.max(abs(part('ee_rot_err_body')-base.inv().apply(r['orientation_error_proxy']))))
            errors['proxy_sin_halfangle'].append(abs(np.linalg.norm(r['orientation_error_proxy'])-np.sin(orientation(r)/2)))
            errors['native_link6_orientation'].append((native*Rotation.from_quat(r['actual_link6_quaternion']).inv()).magnitude())
            errors['actor_FK_orientation'].append((native*Rotation.from_matrix(fk(r)[:3,:3]).inv()).magnitude())
    assert ticks==list(range(result['controlled_ticks'])) and set(historywidth)=={layout['history']}
    complete=result['status']=='WBC_FIXTURE_EXECUTED' and abs(result['actual_elapsed_s']-10)<1e-5
    pm=observed(ep);rm=observed(er);gate=bool(complete and len(tail)>=100 and pm['rms']<=.020 and pm['max']<=.040 and rm['rms']<=.15 and rm['max']<=.30 and hard.max()<=1e-6)
    report=dict(status='CPU_READBACK_COMPLETE',fixture=name,result_status=result['status'],quality=gate,full10s=complete,final2s_samples=len(tail),position_final2s_m=pm,rotation_final2s_rad=rm,
        hard_penetration_max_SI=float(hard.max()),hard_penetration_raw_recomputation_max_abs=float(np.max(abs(hard-saved_hard))),soft_margin_min_mixed_SI=float(min(soft)),
        FK_saved_position_max_abs_m=float(np.max(abs(fp[:,:3,3]-np.array([r['TCP'] for r in rows])))),
        FK_native_position_m=stats(np.linalg.norm(fp[:,:3,3]-np.array([r['native_TCP'] for r in rows]),axis=1)),
        FK_native_rotation_rad=stats((Rotation.from_matrix(fp[:,:3,:3])*Rotation.from_quat([r['native_quaternion'] for r in rows]).inv()).magnitude()),
        raw_to_installed_position_m=stats(np.linalg.norm(np.array([r['installed'] for r in rows])-rawpos,axis=1)),
        raw_to_installed_orientation_rad=stats((rawrot*Rotation.from_quat([r['installed_quaternion'] for r in rows]).inv()).magnitude()),
        intended_reference='Exact delivered canonical curve at s_ref(elapsed); not s_progress or closest achieved point',
        progress=dict(max_step_m=float(np.max(np.diff([r['progress'] for r in rows]))),
            last_m=rows[-1]['progress'],last_s_ref_m=rows[-1]['s_ref'],last_timing_error_m=rows[-1]['timing_err']),
        native_termination=result.get('event'),
        native_proxy_definition='Sign-corrected quaternion vector part, norm=sin(theta/2), dimensionless; true SO3 geodesic theta is radians. Original actor input unchanged.',
        proxy_norm_all=stats([np.linalg.norm(r['orientation_error_proxy']) for r in rows]),
        proxy_recovered_theta_all_rad=stats([2*np.arcsin(np.clip(np.linalg.norm(r['orientation_error_proxy']),0,1)) for r in rows]),
        actor_readback={k:observed(v) for k,v in errors.items()},actor_records=len(ticks),history_frame_width=width,history_frames=layout['history']//width,
        initial_history_distinct_frames=int(len(np.unique(initial_history,axis=0))),initial_history_meaning='Initial saved actor history is inherited simulator history, not invented fresh stationary observations',
        sameTCP='Full URDF chain from floating root to x5_link6 plus translation[.1424,0,.0001057] and identity rotation; q order bound by saved hard_penetration insertion order in sealed snapshot',
        gate_semantics='Original native stop wins; complete10s plus final2s position RMS/max<=.020/.040m, SO3<=.15/.30rad, hard limits<=1e-6SI. Fixture is NOT_AN_OBJECT_TASK.',
        inputs=[identity(d/n) for n in ('RESULT.json','DONE.json','rows.json','reference.npz','actor_inputs.jsonl','observation_layout.json','installation.json')]+[identity(urdf)],
        new_model_queries=0,new_actor_forwards=0,new_physics_steps=0,new_training_updates=0)
    wbcroot=Path('/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage13/extracted/WBCTrackingShare')
    report['source_files']=[identity(wbcroot/p) for p in ('go1_gym/utils/math_utils.py','go1_gym/envs/roboduet/wbc_env.py')]+[identity(ROOT/'code/wbc.py')]
    save('WBC-'+name+'.json',report)
    snap=json.loads((ROOT/'audit/NEW_RESULTS_SNAPSHOT.json').read_text());snap['wbc']=[r for r in snap.get('wbc',[]) if r.get('fixture')!=name]+[report];save('NEW_RESULTS_SNAPSHOT.json',snap)
    print(json.dumps({k:report[k] for k in ('fixture','quality','full10s','final2s_samples','position_final2s_m','rotation_final2s_rad','hard_penetration_max_SI','actor_readback')}))

if __name__=='__main__':
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    p=argparse.ArgumentParser();p.add_argument('fixture');args=p.parse_args();main(args.fixture)
