"""Saved task arrays only: requested future motion versus installed and achieved prefix."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
import argparse,json,sys,xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0,'/home/liuyu-nfs/Projects/go2-arx5/PI0277/R2/code')
import numpy as np
from scipy.spatial.transform import Rotation
from evaluation_card import ROOT,RAW,OLD,read,sha,freeze
from panel import function
sys.path.insert(0,str(OLD/'project-src'))
from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix

def bounds(query,controls):
    start=8*query;return start,min(start+8,controls)

def first(value):return float(np.flatnonzero(value)[0]*.01) if np.any(value) else None

def error(raw,installed,actual):
    rms=lambda a,b:float(np.sqrt(np.mean(np.sum((a-b)**2,axis=-1))))
    return dict(raw_to_installed_XYZ_RMS_m=rms(raw[:,:3],installed[:,:3,3]),installed_to_achieved_XYZ_RMS_m=rms(installed[:,:3,3],actual[:,:3,3]),
        raw_to_installed_SO3_RMS_rad=float(np.sqrt(np.mean(Rotation.from_matrix(rot6d_rows_to_matrix(raw[:,3:9])@installed[:,:3,:3].transpose(0,2,1)).magnitude()**2))),
        installed_to_achieved_SO3_RMS_rad=float(np.sqrt(np.mean(Rotation.from_matrix(installed[:,:3,:3]@actual[:,:3,:3].transpose(0,2,1)).magnitude()**2))))

def audit(points):
    reconstruct=function(OLD/'code/replay.py','reconstruct',dict(np=np));rows=[];fit_rows=[];files=[];summaries=[]
    native_facts=function(OLD/'code/report.py','episode',dict(np=np,json=json,ET=ET,ROOT=OLD,sha=sha,reconstruct=reconstruct,Rotation=Rotation))
    for point in points:
        for seed in range(10000,10004):
            paths=[p for p in (RAW/'episodes'/f'TRAIN_PHYSICS-{seed}-{point}').glob('attempt-*/RESULT.json') if read(p)['status'].startswith('VALID')];assert len(paths)==1
            folder=paths[0].parent;result=read(paths[0]);controls=result['control_calls']
            for name,h in result['files'].items():assert sha(folder/name)==h
            files.append(dict(path=str(paths[0]),sha256=sha(paths[0])))
            with np.load(folder/'physics.npz',allow_pickle=False) as z:physics={k:z[k] for k in z.files}
            native=reconstruct(physics)
            for k,v in native.items():assert np.array_equal(v,physics[k])
            grasp=native['is_cubeA_grasped'];lift=physics['cubeA_pose'][:,2]-physics['cubeA_pose'][0,2]
            drops=np.r_[False,grasp[:-1]&~grasp[1:]]
            command=installed=holds=None
            if controls:
                with np.load(folder/'commands.npz') as z:command=z['action'].copy();holds=z['ik_hold'].copy()
                with np.load(folder/'installed.npz') as z:installed=z['installed_target_fk'].copy()
            contacts=[json.loads(line) for line in (folder/'contacts.jsonl').open()]
            transport=read(folder/'GRIPPER_TRANSPORT.json') if (folder/'GRIPPER_TRANSPORT.json').exists() else []
            summaries.append(dict(point=point,seed=seed,status=result['status'],control_calls=controls,first_grasp_s=first(grasp),first_lift2cm_s=first(lift>=.02),
                first_drop_s=first(drops),first_on_top_s=first(native['is_cubeA_on_cubeB']),first_IK_hold_s=float(np.flatnonzero(holds)[0]*.05) if controls and holds.any() else None,
                max_lift_m=float(lift.max()),object_truth='OFFLINE_ONLY',teacher_phase='NOT_USED',native_contact_joint_tracking_facts=native_facts(folder)))
            with np.load(folder/'observations.npz') as obs:
                for q in read(folder/'QUERIES.json'):
                    output=Path(q['output']);side=read(output.with_suffix('.receipt.json'));assert sha(output)==side['output_sha256']
                    with np.load(output) as z:decoded=z['decoded'][0].copy();state=z['state'].copy()
                    start,stop=bounds(q['index'],controls);assert q['time_s']==start*.05 and np.array_equal(state,obs['state_v1'][start])
                    before=5*start;after=5*stop;assert after<len(physics['tcp'])
                    opening=lambda index:float(physics['qpos'][index,-2:].sum()-.00004)
                    row=dict(point=point,seed=seed,query=q['index'],time_s=q['time_s'],output_sha256=side['output_sha256'],noise_sha256=side['noise_sha256'],
                        executed_count=stop-start,planned_targets={str(i):decoded[i].tolist() for i in (0,7,8,15,39)},planned_prefix_gripper=decoded[:8,9].tolist(),
                        TCP_before=physics['tcp'][before].tolist(),TCP_after=physics['tcp'][after].tolist(),actual_opening_before_m=opening(before),actual_opening_after_m=opening(after),
                        requested_delta_from_actual_m={str(i):(decoded[i,:3]-physics['tcp'][before,:3,3]).tolist() for i in (0,7,8,15,39)},
                        cubeA_lift_before_m=float(lift[before]),cubeA_lift_after_m=float(lift[after]),max_lift_during_prefix_m=float(lift[before:after+1].max()),
                        grasp_samples=int(grasp[before:after+1].sum()),drop_in_prefix=bool(drops[before:after+1].any()),on_top_samples=int(native['is_cubeA_on_cubeB'][before:after+1].sum()),
                        contact_points_during_prefix=sum(len(c['points']) for sample in contacts[before+1:after+1] for c in sample['contacts']),
                        actuator_transport=transport[start:stop],contact_interpretation='Saved contacts only; no stiffness/force inference')
                    if stop>start:
                        assert np.array_equal(command[start:stop],decoded[:stop-start])
                        row.update(error(command[start:stop],installed[start:stop],physics['tcp'][5*(start+1):5*stop+1:5]))
                        row.update(commanded_gripper=command[start:stop,9].tolist(),IK_hold_controls=np.flatnonzero(holds[start:stop]).tolist())
                    else:row.update(commanded_gripper=[],IK_hold_controls=[],installation='REFUSED_NOT_INSTALLED')
                    rows.append(row)
        fit={r['uid']:r for r in read(ROOT/'design/FIT_CARD.json')['rows']}
        for r in read(ROOT/'evidence'/f'FIT16_{point}.json')['results']:
            p=Path(r['output']);assert sha(p)==r['sha256'];item=fit[r['uid']];side=read(p.with_suffix('.receipt.json'))
            with np.load(p) as z:pred=z['decoded'][0]
            with np.load(Path(item['episode'])/'commands.npz') as z:target=z['action'][item['anchor']:item['anchor']+40]
            opening_error=np.abs(pred[:,9]-target[:,9]);entry=dict(point=point,uid=r['uid'],seed=item['seed'],group=item['group'],noise_sha256=side['noise_sha256'],actual_noise_directory=p.parent.name,
                prefix_XYZ_MAE_m=float(np.linalg.norm(pred[:8,:3]-target[:8,:3],axis=1).mean()),prefix_SO3_MAE_rad=float(Rotation.from_matrix(rot6d_rows_to_matrix(pred[:8,3:9])@rot6d_rows_to_matrix(target[:8,3:9]).transpose(0,2,1)).magnitude().mean()))
            for label,mask in [('OPEN',target[:,9]==1),('CLOSE',target[:,9]==0),('TRANSITION',(target[:,9]!=0)&(target[:,9]!=1))]:entry[label]=dict(points=int(mask.sum()),command_MAE=float(opening_error[mask].mean()) if mask.any() else None)
            fit_rows.append(entry)
    for row in fit_rows:
        comparisons=[r for r in fit_rows if r['uid']==row['uid'] and r['point']!=row['point']]
        row['paired_points']=[r['point'] for r in comparisons if r['noise_sha256']==row['noise_sha256']]
        row['UNPAIRED_noise_points']=[r['point'] for r in comparisons if r['noise_sha256']!=row['noise_sha256']]
    freeze(ROOT.with_name('R3')/'evidence'/('PREFIX_V2_'+ '-'.join(points)+'.json'),dict(status='PASS_SAVED_ARRAY_PREFIX_INSTALLATION_AUDIT',code_sha256=sha(__file__),points=points,case_summaries=summaries,queries=rows,FIT16=fit_rows,source_results=files,
        new_model_queries=0,new_physics_steps=0,no_teacher_phase=True,causal_claim='OBSERVATIONAL_ONLY',selected_checkpoint='NONE',deployment='NONE'))
    print(json.dumps(dict(points=points,queries=len(rows),FIT=len(fit_rows),summaries=summaries)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('points',nargs='*');p.add_argument('--check',action='store_true');a=p.parse_args()
    if a.check:
        assert bounds(0,0)==(0,0) and bounds(4,32)==(32,32) and bounds(79,640)==(632,640)
        assert first(np.array([False,True]))==.01 and first(np.zeros(3,dtype=bool)) is None
        print('PASS_REJECT_AND_FINAL_PREFIX_INDEX_BOUNDARIES')
    else:audit(a.points)
