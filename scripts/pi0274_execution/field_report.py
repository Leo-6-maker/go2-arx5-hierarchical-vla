"""CPU-only saved local fields: exact denominators, bias, restoring direction and coupling."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
import argparse,csv,json,resource,sys
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from quality_gate import BASE,ROOT,RAW,read,sha
from local_map import scaled,matrix
sys.path[:0]=[str(BASE/'PI0266/code'),str(BASE/'PI0250/runtime/reference-317a/src')]
from pi0255_motion3 import write
from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix
def main(point,partial=False):
    resource.setrlimit(resource.RLIMIT_AS,(16*2**30,16*2**30));os.nice(19)
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-4:])
    if not partial:assert read(ROOT/'evidence'/('FIELD_COMPLETE_'+point+'.json'))['status']=='PASS_ALL2496_REGISTERED_FIELD_OUTPUTS'
    card=read(ROOT/'design/POSE_CARD.json');seen={r['pose_id']:r for r in read(ROOT/'evidence/EXPOSURE_AND_ACTUAL_SEEN.json')['field_rows']}
    old_card=BASE/'PI0273/design/FITTING_CARD.json';assert sha(old_card)=='0f455c3f0ebf7528b95cbb4f4a4d53fa671f27e987697176cc65ce1f610344ac'
    control=read(old_card);assert read(BASE/'PI0273/evidence/MAIN_TRAINING_AUDIT.json')['actual_inputs_match_CPU_plan']
    broad_supplement=Counter((control['base'][v['source_index']]['condition_sha256'],control['base'][v['source_index']]['labels_masks_sha256']) for v in control['streams']['BROAD'] if v['stream']=='SUPPLEMENT')
    anchor=np.asarray(read(BASE/'PI0267/design/SCENE_CARD.json')['policy_anchor_world_m']);rows=[];maps={};teachers={};checks=[]
    for proposal in card['rows']:
        pid=proposal['id'];gp,gr=np.asarray(proposal['goal_position_world']),np.asarray(proposal['goal_rotation_world'])
        folder=RAW/'data'/pid;data=read(folder/'receipt.json');assert data['proposal']==proposal
        with np.load(folder/'sample.npz',allow_pickle=False) as z:state=z['state_v1'].copy();expert=z['raw_action'].copy();p=z['position_world'].copy();r=z['rotation_world'].copy();tp=z['label_position_world'].copy();tr=z['label_rotation_world'].copy()
        assert sha(folder/'sample.npz')==data['sample_sha256'];input_z=scaled(p,r,gp,gr);teacher_z=scaled(tp[8],tr[8],gp,gr)
        teachers.setdefault(proposal['cell'],{})[proposal['variant']]=teacher_z
        for j in (0,1):
            result=RAW/'field'/point/pid/f'j{j}'/'record.json'
            if not result.exists():assert partial;continue
            rec=read(result);assert rec['row']==proposal and rec['point']==point and rec['j']==j and rec['pose_card_sha256']==sha(ROOT/'design/POSE_CARD.json')
            tensor=Path(rec['tensor']);assert sha(tensor)==rec['tensor_sha256'];receipt=read(tensor.with_suffix('.receipt.json'))
            assert receipt['input_sha256']==rec['condition_sha256']==seen[pid]['condition_sha256'];assert receipt['binding']==read(ROOT/'model_bindings'/f'{point}.json')
            with np.load(tensor,allow_pickle=False) as z:decoded=z['decoded'][0].copy()
            value=dict(point=point,cell=proposal['cell'],pose_id=pid,j=j,family=proposal['family'],task=proposal['task'],command=proposal['command'],variant=proposal['variant'],
                training_permitted=proposal['training_permitted'],actual_supplement_occurrences=seen[pid]['exact_supplement_occurrences'] if point=='EQUIL5120' else broad_supplement[seen[pid]['condition_sha256'],seen[pid]['labels_masks_sha256']],
                exact_base_occurrences=seen[pid]['exact_base_occurrences'],gate=rec['gate']['decision'],finite=bool(receipt['finite_output']),
                input_XYZ_m=float(np.linalg.norm(p-gp)),input_SO3_rad=float(Rotation.from_matrix(r@gr.T).magnitude()),
                teacher_projection= json.dumps(data['teacher'],sort_keys=True),teacher_z_0p4=teacher_z.tolist(),input_z=input_z.tolist(),tensor_sha256=rec['tensor_sha256'],
                model_reference_not_executed=True,blind_TEST=False)
            value['actual_full_training_occurrences']=value['actual_supplement_occurrences']+value['exact_base_occurrences']
            rotations=None
            if receipt['finite_output']:
                try:rotations=rot6d_rows_to_matrix(decoded[:,3:9].astype(float))
                except ValueError:pass
            value['finite_valid_rotation']=rotations is not None
            value['numeric_status']='FINITE_VALID_ROTATION' if rotations is not None else ('INVALID_ROTATION' if receipt['finite_output'] else 'NONFINITE')
            if rotations is not None:
                positions=decoded[:,:3].astype(float)+anchor;f=scaled(positions[8],rotations[8],gp,gr)
                xyz=np.linalg.norm(positions-tp,axis=1);angles=Rotation.from_matrix(rotations@tr.transpose(0,2,1)).magnitude()
                value.update(output_z_0p4=f.tolist(),output_XYZ_m=float(np.linalg.norm(positions[8]-gp)),output_SO3_rad=float(Rotation.from_matrix(rotations[8]@gr.T).magnitude()),
                    first_XYZ_fit_m=float(xyz[0]),first_SO3_fit_rad=float(angles[0]),prefix_XYZ_RMSE_m=float(np.sqrt(np.mean(xyz[:9]**2))),prefix_SO3_RMSE_rad=float(np.sqrt(np.mean(angles[:9]**2))),
                    full_XYZ_RMSE_m=float(np.sqrt(np.mean(xyz**2))),full_SO3_RMSE_rad=float(np.sqrt(np.mean(angles**2))),
                    input_to_next_scaled_error_ratio=float(np.linalg.norm(f)/np.linalg.norm(input_z)) if proposal['variant']!='center' and np.linalg.norm(input_z)>0 else None,
                    input_to_next_SO3_ratio=float(Rotation.from_matrix(rotations[8]@gr.T).magnitude()/value['input_SO3_rad']) if value['input_SO3_rad']>1e-12 else None)
                variant=proposal['variant']
                if variant.startswith('r'):
                    axis=np.eye(3)['xyz'.index(variant[1])]*(1 if variant[-1]=='+' else -1);delta=Rotation.from_matrix(rotations[8]@r.T).as_rotvec();parallel=float(delta@axis)
                    value.update(signed_rotation_increment_rad=parallel,negative_increment_is_toward_center=True,rotation_perpendicular_increment_rad=float(np.linalg.norm(delta-parallel*axis)),coupled_position_increment_m=float(np.linalg.norm(positions[8]-p)))
                if rec['gate']['decision']!='REJECT':maps.setdefault((proposal['cell'],j),{})[proposal['variant']]=f
            rows.append(value);checks.append(dict(record=str(result),sha256=sha(result),tensor=str(tensor),sha256_tensor=rec['tensor_sha256']))
    if not partial:assert len(rows)==2496
    matrices=[]
    for cell in dict.fromkeys(r['cell'] for r in card['rows']):
        teacher=matrix(teachers[cell]);own=next(r for r in card['rows'] if r['cell']==cell)
        for j in (0,1):
            values=maps.get((cell,j),{});complete=len(values)==13;result=dict(point=point,cell=cell,task=own['task'],command=own['command'],j=j,complete_accepted_finite_neighborhood=complete,accepted_variants=sorted(values),teacher_map=teacher,
                missing_or_REJECT_variants=sorted(set(teachers[cell])-set(values)),finite_scale_not_stability_certificate=True)
            if complete:result['model_map']=matrix(values)
            matrices.append(result)
    out=RAW/'delivery';out.mkdir(exist_ok=True)
    stem=('FIELD_PARTIAL_' if partial else 'FIELD_')+point;table=out/(stem+'.csv')
    with table.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)
    summary=[]
    for task in ('M1','M3'):
        own=[r for r in rows if r['task']==task];center=[r for r in own if r['variant']=='center'];rot=[r for r in own if r['variant'].startswith('r')]
        summary.append(dict(task=task,observed=len(own),registered=1248,gate_counts=dict(Counter(r['gate'] for r in own)),center_draws=len(center),
            center_XYZ_median_m=float(np.median([r['output_XYZ_m'] for r in center if r['finite_valid_rotation']])) if any(r['finite_valid_rotation'] for r in center) else None,
            center_SO3_median_rad=float(np.median([r['output_SO3_rad'] for r in center if r['finite_valid_rotation']])) if any(r['finite_valid_rotation'] for r in center) else None,
            pure_rotation_observed=len(rot),toward_center_increment=sum(r.get('signed_rotation_increment_rad',0)<0 for r in rot),
            full_accepted_matrices=sum(r['complete_accepted_finite_neighborhood'] for r in matrices if r['task']==task),
            sampled_matrices_spectral_radius_above_one=sum(r.get('model_map',{}).get('spectral_radius',0)>1 for r in matrices if r['task']==task),
            sampled_matrices_largest_singular_value_above_one=sum(max(r.get('model_map',{}).get('singular_values',[0]))>1 for r in matrices if r['task']==task)))
    write(out/(stem+'.json'),dict(status='PARTIAL_FIXED_FIELD_OBSERVATION' if partial else 'ALL2496_FIXED_FIELD_CPU_ANALYSIS',point=point,registered_rows=2496,observed_rows=len(rows),
        summary=summary,matrices=matrices,source_checks=checks,table_sha256=sha(table),draws_not_independent_scenes=True,two_draws_not_output_averaging_policy=True,
        REJECT_references_never_executed=True,finite_REJECT_metrics_retained=True,new_model_queries=0,new_teacher_calls=0,new_renders=0,new_physics=0))
    print('SAVED_FIELD_CPU_ANALYSIS',point,len(rows),json.dumps(summary),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--point',choices=('BROAD5120','EQUIL5120'),required=True);p.add_argument('--partial',action='store_true');a=p.parse_args();main(a.point,a.partial)
