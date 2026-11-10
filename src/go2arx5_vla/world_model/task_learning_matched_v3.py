"""Fixed A/B/C TRAIN40x3 and DEV-reference20x3, all ordinary inference.

Saved group0 TRAIN predictions are reused, not rerun or selected by score.
This development comparison grants no physical execution or TEST admission.
"""
import json
from pathlib import Path

import numpy as np

from .storage import digest
from . import task_learning_features_v1 as features
from .task_learning_local_analysis_v1 import geometry


def summary(rows):
    def portion(group):
        finite=[r for r in group if r['finite']]
        peak=[max(0.,r['envelope']['metrics']['maximum_discrete_translation_acceleration_m_s2']/.25-1.)for r in finite]
        errors=[v for r in finite for v in r['errors']['pointwise']['XYZ_euclidean_m']]
        moving=[r['motion']for r in finite if r['motion']['expert_displacement_m']>=.001]
        return dict(requested=len(group),completed=len(group),finite=len(finite),
            envelope_pass=sum(r['finite']and r['envelope']['original_envelope_pass']for r in group),
            pass_fraction=sum(r['finite']and r['envelope']['original_envelope_pass']for r in group)/len(group),
            p95_window_peak_acceleration_excess_ratio=float(np.quantile(peak,.95))if peak else None,
            mean_XYZ_m=float(np.mean(errors))if errors else None,p95_point_XYZ_m=float(np.quantile(errors,.95))if errors else None,
            moving_expert_windows=len(moving),positive_direction_fraction=(sum(r['direction_cosine']is not None and r['direction_cosine']>0 for r in moving)/len(moving))if moving else None,
            median_displacement_ratio=float(np.median([r['displacement_ratio']for r in moving]))if moving else None,
            gripper_event_missing={k:sum(r['errors'][k+'_events']['missing']for r in finite)for k in ('close','open')},
            gripper_event_spurious={k:sum(r['errors'][k+'_events']['spurious']for r in finite)for k in ('close','open')})
    return {split:dict(overall=portion([r for r in rows if r['split']==split]),
        phases={phase:portion([r for r in rows if r['split']==split and r['phase']==phase])for phase in
            ('PREGRASP','DESCEND','CLOSE','LIFT','HOLD')},
        noises={str(noise):portion([r for r in rows if r['split']==split and r['noise_group']==noise])for noise in range(3)})
        for split in ('TRAIN','DEV')}


def evaluate(spec,model,data,root,guard):
    import torch
    from scripts.pi0292.durable import write
    from scripts.pi0292.eval_contract import error_metrics
    from gr00t.data.embodiment_tags import EmbodimentTag
    from .task_learning_native_dataset_v1 import TaskLearningNativeDataset
    from .task_learning_probes_v1 import selections,contract as probe_contract
    package=Path(spec['last2_package']['path'])
    features.need(digest(package/'PACKAGE_MANIFEST.json')==spec['last2_package']['sha256'],'PINNED_OWN_LAST2_INFERENCE_PACKAGE')
    manifest=json.loads((package/'PACKAGE_MANIFEST.json').read_bytes())
    features.need(manifest['parent_package']==spec['package']and manifest['adaptation_step']==500
        and manifest['dataset_sha256']==spec['dataset_sha256']and manifest['statistics_sha256']==spec['statistics_sha256']
        and all(digest(package/k)==v for k,v in manifest['files'].items()),'EXACT_COMPLETE500_OWN_ORDINARY_PACKAGE')
    values=torch.load(package/'target_parameters.pt',map_location='cpu',weights_only=True)
    features.install_ordinary_diagnostic_delta(model,values,json.loads((package/'PARAMETER_CONTRACT.json').read_bytes()));del values
    dev=TaskLearningNativeDataset(spec['bindings']['dataset'],spec['bindings']['source'],spec['bindings']['cosmos'],split='DEV',for_optimization=False)
    dev.processor=data.processor
    features.need(dev.manifest_sha==data.manifest_sha and dev.statistics_sha==data.statistics_sha,'SAME_TRAIN_ONLY_STATS_FOR_DEV_REFERENCE')
    saved=Path(spec['saved_TRAIN40']['path']);features.need(digest(saved)==spec['saved_TRAIN40']['sha256'],'SEALED_REUSED_GROUP0_TRAIN40')
    originals=json.loads(saved.read_bytes())['records'];features.need(len(originals)==40,'REUSE_ALL_ORIGINAL_TRAIN40')
    output=root/'matched';output.mkdir();rows=[];actual=0;reused=0;model.to('cuda').eval()
    for split,dataset,count in [('TRAIN',data,40),('DEV',dev,20)]:
        probes=selections(dataset);features.need(len(probes)==count,'FIXED_COMPLETE_PHASE_PROBES')
        for number,(index,phase)in enumerate(probes):
            episode,anchor=dataset.rows[index]
            if split=='TRAIN':features.need(all(originals[number][k]==v for k,v in
                [('episode',episode['id']),('anchor',anchor),('phase_audit_only',phase),('seed',2026110900+number)]),'UNCHANGED_TRAIN40_IDENTITIES_AND_GROUP0_NOISE')
            for noise in range(3):
                guard();seed=2026110900+number+1000000*noise
                if split=='TRAIN'and noise==0:
                    old=originals[number];file=Path(spec['saved_TRAIN40_samples'])/old['sample_file']
                    features.need(digest(file)==old['sha256'],'UNCHANGED_REUSED_TRAIN_PREDICTION')
                    with np.load(file,allow_pickle=False)as z:normalized,physical,expert,state=[z[k].copy()for k in
                        ('normalized','physical_H40','expert_policy_H40','actual_state10')]
                    reused+=1;origin='SEALED_GROUP0_REUSED'
                else:
                    state,expert,_,_=dataset.raw(index)
                    inputs=dataset.processor.collator([dataset[index]['processed']])['inputs']
                    with torch.random.fork_rng(devices=[0]),torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                        torch.manual_seed(seed);normalized=model.get_action(features.inference_inputs(inputs))['action_pred'].detach().float().cpu().numpy()
                    if np.isfinite(normalized).all():
                        decoded=dataset.processor.decode_action(normalized,EmbodimentTag.NEW_EMBODIMENT,
                            dict(eef_9d=state[None,None,:9],gripper_position=state[None,None,9:]))
                        physical=np.concatenate([decoded[k]for k in ('eef_9d','gripper_position')],axis=-1)[0]
                    else:physical=np.full((40,10),np.nan)
                    file=output/f'{split}-{number:02d}-noise{noise}.npz'
                    np.savez(file,normalized=normalized,physical_H40=physical,expert_policy_H40=expert,actual_state10=state,noise_seed=np.asarray(seed,np.int64))
                    actual+=1;origin='ACTUAL_ORIGINAL_FULL_INFERENCE'
                finite=bool(np.isfinite(normalized).all()and np.isfinite(physical).all())
                displacement=physical[-1,:3]-physical[0,:3];reference=expert[-1,:3]-expert[0,:3]
                d,e=(float(np.linalg.norm(displacement))if finite else None),float(np.linalg.norm(reference))
                row=dict(split=split,episode=episode['id'],anchor=anchor,phase=phase,noise_group=noise,seed=seed,
                    origin=origin,sample=dict(path=str(file),sha256=digest(file)),finite=finite,
                    envelope=geometry(physical,state)if finite else None,errors=error_metrics(physical,expert,state)if finite else None,
                    motion=dict(expert_displacement_m=e,predicted_displacement_m=d,
                        displacement_ratio=d/e if finite and e>1e-14 else None,direction_cosine=float(displacement@reference/(d*e))if finite and d*e>1e-14 else None))
                rows.append(row);write(root/'PROGRESS.private.json',dict(completed=len(rows),requested=180,actual_new_queries=actual,reused_queries=reused,method=spec['method']))
    features.need(len(rows)==180 and actual==140 and reused==40,'ALL_MATCHED180_WITH140_NEW_AND40_REUSED')
    result=dict(schema='TASK_LEARNING_MATCHED_A_B_C_ORDINARY_PROBES_V3',method=spec['method'],source_sha=spec['source_sha'],
        last2_package=spec['last2_package'],dataset_sha256=spec['dataset_sha256'],statistics_sha256=spec['statistics_sha256'],
        selection_contract=probe_contract(),rows=rows,summary=summary(rows),requested_predictions=180,
        actual_original_model_queries=actual,reused_sealed_predictions=reused,unique_TRAIN_layouts=8,unique_DEV_reference_layouts=4,
        DEV_reference_predictions_are_not_independent_scenes=True,optimizer_updates=0,physics_integrations=0,
        original_DEV80_TEST_conditions=0,task_success='NOT_MEASURED_BY_OPEN_LOOP',action_clipping_added=False,Teacher_fallback=False)
    write(root/'RESULT_MATCHED.private.json',result);return result


def comparison(results):
    """Both split signals and fixed motion/error safeguards precede D admission."""
    a,c=results['A']['summary'],results['C']['summary'];splits={}
    for split in ('TRAIN','DEV'):
        old,new=a[split]['overall'],c[split]['overall']
        excess=old['p95_window_peak_acceleration_excess_ratio'];after=new['p95_window_peak_acceleration_excess_ratio']
        gain=new['pass_fraction']-old['pass_fraction']
        reduction=(1-after/excess)if excess is not None and excess>0 and after is not None else None
        errors=all(c[split]['phases'][p]['finite']==c[split]['phases'][p]['requested']
            and a[split]['phases'][p]['finite']==a[split]['phases'][p]['requested']
            and c[split]['phases'][p]['mean_XYZ_m']<=a[split]['phases'][p]['mean_XYZ_m']+.0005 and
            c[split]['phases'][p]['p95_point_XYZ_m']<=a[split]['phases'][p]['p95_point_XYZ_m']+.001
            for p in a[split]['phases'])
        motion=(new['moving_expert_windows']>0 and new['positive_direction_fraction']>=.9 and new['median_displacement_ratio']>=.8)
        signal=gain>=.2-1e-12 or(reduction is not None and reduction>=.5-1e-12)
        splits[split]=dict(pass_fraction_gain=gain,p95_excess_reduction_fraction=reduction,
            preregistered_mechanism_signal=signal,phase_error_safeguards_pass=errors,motion_safeguards_pass=motion,
            all_finite=new['finite']==new['requested'],qualified_mechanism=signal and errors and motion and new['finite']==new['requested'])
    return dict(schema='TASK_LEARNING_MATCHED_A_B_C_DECISION_V3',splits=splits,
        decision='C_MECHANISM_BENEFIT_PREPARE_INDEPENDENT_LOCAL_ADMISSION_AND_D'if all(r['qualified_mechanism']for r in splits.values())else
            'C_V1_NO_QUALIFIED_MECHANISM_BENEFIT_PRESERVE_NEGATIVE_AND_USE_PREDEFINED_NEXT_OBJECTIVE_OR_STRUCTURED_BRANCH',
        D_admitted=all(r['qualified_mechanism']for r in splits.values()),physical_policy_success_proven=False)
