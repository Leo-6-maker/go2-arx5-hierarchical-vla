"""R3 prospective H1 admission: accept measured time aliases, retain all other gates."""
import json

from pi0260_r2 import ROOT,read,sha

R3=ROOT/'R3'
AUTH='fd8e69b59a16aa90b1761d23b3a8f5a78cee8763'


def route(alias_known,codec_ok,image_ok,input_ok,semantic_ok):
    # ponytail: one transition; the documented alias is evidence, not a veto.
    return 'H1_R3_ADMITTED_WITH_MEASURED_TEMPORAL_AMBIGUITY' if all((alias_known,codec_ok,image_ok,input_ok,semantic_ok)) else 'DENY_UNRESOLVED_H1_INTEGRITY'


def self_check():
    cases={
        'known_alias_and_all_other_checks':route(True,True,True,True,True),
        'bad_codec':route(True,False,True,True,True),
        'missing_image_review':route(True,True,False,True,True),
        'illegal_input':route(True,True,True,False,True),
        'wrong_semantics':route(True,True,True,True,False),
    }
    assert cases['known_alias_and_all_other_checks']=='H1_R3_ADMITTED_WITH_MEASURED_TEMPORAL_AMBIGUITY'
    assert all(v=='DENY_UNRESOLVED_H1_INTEGRITY' for k,v in cases.items() if k!='known_alias_and_all_other_checks')
    return cases


def main():
    cases=self_check()
    old=read(ROOT/'state/H1_OBSERVABILITY_BLOCKED.json')
    codec=read(ROOT/'dataset/codec_receipt.json')
    geometry=read(ROOT/'review_geometry/receipt.json')
    conditions=read(R3/'condition_audit.json')
    review=read(R3/'natural_image_review.json')
    inputs=read(R3/'input_boundary.json')
    physical=read(R3/'physical_binding_cpu.json')
    cases_eval=read(R3/'motion2_eval_cases.json')
    o64=read(R3/'O64_eval_cases.json')
    visits=read(R3/'train_visits.json')
    floor=read(R3/'exposure_floor.json')
    noise=read(R3/'noise/manifest.json')
    assert old['status']=='H1_OBSERVABILITY_BLOCKED_EXACT_INPUT_FUTURE_ALIAS' and old['H1_admitted'] is False
    assert old['main_updates']==old['quality_queries']==0
    assert codec['status']=='COMPACT_30_10' and codec['selected_post_codec']['TRAIN'] and codec['statistics_fit_count']==1
    assert len(codec['selected_post_codec']['TRAIN'])==30 and len(codec['selected_post_codec']['KIN-VAL'])==10
    assert geometry['status']=='PASS_R2_COLLECTION_READONLY_REPLAY'
    assert conditions['status']=='R3_CONDITION_AUDIT_COMPLETE' and conditions['windows']==17377
    assert conditions['cross_split_conditions']==0
    assert all(not x['mixed_semantics'] for x in conditions['partitions'].values())
    assert review['status']=='PASS_SCOPED_MANUAL_NATURAL_RGB_REVIEW'
    assert all(sha(R3/'image_review'/(task+'_contact.png'))==review['contact_sheets_sha256'][task] for task in ('M1','M3'))
    assert sha(R3/'image_review/manifest.json')==review['sample_manifest_sha256']
    assert inputs['status']=='PASS_ACTUAL_TRAIN_FREE_CONDITION_EQUALITY_AND_ALLOWLIST' and len(inputs['cases'])==8
    assert physical['status']=='PASS_R3_UNCLAMPED_TRAIN_LOSS_INFERENCE_DECODE_BINDING' and len(physical['cases'])==8
    assert len(cases_eval['cases'])==156 and len(cases_eval['semantic_pairs'])==12
    assert len(o64['rows'])==64 and len(visits['visits'])==6000 and floor['updates']==6000
    assert noise['status']=='R3_GLOBAL_B0_B1_FROZEN' and [x['seed'] for x in noise['rows']]==[26031,26032]
    registry=read(ROOT/'design/registry.json')
    family={x['family_id']:x for x in registry['families']}
    selected=codec['selected_post_codec']
    identity={split:{json.dumps(family[f]['r2_full_identity'],sort_keys=True) for f in selected[split]} for split in selected}
    assert len(identity['TRAIN'])==30 and len(identity['KIN-VAL'])==10 and not identity['TRAIN']&identity['KIN-VAL']
    status=route(True,True,True,True,True)
    files=['../state/H1_OBSERVABILITY_BLOCKED.json','../dataset/codec_receipt.json','../review_geometry/receipt.json',
           'condition_audit.json','natural_image_review.json','input_boundary.json','physical_binding_cpu.json',
           'motion2_eval_cases.json','O64_eval_cases.json','train_visits.json','exposure_floor.json','noise/manifest.json']
    hashes={name:sha((R3/name).resolve()) for name in files}
    result=dict(status=status,authority_commit=AUTH,old_H1_status=old['status'],old_H1_sha256=hashes['../state/H1_OBSERVABILITY_BLOCKED.json'],
                old_DEV_4_of_20_FAIL_retained=True,review_mode='SAME_OWNER_READONLY_REPLAY',full_observability_proven=False,
                accepted_limitation='KNOWN_TEMPORAL_ALIAS_ACCEPTED_FOR_EMPIRICAL_LEARNING',
                dataset='COMPACT_30_10',train_windows=conditions['partitions']['TRAIN']['windows'],val_windows=conditions['partitions']['KIN-VAL']['windows'],
                condition_floor_uniform_train_m=conditions['partitions']['TRAIN']['empirical_rmse_m'],
                condition_floor_uniform_val_m=conditions['partitions']['KIN-VAL']['empirical_rmse_m'],
                condition_floor_frozen_exposure_m=floor['empirical_rmse_m'],
                natural_RGB_pairs_manually_viewed=56,source_specific_geometry_replay=geometry['status'],
                unique_train_geometries=30,unique_val_geometries=10,cross_split_numeric_geometry_overlap=0,
                fixed_motion2_eval_cases=156,fixed_O64_cases=64,fixed_noise_packets=2,planned_logical_scores=1760,
                main_updates=0,model_queries=0,physics_steps=0,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE',
                next='REAL_BASE_AND_G_RECOVERY_CANARIES_THEN_BOTH_6000',source_sha256=hashes,routing_cpu_cases=cases)
    path=R3/'H1_R3_ADMITTED.json';assert not path.exists()
    path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(status=status,sha256=sha(path),fixed_motion2_eval_cases=156,fixed_O64_cases=64)))


if __name__=='__main__':main()
