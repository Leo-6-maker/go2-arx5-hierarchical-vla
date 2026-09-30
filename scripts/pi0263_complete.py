"""Final admission audit against actual local receipts and the public member seal."""
import collections
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'reports/PI0263'
FINAL=ROOT/'final'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    public=read(FINAL/'PUBLIC_RECEIPT.json');manifest=read(FINAL/'MEMBER_MANIFEST.json')
    assert public['status']=='PASS_PUBLIC_ANONYMOUS_FULL_FILE_AND_MEMBER_READBACK'
    assert public['tree_sha256']==manifest['tree_sha256'] and public['members']==len(manifest['files'])
    assert hashlib.sha256(json.dumps(manifest['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()==public['tree_sha256']
    members={r['path']:r for r in manifest['files']};assert len(members)==len(manifest['files'])
    for point in ('P2000','F2000','X2000'):
        for i in range(1152):
            for suffix in ('record.json','tensors.npz'):
                assert f'eval/{point}/cells/u-{i:04d}/{suffix}' in members
    for name in ('CPU_REPLAY.json','QUALIFICATION.json','MODEL_QUALITY_REPORT.md','FINAL_CLOSEOUT.json',
        'LEGACY_PANEL_ROWS.csv','CROSSED_PANEL_ROWS.csv','COLOR_COMMAND_GROUPS.csv','ORIGINAL_VS_OTHER_BINDINGS.csv',
        'LEGACY_REGRESSION_GROUPS.csv','WORDING_FLIPS.csv','TARGET_CHOICE_DECOMPOSITION.csv','NOISE_BOUNDARY_CROSSINGS.csv',
        'VISUAL_QA_RECEIPT.json'):
        assert sha(FINAL/name)==members[name]['sha256']
    audit=read(ROOT/'MAIN_TRAINING_AUDIT.json');cpu=read(FINAL/'CPU_REPLAY.json');q=read(FINAL/'QUALIFICATION.json')
    assert audit['status']=='PASS_BOTH_2000_MATCHED_TRAINING_INTEGRITY'
    assert audit['optimizer_updates']=={'F':2000,'X':2000} and audit['supervised_microbatches']=={'F':4000,'X':4000}
    assert audit['raw_completed_updates']==4032<=4288 and audit['raw_training_heads']==72624<=100000
    assert audit['technical_completed_update_reserve_used']==0 and audit['discarded_canary_updates']==32
    assert audit['actual_random_packets_matched_within_unit_and_across_arms'] and audit['actual_four_step_gradient_and_mode_checks']
    assert members['raw_training_heads.jsonl']['sha256']==audit['training_heads_sha256']
    assert members['raw_completed_updates.jsonl']['sha256']==audit['completed_updates_sha256']
    pre=read(ROOT/'PROCESSED_PLAN_AUDIT.json')
    assert pre['status']=='PASS_FINITE_ACTUAL_COLOR_INPUT_LABEL_TEMPLATE_RNG_CHECKS' and len(pre['rows'])==8000
    assert set(pre['negative_controls'])=={'wrong_color','wrong_template','audit_metadata_leak','duplicated_unit','skipped_unit','wrong_half_loss','false_cache_reuse'}
    models=[]
    for arm in ('F','X'):
        canary=read(ROOT/f'{arm}_CANARY_COMPARE.json')
        assert canary['status']=='PASS_PI0263_REAL_TWO_MICROBATCH_EXACT_RESUME_CANARY'
        assert canary['units']==8 and canary['packet_equal'] and canary['mode_restored']
        assert canary['model_max_abs']==canary['optimizer_max_abs']==canary['scheduler_lr_abs']==0
        scope=read(ROOT/f'{arm}2000_FROZEN_SCOPE.json')
        assert scope['status']=='PASS_PI0263_SERIALIZED_PRECISION_AND_FROZEN_TENSOR_IDENTITY'
        assert len(scope['frozen'])==559 and len(scope['trainable'])==471 and all(r['changed'] for r in scope['trainable'])
        for step in (1000,2000):
            r=read(ROOT/f'{arm}{step}_FULL_READBACK.json')
            assert r['status']=='PASS_FULL_FILE_READBACK' and r['complete_trainer'] and len(r['files'])==16
            assert members[f'evidence/{arm}/checkpoint-{step}.full_readback.json']['sha256']==sha(ROOT/f'{arm}{step}_FULL_READBACK.json')
            models.append(dict(arm=arm,step=step,revision=r['revision'],tree_sha256=r['tree_sha256']))
    assert cpu['status']=='PASS_INDEPENDENT_COMPUTATION_PATH_CPU_3456_REPLAY' and cpu['rows']==3456
    assert collections.Counter(r['point'] for r in cpu['provenance'])==dict.fromkeys(('P2000','F2000','X2000'),1152)
    assert cpu['reused_donor_rows']==1152 and cpu['new_attempts']==2232<=2304
    assert cpu['global_attempts']==2232<=2432 and cpu['global_heads']==8928<=9728
    assert cpu['invalid_only_reserve_used']==0 and cpu['new_endpoint_exact_forward_reuses']==72
    for name,n in (('LEGACY_PANEL_ROWS.csv',680),('CROSSED_PANEL_ROWS.csv',640)):
        with (FINAL/name).open(newline='') as stream:rows=list(csv.DictReader(stream))
        assert collections.Counter(r['point'] for r in rows)==dict.fromkeys(('P2000','F2000','X2000'),n)
    assert q['selected_checkpoint']==q['deployment_selected_checkpoint']==q['synthetic_research_candidate']=='NONE'
    assert {r['point'] for r in q['points']}=={'P2000','F2000','X2000'} and all(not r['combined_pass'] for r in q['points'])
    video=read(FINAL/'VIDEO_RECEIPT.json');visual=read(FINAL/'VISUAL_QA_RECEIPT.json')
    assert video['status']=='PASS_FIXED_ARRAY_VIDEO_DECODE' and video['frames']==720 and visual['chapters']==18
    assert visual['status']=='PASS_REPRESENTATIVE_CHAPTER_VISUAL_QA' and visual['video_sha256']==video['sha256']
    assert members['eval/media/PI0263_P_F_X_fixed_arrays.mp4']['sha256']==video['sha256']
    checks={
        'source_bound_diagnosis_and_frozen_schedule':'PROCESSED_PLAN_AUDIT.json; PAIRING_PLAN.json; MODEL_QUALITY_REPORT.md',
        'finite_actual_negative_controls':'PROCESSED_PLAN_AUDIT.json: seven rejected controls,8000 samples',
        'both_real_exact_restore_canaries':'F_CANARY_COMPARE.json; X_CANARY_COMPARE.json; actual packet audits',
        'both_complete_final2000_and_original_objective':'MAIN_TRAINING_AUDIT.json; frozen tensor scope; four full Trainer HF readbacks',
        'full3456_union_and_both_panel_views':'CPU_REPLAY.json;2040 legacy memberships;1920 crossed memberships; all3456 raw record/tensor pairs in public seal',
        'independent_computation_path_cpu':'Same owner, not external signoff; preserved raw CPU scope erratum',
        'family_wording_direction_color_noise_and_regressions':'MODEL_QUALITY_REPORT.md and all eight row/stratum diagnostic CSVs',
        'recoverable_public_full_trainer':'All four checkpoints sixteen files, fixed-revision full-file readback',
        'public_results_and_anonymous_every_member_readback':'PUBLIC_RECEIPT.json; MEMBER_MANIFEST.json; immutable revision/tree',
        'saved_array_video':'18 fixed chapters,720 decoded frames, visual watermarks,0 new queries/renders/physics',
        'budgets_and_valid_fail':'4032 completed updates;72624 training heads;2232 queries/8928 inference heads; both valid quality FAIL',
        'closed_admission':'selected/deployment/candidate NONE; stop for independent generalization or physical admission'}
    result=dict(schema='PI0263_FINAL_COMPLETION_ADMISSION_AUDIT_V1',
        status='PI0263_TEMPLATE_COUNTERBALANCING_PILOT_COMPLETE',stop='STOP_FOR_INDEPENDENT_GENERALIZATION_OR_PHYSICAL_ADMISSION',
        quality={'F2000':'VALID_FAIL','X2000':'VALID_FAIL'},selected_checkpoint='NONE',deployment_selected_checkpoint='NONE',
        synthetic_research_candidate='NONE',all_required_deliverables_verified=True,requirements=checks,models=models,
        public_revision=public['revision'],public_tree_sha256=public['tree_sha256'],public_members=public['members'],
        receipt_sha256=sha(FINAL/'PUBLIC_RECEIPT.json'),report_sha256=sha(FINAL/'MODEL_QUALITY_REPORT.md'),
        public_readback_status=public['status'],final_audit_code_sha256=sha(Path(__file__)),
        next_scientific_execution_authorized=False,model_queries_by_final_audit=0,physics_steps=0)
    (FINAL/'FINAL_COMPLETION_AUDIT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
