"""Final evidence admission; quality outcomes never control completion."""
import collections
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'reports/PI0264'
FINAL=ROOT/'final'


def read(path):return json.loads(path.read_text())


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    public=read(FINAL/'PUBLIC_RECEIPT.json');manifest=read(FINAL/'MEMBER_MANIFEST.json')
    assert public['status']=='PASS_PUBLIC_ANONYMOUS_FULL_FILE_AND_MEMBER_READBACK' and public['rows']==6336
    assert public['tree_sha256']==manifest['tree_sha256'] and public['members']==len(manifest['files'])
    assert hashlib.sha256(json.dumps(manifest['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()==public['tree_sha256']
    members={r['path']:r for r in manifest['files']};assert len(members)==len(manifest['files'])
    for point in ('X2000','B2048','L2048'):
        for i in range(2112):
            for suffix in ('record.json','tensors.npz'):assert f'eval/{point}/cells/u-{i:04d}/{suffix}' in members
    for name in ('CPU_REPLAY.json','QUALIFICATION.json','MODEL_QUALITY_REPORT.md','FINAL_CLOSEOUT.json',
        'LEGACY_PANEL_ROWS.csv','CROSSED_PANEL_ROWS.csv','TRAIN_NEW_TEXT_ROWS.csv','PROSPECTIVE_TEXT_NOISE_ROWS.csv',
        'M1_ALL_PANELS_ROWS.csv','M1_FAMILY_COMMAND_ROWS.csv','M1_TEMPLATE_NUMERIC_GROUPS.csv','COLOR_COMMAND_GROUPS.csv',
        'ORIGINAL_VS_OTHER_BINDINGS.csv','LEGACY_REGRESSION_GROUPS.csv','WORDING_FLIPS.csv','TARGET_CHOICE_DECOMPOSITION.csv',
        'NOISE_BOUNDARY_CROSSINGS.csv','VISUAL_QA_RECEIPT.json','DISPLAY_FAILURE_COVERAGE.json'):
        assert sha(FINAL/name)==members[name]['sha256']
    audit=read(ROOT/'MAIN_TRAINING_AUDIT.json');cpu=read(FINAL/'CPU_REPLAY.json');q=read(FINAL/'QUALIFICATION.json')
    assert audit['status']=='PASS_BOTH_2048_MATCHED_TRAINING_INTEGRITY'
    assert audit['optimizer_updates']=={'B':2048,'L':2048} and audit['supervised_microbatches']=={'B':4096,'L':4096}
    assert 4128<=audit['raw_completed_updates']<=4384 and audit['raw_training_heads']<=110000
    assert 0<=audit['technical_completed_update_reserve_used']<=256 and audit['discarded_canary_updates']>=32
    assert audit['actual_random_packets_matched_within_unit_and_across_arms'] and audit['actual_four_step_gradient_and_mode_checks']
    assert members['raw_training_heads.jsonl']['sha256']==audit['training_heads_sha256']
    assert members['raw_completed_updates.jsonl']['sha256']==audit['completed_updates_sha256']
    pre=read(ROOT/'PROCESSED_PLAN_AUDIT.json')
    assert pre['status']=='PASS_FINITE_ACTUAL_COLOR_INPUT_LABEL_TEMPLATE_RNG_CHECKS' and len(pre['rows'])==8192
    assert set(pre['negative_controls'])=={'wrong_color','wrong_template','audit_metadata_leak','template_metadata_leak','target_metadata_leak',
        'duplicated_unit','skipped_unit','wrong_half_loss','false_cache_reuse'}
    fresh=read(ROOT/'FRESH_NOISE_MANIFEST.json');card=read(ROOT/'UNION_EVAL_CARD.json')
    assert fresh['status']=='PI0264_ACTUAL_N0_N1_BEFORE_MODEL_FROZEN' and len(fresh['rows'])==20
    assert card['schema']=='PI0264_FIXED_THREE_PANEL_6336_V1' and card['fresh_noise_manifest_sha256']==sha(ROOT/'FRESH_NOISE_MANIFEST.json')
    assert len(card['rows'])==2112 and len(card['legacy_aliases'])==1320 and card['model_queries_before_freeze']==0
    for row in fresh['rows']:assert members[row['file']]['sha256']==row['file_sha256']
    models=[]
    for arm in ('B','L'):
        canary=read(ROOT/f'{arm}_CANARY_COMPARE.json')
        assert canary['status']=='PASS_PI0264_REAL_TWO_MICROBATCH_EXACT_RESUME_CANARY'
        assert canary['units']==8 and canary['packet_equal'] and canary['mode_restored']
        assert canary['model_max_abs']<=1e-6 and canary['optimizer_max_abs']<=1e-6 and canary['scheduler_lr_abs']<=1e-12
        scope=read(ROOT/f'{arm}2048_FROZEN_SCOPE.json')
        assert scope['status']=='PASS_PI0264_SERIALIZED_PRECISION_AND_FROZEN_TENSOR_IDENTITY'
        assert len(scope['frozen'])==559 and len(scope['trainable'])==471 and any(r['changed'] for r in scope['trainable'])
        for step in (1024,2048):
            r=read(ROOT/f'{arm}{step}_FULL_READBACK.json')
            assert r['status']=='PASS_FULL_FILE_READBACK' and r['complete_trainer'] and len(r['files'])==16
            assert members[f'evidence/{arm}/checkpoint-{step}.full_readback.json']['sha256']==sha(ROOT/f'{arm}{step}_FULL_READBACK.json')
            models.append(dict(arm=arm,step=step,revision=r['revision'],tree_sha256=r['tree_sha256']))
    assert cpu['status']=='PASS_INDEPENDENT_COMPUTATION_PATH_CPU_6336_REPLAY' and cpu['rows']==6336
    assert collections.Counter(r['point'] for r in cpu['provenance'])==dict.fromkeys(('X2000','B2048','L2048'),2112)
    assert cpu['reused_donor_rows']==1152 and cpu['global_attempts']<=5440 and cpu['global_heads']<=21760
    assert cpu['new_attempts']-cpu['invalid_only_reserve_used']<=5184 and cpu['invalid_only_reserve_used']<=256
    for name,n in (('LEGACY_PANEL_ROWS.csv',680),('CROSSED_PANEL_ROWS.csv',640),('TRAIN_NEW_TEXT_ROWS.csv',720),('PROSPECTIVE_TEXT_NOISE_ROWS.csv',240)):
        with (FINAL/name).open(newline='') as stream:rows=list(csv.DictReader(stream))
        assert collections.Counter(r['point'] for r in rows)==dict.fromkeys(('X2000','B2048','L2048'),n)
    assert q['selected_checkpoint']==q['deployment_selected_checkpoint']=='NONE'
    assert {r['point'] for r in q['points']}=={'X2000','B2048','L2048'}
    assert q['synthetic_research_candidates']==[r['point'] for r in q['points'] if r['point'] in ('B2048','L2048') and r['combined_pass']]
    assert all(r['combined_pass']==(r['legacy_full_quality_pass'] and r['crossed_M1_robustness_pass'] and r['prospective_text_noise_pass']) for r in q['points'])
    video=read(FINAL/'VIDEO_RECEIPT.json');visual=read(FINAL/'VISUAL_QA_RECEIPT.json');display=read(ROOT/'DISPLAY_CARD.json')
    assert video['status']=='PASS_FIXED_ARRAY_VIDEO_DECODE' and video['frames']==40*video['chapters']
    assert video['semantic_margins_checked_against_CPU']==3*video['chapters']
    assert len(display['chapters'])==54 and video['chapters']>=54
    assert visual['status']=='PASS_REPRESENTATIVE_CHAPTER_VISUAL_QA' and visual['video_sha256']==video['sha256']
    frames=read(FINAL/'QA_FRAME_INDEX.json')
    assert frames['video_sha256']==video['sha256'] and frames['chapter_count']==video['chapters']
    assert frames['reviewed_frame_indices']==[40*i+20 for i in range(video['chapters'])]
    assert visual['reviewed_chapter_count']==video['chapters'] and len(frames['files'])==12
    assert members['eval/media/QA_FRAME_INDEX.json']['sha256']==sha(FINAL/'QA_FRAME_INDEX.json')
    for frame in frames['files']:
        assert members['eval/media/qa/'+frame['path']]['sha256']==frame['sha256']
    coverage=read(FINAL/'DISPLAY_FAILURE_COVERAGE.json')
    assert coverage['chapters'][:54]==display['chapters']
    chapter_uids={r['uid'] for r in coverage['chapters']}
    with (FINAL/'M1_ALL_PANELS_ROWS.csv').open(newline='') as stream:scored=list(csv.DictReader(stream))
    expected=[]
    for point in ('X2000','B2048','L2048'):
        for panel in ('A_CROSSED','B','C'):
            failed=sorted((r for r in scored if r['point']==point and r['panel']==panel and (r['correct']=='False' or r['reject']=='True')),key=lambda r:r['uid'])
            if failed:
                expected.append(dict(point=point,panel=panel,uid=failed[0]['uid'],reason='incorrect target or REJECT'))
                assert failed[0]['uid'] in chapter_uids
    assert coverage['failures']==expected
    assert members['eval/media/PI0264_X_B_L_fixed_arrays.mp4']['sha256']==video['sha256']
    checks={
        'fixed256_source_units_eight_visits':'PAIRING_PLAN.json;8192 actual processed samples; unchanged COMPACT_30_10/processor',
        'nine_negative_controls':'PROCESSED_PLAN_AUDIT.json',
        'prospective_inputs_and20_actual_noise_packets_sealed_before_canary':'UNION_EVAL_CARD.json; FRESH_NOISE_MANIFEST.json; EVAL_TEXT_INPUT_CHECKS.json',
        'both_real_exact_restore_canaries':'B/L_CANARY_COMPARE.json and actual packet audits',
        'both_complete2048_and_full_S4_scope':'MAIN_TRAINING_AUDIT.json; serialized frozen scope; actual gradient/mode/loss receipts',
        'four_recoverable_full_trainers':'Four fixed HF revisions and complete16-file readbacks, Corelab model-only',
        '6336_union_and_all_panel_views':'Full raw6336 record/tensor pairs, legacy2040/crossed1920/newTRAIN2160/prospective720 memberships',
        'three_separate_qualifications':'QUALIFICATION.json; each numeric/semantic gate independently preserved',
        'independent_computation_path':'Same owner CPU replay, not external signoff',
        'template_color_command_family_noise_M3_O64_g_b':'Complete tables and source-bound model-quality report',
        'fixed_array_video_and_failure_coverage':'54 fixed chapters plus frozen-rule first-failure additions,0 new queries/renders/physics',
        'public_anonymous_every_member_readback':'PUBLIC_RECEIPT.json and MEMBER_MANIFEST.json including all20 actual noise NPZs',
        'closed_admission':'Historical PI0263 COMPLETE/FAIL; selected/deployment NONE; stop for new-scene or physical admission'}
    result=dict(schema='PI0264_FINAL_COMPLETION_ADMISSION_AUDIT_V1',status='PI0264_LANGUAGE_DIVERSITY_TRANSFER_PILOT_COMPLETE',
        stop='STOP_FOR_NEW_SCENE_OR_PHYSICAL_TRANSFER_ADMISSION',all_required_deliverables_verified=True,requirements=checks,
        quality={r['point']:'PASS' if r['combined_pass'] else 'VALID_FAIL' for r in q['points']},
        synthetic_research_candidates=q['synthetic_research_candidates'],selected_checkpoint='NONE',deployment_selected_checkpoint='NONE',
        models=models,public_revision=public['revision'],public_tree_sha256=public['tree_sha256'],public_members=public['members'],
        receipt_sha256=sha(FINAL/'PUBLIC_RECEIPT.json'),report_sha256=sha(FINAL/'MODEL_QUALITY_REPORT.md'),
        final_audit_code_sha256=sha(Path(__file__)),next_scientific_execution_authorized=False,model_queries_by_final_audit=0,physics_steps=0)
    write_path=FINAL/'FINAL_COMPLETION_AUDIT.json';write_path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
