"""Synthetic label/receipt contracts; no native Teacher/codec claim."""
from copy import deepcopy
import os
from pathlib import Path
import pytest
from scripts.pi0294.generalization import amd_export as e, amd_pilot_scope as p, amd_capture as c
from scripts.pi0294.test_pv1_export import fixture
from scripts.pi0294.pv1.export import packet as legacy_packet

W=Path(os.environ.get('PI0294_PRIVATE_WORKSPACE',str(Path(__file__).resolve().parents[4])))
P=W/'work/private/science-next/wdr1-target-identity/v4a-core-eval-preparation-01/GEN1_V5_PREVIEW_V2_1792914629517993678.private.json'
if not P.exists():pytest.skip('private frozen packet unavailable',allow_module_level=True)


def test_same_committed_label_math_but_exact_gen_card_and_causal_identity():
    old,samples=fixture();card=c.cards(P.read_bytes(),0)[0]
    for s in samples:
        s['policy_obs']['instruction']=card['instruction']
        s['causal_history']['episode_id']=card['id']
    a=e.packet(samples,card,P.read_bytes())
    assert a['committed_absolute10'].shape==(640,10) and a['valid_window'].sum()==601
    assert abs(a['committed_absolute10'][0,2]-a['actual_state10'][0,2]-.002)<1e-12
    with pytest.raises(PermissionError):legacy_packet(samples,card)
    for field,value in [('split','TRAIN'),('visual_profile','V1'),('nominal_clearance_m',.012),('GEN1_lineage_id','another')]:
        with pytest.raises(PermissionError):e.packet(samples,dict(card,**{field:value}),P.read_bytes())
    samples[13]['causal_history']['episode_id']='OLD_PV1'
    with pytest.raises(PermissionError):e.packet(samples,card,P.read_bytes())


def receipts():
    captured=[];exported=[]
    for i,x in enumerate(p.pilot_plan(P.read_bytes())['attempts']):
        h=f'{i:064x}'
        captured.append(dict(x,unknown_effects=[],terminal=True,native_safety_failures=0,
            full_task_success=True,independent_raw_audit='PASS_INDEPENDENT_RAW_AUDIT',
            raw_manifest_sha256=h,audit_sha256=h))
        exported.append(dict(id=x['attempt_id'],**{k:x[k]for k in('task','geometry_id','design_hash','study_split','visual_profile')},
            source_raw_manifest_sha256=h,source_full_audit_sha256=h,source_packet_sha256=p.PACKET_SHA,
            statistics_sha256=p.a.STATS,admitted_for_training=True,
            status='PASS_SAVED_TEACHER_EXPORT_AND_OFFICIAL_CHAIN',full_task_success=True,
            independent_raw_admitted=True,windows=601,
            codec=dict(status='PASS_FIXED_ORIGINAL_V3_0_NO_SILENT_CLIP',statistics_refit=False,
                encode_clip_components=0,native_decode_unchanged=True,H40_windows_actually_checked=601)))
    return captured,exported


def test_teacher_success_without_official_export_cannot_enable_training():
    captured,exported=receipts()
    result=e.support_after_export(P.read_bytes(),captured,[])
    assert result['status']=='DATA_SUPPORT_INCOMPLETE_NO_TRAINING' and len(result['missing'])==8
    result=e.support_after_export(P.read_bytes(),captured,exported)
    assert result['status']=='DATA_SUPPORT_READY_FOR_FRESH_TRAINING_ADMISSION'
    assert not result['native_admission'] and result['raw_custody_and_native_resource_admission_still_required']


def test_valid_failures_retained_and_stratum_missing_stops_training():
    captured,exported=receipts()
    rejected={x['attempt_id']for x in captured if x['task']=='T2'and x['layout_stratum']=='+Y/NEAR'}
    for x in captured:
        if x['attempt_id']in rejected:x.update(full_task_success=False,independent_raw_audit='KNOWN_S1_REJECT')
    exported=[x for x in exported if x['id']not in rejected]
    v=e.support_after_export(P.read_bytes(),captured,exported)
    assert v['missing']==[('T2','+Y/NEAR')] and v['all_first_attempts_retained']


@pytest.mark.parametrize('bad',['drop','duplicate','foreign','manifest','audit','unsafe','unknown','R0','split','visual','codec','clip','H40'])
def test_wrong_join_scope_or_codec_rejected(bad):
    captured,exported=receipts()
    if bad=='drop':captured.pop()
    elif bad=='duplicate':exported.append(deepcopy(exported[0]))
    elif bad=='foreign':exported[0]['id']='foreign'
    elif bad=='manifest':exported[0]['source_raw_manifest_sha256']='f'*64
    elif bad=='audit':exported[0]['source_full_audit_sha256']='f'*64
    elif bad=='unsafe':captured[0]['native_safety_failures']=1
    elif bad=='unknown':captured[0]['unknown_effects']=['unknown tick']
    elif bad=='R0':exported[0]['statistics_sha256']='f'*64
    elif bad=='split':exported[0]['study_split']='G_TEST'
    elif bad=='visual':exported[0]['visual_profile']='V1'
    elif bad=='codec':exported[0]['codec']['status']='BLOCKED'
    elif bad=='clip':exported[0]['codec']['encode_clip_components']=1
    elif bad=='H40':exported[0]['codec']['H40_windows_actually_checked']=6
    with pytest.raises(PermissionError):e.support_after_export(P.read_bytes(),captured,exported)
