"""Actual private design/budget bytes; Teacher outcomes remain CPU fixtures."""
import base64,json,os
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.generalization import amd_pilot_scope as m
from scripts.pi0294.pv1 import mix_accounting as a

W=Path(os.environ.get('PI0294_PRIVATE_WORKSPACE',str(Path(__file__).resolve().parents[4])))
B=W/'work/private/v5-execution-20261025'
P=W/'work/private/science-next/wdr1-target-identity/v4a-core-eval-preparation-01/GEN1_V5_PREVIEW_V2_1792914629517993678.private.json'
if not P.exists():pytest.skip('actual private design packet unavailable',allow_module_level=True)

def results(plan):
    return [dict(r,terminal=True,unknown_effects=[],native_safety_failures=0,admitted_for_training=True,full_task_success=True,
        independent_raw_audit='PASS_INDEPENDENT_RAW_AUDIT',codec_status='PASS_FIXED_ORIGINAL_V3_0_NO_SILENT_CLIP',
        statistics_sha256=a.STATS,H40_windows=601,Teacher_takeover_of_student=False)for r in plan['attempts']]

def test_actual_fixed_twelve_first_attempts_and_independent_parent():
    p=m.pilot_plan(P.read_bytes())
    assert len(p['attempts'])==12 and len({r['geometry_id']for r in p['attempts']})==6
    assert {r['task']for r in p['attempts']}=={'T1','T2'}
    assert p['preview_images']==24 and p['training_visuals']==['V0']
    assert p['train']['initial_checkpoint_manifest_sha256']==a.PARENT
    assert p['train']['updates']==1024 and p['replacement_attempts']==0 and not p['native_admission']
    assert p==m.pilot_plan(P.read_bytes())

def test_changed_packet_rejected():
    with pytest.raises(PermissionError):m.pilot_plan(P.read_bytes()+b' ')

def test_fixture_success_gate_never_grants_training():
    p=m.pilot_plan(P.read_bytes());v=m.training_support(p,results(p))
    assert len(v['admitted'])==12 and not v['missing']and not v['physics_or_training_authorized']

def test_missing_task_stratum_blocks_training_keeps_failures():
    p=m.pilot_plan(P.read_bytes());rows=results(p)
    for r in rows:
        if r['task']=='T2'and r['layout_stratum']=='+Y/NEAR':r['admitted_for_training']=False
    v=m.training_support(p,rows)
    assert v['status']=='DATA_SUPPORT_INCOMPLETE_NO_TRAINING'and v['missing']==[('T2','+Y/NEAR')]

@pytest.mark.parametrize('bad',['drop','unknown','test_split','R0','codec','geometry','native_safety'])
def test_partial_unknown_or_forged_data_rejected(bad):
    p=m.pilot_plan(P.read_bytes());rows=results(p)
    if bad=='drop':rows.pop()
    elif bad=='unknown':rows[0]['unknown_effects']=['unknown tick']
    elif bad=='test_split':rows[0]['study_split']='G_TEST'
    elif bad=='R0':rows[0]['statistics_sha256']='0'*64
    elif bad=='codec':rows[0]['codec_status']='BLOCKED'
    elif bad=='native_safety':rows[0]['native_safety_failures']=1;rows[0]['admitted_for_training']=False
    else:rows[0]['geometry_id']='FORGED'
    with pytest.raises(PermissionError):m.training_support(p,rows)

def test_actual_captured_whole_future_forecast_preserves_costs_clock_and_protection():
    path=B/'C50_VANDA_EVAL_COLD_02_C50_AFFINITY_PREENTRY_FAILURE_1497143_ONCE_GLOBAL_SETTLEMENT.private.json'
    if not path.exists():pytest.skip('actual closed cost capture unavailable')
    gb=base64.b64decode(json.loads(path.read_bytes())['value']['GLOBAL_base64'])
    lb=base64.b64decode(json.loads((B/'VANDA_COHOST_PREPARED.private.json').read_bytes())['historical_LOCAL_base64'])
    v=m.full_forecast(gb,lb)
    assert not v['scope_adopted']and not v['native_admission']and not v['old_costs_refunded']
    assert v['T_CD']==1792932749.8133287 and v['deadline']==1793537549.8133287
    assert v['dimensions']['H200_s']['projected']==421663
    assert v['dimensions']['PBS_submissions']['projected']==36
    assert v['dimensions']['optimizer_updates']['projected']==14786
    changes=v['cap_changes_required']
    assert set(changes)=={'PBS_submissions','Hopper_storage_bytes','Core_storage_bytes','transfer_bytes'}
    assert {k:changes[k]['after_GiB']for k in ('Hopper_storage_bytes','Core_storage_bytes','transfer_bytes')}==dict(Hopper_storage_bytes=1176,Core_storage_bytes=477,transfer_bytes=332)
    assert v['matrix3_and_final20GPUh40GiB_retained']

@pytest.mark.parametrize('difference',[None,'state','RGB','camera'])
def test_paired_task_native_inspection_seam_checks_state_RGB_and_calibration(monkeypatch,difference):
    from scripts.pi0294.generalization import static_preview as v
    spec=next(iter(v.specifications(json.loads(P.read_bytes())).values()))['V0']
    row=dict(canonical_t0_state_sha256='state',all_other_XML_sha256='xml',native_dimensions=dict(nq=1),
        arm_q=[0],actual_initial_tcp=[0,0,0],table_rgba_xml='color',
        cameras={c:dict(snapshot_id='t1',calibration=[1])for c in v.CAMERAS},
        raw_RGB_sha256={c:'pixels'for c in v.CAMERAS})
    calls=[];charged=[]
    def inspect(card,asset_root,*,construct,guard,save_images=None):
        construct();guard();calls.append(card['task']);out=deepcopy(row)
        if card['task']=='T2':
            for c in v.CAMERAS:out['cameras'][c]['snapshot_id']='t2'
            if difference=='state':out['canonical_t0_state_sha256']='changed'
            elif difference=='RGB':out['raw_RGB_sha256']['body_rgb']='changed'
            elif difference=='camera':out['cameras']['body_rgb']['calibration']=[2]
        return out
    monkeypatch.setattr(v,'inspect',inspect)
    invoke=lambda:v.inspect_task_pair(spec,'fixture-only',construct=lambda:charged.append(1),guard=lambda:None)
    if difference is None:
        proof=invoke();assert proof['actual_task_pair_reset_checked']and proof['integrations']==0
    else:
        with pytest.raises(PermissionError):invoke()
    assert calls==['T1','T1','T2']and len(charged)==3
