"""Synthetic production gates/partial-state tests; no PBS or model execution."""
import copy,json,time
from pathlib import Path
import pytest
from go2arx5_vla.world_model import target_gr00t_formal_v1 as formal
from go2arx5_vla.world_model import target_gr00t_profile_v1 as profile
from go2arx5_vla.world_model.target_gr00t_collection_v1 import COMMANDS
from go2arx5_vla.world_model.storage import digest


def fixture(tmp_path):
    q=tmp_path/'qual';q.mkdir();data=tmp_path/'data';data.mkdir()
    result=dict(status='FIT4_CHAIN_SEALED',fit4_learnability_passed=True,
        selected_variant='INTERFACES_LAST2_DIT',actual_GR00T_updates=203)
    (q/'ALLOCATION_NATIVE_FINAL.json').write_text(json.dumps(result))
    (q/'NEXT_UPDATE_RESTORE_ACTUAL.json').write_text(json.dumps(dict(
        status='PASS_ACTUAL_UNINTERRUPTED_VERSUS_RESUMED_NEXT_UPDATE',
        next_update_parameters_optimizer_scheduler_RNG_sampler_exact=True)))
    (q/'ALLOCATION_CONFIG.json').write_text(json.dumps(dict(parent=dict(files={'weights':'test-only-sha'}))))
    rows=[dict(split='TRAIN',parent=str(p),command=cmd)for p in range(4)for cmd in COMMANDS for restart in range(2)]
    rows += [dict(split='DEV',parent='DEV'+str(p),command=cmd)for p in range(4)for cmd in COMMANDS]
    manifest=dict(episodes=rows,bundle_kind='FORMAL',TRAIN_only_statistics=True,
        TEST_results_read=False,WM_or_HOLD_or_diagnostic_relabelled=False)
    (data/'DATASET_MANIFEST.json').write_text(json.dumps(manifest))
    c=dict(allocation_mode='FORMAL_SLICE',formal_target_endpoint=2000,start_step=0,
        science_generation_id='new-formal-not-diagnostic',dataset_sha256='a'*64,
        statistics_sha256='b'*64,variant='INTERFACES_LAST2_DIT',parent=dict(files={'weights':'test-only-sha'}),
        qualification=dict(path=str(q/'ALLOCATION_NATIVE_FINAL.json'),sha256=digest(q/'ALLOCATION_NATIVE_FINAL.json')),
        parent_manifest_sha256=profile.PARENT,resume=None,bindings=dict(dataset=str(data)))
    return c,manifest


def test_formal_keeps_fixed_schedule_and_requires_actual_gate_and_next_update(tmp_path):
    c,m=fixture(tmp_path);r=formal.validate_request(c,m)
    assert r['scheduler_horizon']==8000 and r['warmup_updates']==200 and r['endpoint']==2000
    q=Path(c['qualification']['path']);v=json.loads(q.read_text());v['fit4_learnability_passed']=False;q.write_text(json.dumps(v));c['qualification']['sha256']=digest(q)
    with pytest.raises(PermissionError,match='ACTUAL_FIT4_LEARNABILITY'):formal.validate_request(c,m)


@pytest.mark.parametrize('change',['tiny-data','no-dev','same-layout','diagnostic','different-parent','dev-leakage'])
def test_production_cannot_start_from_small_or_mismatched_inputs(tmp_path,change):
    c,m=fixture(tmp_path)
    if change=='tiny-data':m['episodes']=m['episodes'][:4]+m['episodes'][16:]
    elif change=='no-dev':m['episodes']=m['episodes'][:16]
    elif change=='same-layout':
        for r in m['episodes'][:16]:r['parent']='ONE'
    elif change=='diagnostic':m['bundle_kind']='FIT4_DIAGNOSTIC'
    elif change=='dev-leakage':m['episodes'][-1]['parent']=m['episodes'][0]['parent']
    else:c['parent']['files']['weights']='different'
    with pytest.raises(PermissionError):formal.validate_request(c,m)


def test_partial_allocation_is_not_the_scientific_2000_endpoint(tmp_path):
    c,m=fixture(tmp_path);seen=[]
    def child(*args,**kwargs):
        seen.append((args,kwargs));return dict(exit_code=0,global_step=250,GR00T_updates=250,
            checkpoint='/synthetic/checkpoint250',endpoint_complete=False)
    result=formal.execute(c,tmp_path,profile.need,time.time()+3600,{},child,[],time.monotonic(),[])
    assert result['status']=='FORMAL_PARTIAL_COMPLETE_CHECKPOINT_SEALED'and not result['endpoint_complete']
    assert result['actual_GR00T_updates']==250 and result['formal4000_or8000_started']is False
    assert seen[0][0][1:] == ('FORMAL','INTERFACES_LAST2_DIT',2000)
    assert seen[0][1]['generation']=='new-formal-not-diagnostic'and seen[0][1]['start_step']==0


def test_diagnostic_checkpoint_cannot_resume_production(tmp_path):
    c,m=fixture(tmp_path);cp=tmp_path/'cp';cp.mkdir();saved=dict(schema='WM02_TARGET_GR00T_FULL_STATE_DELTA_V1',
        generation_id=c['science_generation_id'],global_step=250,cursor=8000,
        recipe=dict(mode='FIT4',dataset_sha256=c['dataset_sha256'],statistics_sha256=c['statistics_sha256'],variant=c['variant']))
    (cp/'CHECKPOINT_COMPLETE.json').write_text(json.dumps(saved));c.update(start_step=250,
        resume=dict(path=str(cp),sha256=digest(cp/'CHECKPOINT_COMPLETE.json')))
    with pytest.raises(PermissionError,match='SAME_FORMAL_GENERATION'):formal.validate_request(c,m)


def test_unmeasured_larger_batch_cannot_be_smuggled_into_production(tmp_path):
    c,m=fixture(tmp_path);c['microbatch']=2
    with pytest.raises(PermissionError,match='ACTUALLY_QUALIFIED_MICROBATCH1'):formal.validate_request(c,m)
