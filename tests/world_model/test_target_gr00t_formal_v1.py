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


def continuation_fixture(tmp_path,endpoint=4000,start=None):
    """Explicit CPU fixtures; none are actual training or Core receipts."""
    c,m=fixture(tmp_path);previous={4000:2000,8000:4000}[endpoint]
    c.update(formal_target_endpoint=endpoint,start_step=previous if start is None else start)
    def pinned(name,value):
        path=tmp_path/name;path.write_text(json.dumps(value))
        return dict(path=str(path),sha256=digest(path))
    def checkpoint(step,target):
        return dict(schema='WM02_TARGET_GR00T_FULL_STATE_DELTA_V1',generation_id=c['science_generation_id'],
            global_step=step,cursor=32*step,recipe=profile.recipe(c['dataset_sha256'],c['statistics_sha256'],
                mode='FORMAL',variant=c['variant'],endpoint=target))
    cp=tmp_path/'production-checkpoint';cp.mkdir()
    (cp/'CHECKPOINT_COMPLETE.json').write_text(json.dumps(checkpoint(c['start_step'],
        previous if c['start_step']==previous else endpoint)))
    c['resume']=dict(path=str(cp),sha256=digest(cp/'CHECKPOINT_COMPLETE.json'))
    prior=pinned('previous-endpoint.json',checkpoint(previous,previous))
    probes={}
    for split in ('TRAIN','DEV'):
        for when in ('before','after'):
            rows=[]
            for i in range(4):
                rows.append(dict(episode=split+str(i),split=split,anchor=0,seed=i,
                    noise=dict(sha256=str(i)),command=COMMANDS[i%2],
                    model_global_step=previous-500 if when=='before'else previous,
                    model_identity=dict(generation_id=c['science_generation_id'],
                        dataset_sha256=c['dataset_sha256'],statistics_sha256=c['statistics_sha256'],
                        variant=c['variant'],mode='FORMAL',parent_manifest_sha256=profile.PARENT,
                        target_checkpoint_sha256=prior['sha256']if when=='after'else'earlier-probe'),
                    errors={h:dict(XYZ_euclidean_m=dict(mean=.0008 if when=='before'else .0004),
                        SO3_rad=dict(mean=.0001),gripper_MAE=0.)for h in ('H40','prefix8')}))
            probes[split+'_'+when]=pinned(split+'-'+when+'.json',dict(records=rows))
    package=pinned('candidate-package.json',dict(generation_id=c['science_generation_id'],
        global_step=previous,checkpoint_sha256=prior['sha256']))
    evidence=[];assignments=[]
    for i in range(32):
        arm='B0'if i%2==0 else'B1'
        spec=pinned('core-spec-'+str(i)+'.json',dict(data_role='DEV',source_sha='f'*40,
            registered_root='/synthetic-native-'+str(i),phase='collect-target-gr00t-prefix-loop-v1',
            data_parent='DEV-'+str(i//2),task_command=COMMANDS[(i//2)%2],target_gr00t_query_seeds=[i//2*2,i//2*2+1],
            target_gr00t_policy=dict(arm=arm,package=package if arm=='B1'else None)))
        assignments.append(dict(native_run='synthetic-native-'+str(i),arm=arm,split='DEV',
            parent='DEV-'+str(i//2),command=COMMANDS[(i//2)%2],seeds=[i//2*2,i//2*2+1]))
        result=pinned('core-result-'+str(i)+'.json',dict(source_sha='f'*40,
            actual_target_integrations=40503,shutdown_clean=True,status='NATIVE_STAGE_PASSED'))
        worker=pinned('core-worker-'+str(i)+'.json',dict(exit_code=0))
        lease=pinned('core-lease-'+str(i)+'.json',dict(status='CLOSED',worker_exit_code=0,spec_sha256=spec['sha256']))
        evidence.append(dict(spec=spec,result=result,worker=worker,lease=lease))
    panel=pinned('preregistered-panel.json',dict(B1_global_step=previous,
        planned_native_conditions=32,assignments=assignments))
    core=pinned('core-dev-review.json',dict(generation_id=c['science_generation_id'],
        candidate_global_step=previous,candidate_checkpoint_sha256=prior['sha256'],
        dataset_sha256=c['dataset_sha256'],statistics_sha256=c['statistics_sha256'],
        planned_conditions=32,terminal_conditions=32,all_worker_leases_closed=True,
        TEST_opened=False,WBC_changes=False,decision='CONTINUE_SAME_DATA',
        candidate_package=package,native_evidence=evidence,preregistration=panel))
    review=dict(schema='WM02_TARGET_GR00T_CONTINUATION_REVIEW_V1',generation_id=c['science_generation_id'],
        from_endpoint=previous,to_endpoint=endpoint,checkpoint=prior,probes=probes,Core_DEV_review=core)
    c['continuation_review']=pinned('continuation.json',review)
    return c,m,review


@pytest.mark.parametrize('endpoint,start',[(4000,2000),(4000,2500),(8000,4000),(8000,4250)])
def test_extended_slice_preserves_same_schedule_generation_and_complete_partial_state(tmp_path,endpoint,start):
    c,m,review=continuation_fixture(tmp_path,endpoint,start)
    r=formal.validate_request(c,m)
    assert r['endpoint']==endpoint and r['scheduler_horizon']==8000 and r['warmup_updates']==200
    seen=[]
    def child(*args,**kwargs):
        seen.append((args,kwargs));return dict(exit_code=0,global_step=endpoint,
            GR00T_updates=endpoint-start,checkpoint='fixture-only',endpoint_complete=True)
    result=formal.execute(c,tmp_path,profile.need,time.time()+3600,{},child,[],time.monotonic(),[])
    assert seen[0][0]==('formal-to-'+str(endpoint),'FORMAL',c['variant'],endpoint)
    assert seen[0][1]['start_step']==start and result['actual_GR00T_updates']==endpoint-start
    assert result['target_endpoint']==endpoint and result['formal4000_or8000_started']is True


@pytest.mark.parametrize('change',['no-review','fresh-optimizer','skip-4000','new-data','new-generation','new-schedule'])
def test_continuation_cannot_skip_actual_endpoint_or_change_learning_identity(tmp_path,change):
    c,m,review=continuation_fixture(tmp_path)
    if change=='no-review':c.pop('continuation_review')
    elif change=='fresh-optimizer':c.update(start_step=0,resume=None)
    elif change=='skip-4000':c['formal_target_endpoint']=8000
    else:
        desc=review['checkpoint'];p=Path(desc['path']);j=json.loads(p.read_bytes())
        if change=='new-data':j['recipe']['dataset_sha256']='c'*64
        elif change=='new-generation':j['generation_id']='different-generation'
        else:j['recipe']['scheduler_horizon']=4000
        p.write_text(json.dumps(j));desc['sha256']=digest(p)
        p=Path(c['continuation_review']['path']);p.write_text(json.dumps(review));c['continuation_review']['sha256']=digest(p)
    with pytest.raises(PermissionError):formal.validate_request(c,m)


@pytest.mark.parametrize('change',['no-train-improvement','DEV-position-worse','DEV-rotation-worse',
    'DEV-gripper-worse','nonfinite','wrong-noise','wrong-split','wrong-statistics','stale-step'])
def test_more_steps_require_matched_finite_improvement_without_dev_tradeoff(tmp_path,change):
    c,m,review=continuation_fixture(tmp_path)
    key='TRAIN_after'if change=='no-train-improvement'else'DEV_after'
    desc=review['probes'][key];p=Path(desc['path']);j=json.loads(p.read_bytes());row=j['records'][0]
    if change=='no-train-improvement':
        for row in j['records']:row['errors']['H40']['XYZ_euclidean_m']['mean']=.0008
    elif change=='DEV-position-worse':row['errors']['prefix8']['XYZ_euclidean_m']['mean']=.01
    elif change=='DEV-rotation-worse':row['errors']['H40']['SO3_rad']['mean']=.01
    elif change=='DEV-gripper-worse':row['errors']['H40']['gripper_MAE']=.02
    elif change=='nonfinite':row['errors']['H40']['gripper_MAE']=float('nan')
    elif change=='wrong-noise':row['noise']['sha256']='changed'
    elif change=='wrong-split':row['split']='TEST'
    elif change=='wrong-statistics':row['model_identity']['statistics_sha256']='c'*64
    else:row['model_global_step']=1500
    p.write_text(json.dumps(j));desc['sha256']=digest(p)
    p=Path(c['continuation_review']['path']);p.write_text(json.dumps(review));c['continuation_review']['sha256']=digest(p)
    with pytest.raises(PermissionError):formal.validate_request(c,m)


@pytest.mark.parametrize('key,value',[('decision','COLLECT_NEW_TRAIN'),('terminal_conditions',31),
    ('all_worker_leases_closed',False),('candidate_global_step',1500),('TEST_opened',True),
    ('candidate_checkpoint_sha256','c'*64)])
def test_open_loop_improvement_cannot_override_core_failure_review_or_unclosed_panel(tmp_path,key,value):
    c,m,review=continuation_fixture(tmp_path);d=review['Core_DEV_review'];p=Path(d['path'])
    j=json.loads(p.read_bytes());j[key]=value;p.write_text(json.dumps(j));d['sha256']=digest(p)
    p=Path(c['continuation_review']['path']);p.write_text(json.dumps(review));c['continuation_review']['sha256']=digest(p)
    with pytest.raises(PermissionError,match='SEALED_CORE_DEV_DECISION'):formal.validate_request(c,m)


@pytest.mark.parametrize('change',['missing-workers','duplicate-worker','TRAIN-diagnostic',
    'open-lease','wrong-spec-pin','zero-native','different-candidate'])
def test_core_summary_booleans_do_not_replace_actual_distinct_native_worker_evidence(tmp_path,change):
    c,m,review=continuation_fixture(tmp_path);d=review['Core_DEV_review'];p=Path(d['path']);j=json.loads(p.read_bytes())
    if change=='missing-workers':j['native_evidence']=[]
    elif change=='duplicate-worker':j['native_evidence'][1]=j['native_evidence'][3]
    else:
        key='spec'if change in('TRAIN-diagnostic','different-candidate')else'lease'if change in('open-lease','wrong-spec-pin')else'result'
        row=j['native_evidence'][1];desc=row[key];path=Path(desc['path']);v=json.loads(path.read_bytes())
        if change=='TRAIN-diagnostic':v['data_role']='TRAIN'
        elif change=='different-candidate':v['target_gr00t_policy']['package']['sha256']='c'*64
        elif change=='open-lease':v['status']='ACTIVE'
        elif change=='wrong-spec-pin':v['spec_sha256']='c'*64
        else:v['actual_target_integrations']=0
        path.write_text(json.dumps(v));desc['sha256']=digest(path)
        if key=='spec':
            lease=row['lease'];path=Path(lease['path']);v=json.loads(path.read_bytes());v['spec_sha256']=desc['sha256'];path.write_text(json.dumps(v));lease['sha256']=digest(path)
    p.write_text(json.dumps(j));d['sha256']=digest(p)
    p=Path(c['continuation_review']['path']);p.write_text(json.dumps(review));c['continuation_review']['sha256']=digest(p)
    with pytest.raises(PermissionError):formal.validate_request(c,m)
