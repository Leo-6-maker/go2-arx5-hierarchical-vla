"""Synthetic cold filesystems only, not native model/physics qualification."""
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.pi0294.native_dataset import sha
from scripts.pi0294.pv1 import evaluation_platform as e
from scripts.pi0294.pv1 import neural_scope as scope
from scripts.pi0294.pv1 import runtime
from scripts.pi0294.pv1 import evaluation_plan as plan
from scripts.pi0294.run_admission import validate as validate_entry
from scripts.pi0294.test_pv1_neural_scope import prepared_fixture


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():path.chmod(0o600)
    path.write_text(json.dumps(value));path.chmod(0o400)


@pytest.fixture
def cold(tmp_path):
    root,code,worker,digest,_=prepared_fixture(tmp_path)
    parent=root.parent.parent
    m=json.loads((root/'RUN_MANIFEST.json').read_text())
    m.update(platform_evaluation_version=e.VERSION,scope=e.ROLE,owner_user='e1192548')
    role=e.ROLE+'__'+root.name
    m['execution'].update(profile=e.PROFILE,private_path_role=role,control_hostnames=['stdct-login-01'])
    b=json.loads((root/'OUTPUT_BINDING.private.json').read_text());b['role']=role
    put(root/'OUTPUT_BINDING.private.json',b)
    put(Path(b['registry']),dict(entries=[dict(role=role,path=str(root),run_uuid=root.name)]))
    m['execution']['output_binding_sha256']=sha(root/'OUTPUT_BINDING.private.json')
    a=json.loads((parent/'PV1_ADOPTION_AND_RESOURCE_BINDING.json').read_text())
    a.update(run_uuid='20261018T182956Z-localpc-pv1',T0_epoch=e.T0,
        stop_new_jobs_epoch=e.STOP_NEW,stop_new_training_epoch=e.T0+156*3600,
        stop_new_physics_epoch=e.STOP_PHYSICS,hard_closeout_epoch=e.HARD_CLOSE)
    put(parent/'PV1_ADOPTION_AND_RESOURCE_BINDING.json',a)
    pl=json.loads((parent/'leases/parent.json').read_text());pl.update(run_uuid=a['run_uuid'],
        T0_epoch=e.T0,deadline_epoch=e.HARD_CLOSE)
    put(parent/'leases/parent.json',pl)
    put(parent/'OUTPUT_BINDING.private.json',dict(lease=str(parent/'leases/parent.json'),run_uuid=a['run_uuid']))
    lease=json.loads(Path(b['lease']).read_text());lease['deadline_epoch']=e.T0+110000
    put(Path(b['lease']),lease)
    stage=json.loads((root/'PV1_STAGE_BINDING.json').read_text())
    stage.update(campaign_run_uuid=a['run_uuid'],route='COVERAGE_FIRST')
    put(root/'PV1_STAGE_BINDING.json',stage)
    population=Path(__file__).resolve().parents[2]/'reports/PI0294/PV1/20261018T182956Z-localpc-pv1/PV1_DATA_PLAN_AND_SPLIT_SEAL.json'
    p=root/plan.POPULATION_FILE;p.parent.mkdir(exist_ok=True)
    p.write_bytes(population.read_bytes());p.chmod(0o400)
    groups=json.loads(p.read_text())['groups']
    development=plan.development(groups,route=stage['route'],candidate='B0_CORR',
        new_bank_manifest_sha256=stage['new_bank_manifest_sha256'])
    put(root/plan.PLAN,development)
    put(root/'plans/PRIOR_EVALUATION_OUTCOMES.json',[])
    helper=code/'scripts/pi0294/pv1/evaluation_platform.py'
    helper.parent.mkdir(parents=True,exist_ok=True);helper.write_bytes(Path(e.__file__).read_bytes())
    seal=code.parent/'SOURCE_SEAL.json';sv=json.loads(seal.read_text())
    sv['files'][str(helper.relative_to(code))]=sha(helper);put(seal,sv)
    m['code']['source_seal_sha256']=sha(seal)
    for name in ('PV1_STAGE_BINDING.json',plan.PLAN,plan.POPULATION_FILE,'plans/PRIOR_EVALUATION_OUTCOMES.json'):
        m['execution']['sealed_inputs'][name]=sha(root/name)
    m['limits'].update(PV1_T0_epoch=e.T0,PV1_hard_close_epoch=e.HARD_CLOSE,
        operation_deadline_epoch=lease['deadline_epoch'])
    def manifest():
        put(root/'RUN_MANIFEST.json',m);put(worker,m)
    manifest()
    return dict(root=root,code=code,worker=worker,digest=digest,helper=helper,m=m,stage=stage,
        manifest=manifest,now=e.T0+100100,b=b,population_fixture_sha256=sha(root/plan.POPULATION_FILE))


def guards(c):
    """Fake host/source metadata is intentionally confined to CPU fixtures."""
    from contextlib import ExitStack
    stack=ExitStack()
    stack.enter_context(patch.dict(os.environ,dict(PV1_RUN_MANIFEST=str(c['worker']),
        CUDA_VISIBLE_DEVICES='',USER='e1192548'),clear=True))
    stack.enter_context(patch.object(scope,'__file__',str(c['code']/'scripts/pi0294/pv1/neural_scope.py')))
    stack.enter_context(patch.object(e,'__file__',str(c['helper'])))
    stack.enter_context(patch.object(scope,'sha',side_effect=c['digest']))
    # The public summary omits the private full static receipts. This fixture
    # uses its own exact hash, not a claimed native population qualification.
    stack.enter_context(patch.object(e,'POPULATION',c['population_fixture_sha256']))
    stack.enter_context(patch.object(scope.time,'time',return_value=c['now']))
    stack.enter_context(patch.object(scope.os,'uname',return_value=type('Host',(),dict(nodename='stdct-login-01'))()))
    return stack


def test_standard_entry_routes_actual_cold_profile_through_all_existing_guards(cold):
    c=cold
    with guards(c):
        assert scope.validate(c['root'])[0]['scope']==e.ROLE
        assert validate_entry(c['root'],profile=e.PROFILE)==c['m']
        with pytest.raises(PermissionError,match='EXACT_PHYSICAL'):
            runtime.validate_runtime(c['root'],'hopper-neural')
        for extra in (dict(now=0),dict(hostname='fake'),dict(verify_source=False)):
            with pytest.raises(PermissionError):validate_entry(c['root'],profile=e.PROFILE,**extra)
    assert not (c['root']/'pbs').exists()


@pytest.mark.parametrize('change',['old_role','training_version','wrong_host','wrong_user','old_window',
    'new_lease','unregistered','data_changed','source_changed','helper_outside','unsealed_plan','altered_population',
    'writable_prior','changed_plan','missing_prior','stop'])
def test_cold_vanda_scope_cannot_skip_original_lease_registry_source_data_or_plan(cold,change):
    c=cold;m=c['m'];root=c['root']
    if change=='old_role':m['scope']=scope.ROLE;c['manifest']()
    if change=='training_version':m['platform_training_version']='old-train';c['manifest']()
    if change=='wrong_host':m['execution']['control_hostnames']=['hopper-login-01'];c['manifest']()
    if change=='wrong_user':m['owner_user']='other';c['manifest']()
    if change=='old_window':m['limits']['PV1_T0_epoch']+=1;c['manifest']()
    if change=='new_lease':
        lease=json.loads(Path(c['b']['lease']).read_text());lease['status']='CLOSED';put(Path(c['b']['lease']),lease)
    if change=='unregistered':put(Path(c['b']['registry']),dict(entries=[]))
    if change=='data_changed':put(root/'plans/PV1_TRAIN_INPUTS.private.json',{})
    if change=='source_changed':(c['code']/'fixture-frozen-0.py').write_text('CHANGED')
    if change=='unsealed_plan':m['execution']['sealed_inputs'].pop(plan.PLAN);c['manifest']()
    if change=='altered_population':
        pop=json.loads((root/plan.POPULATION_FILE).read_text());pop['groups'].reverse();put(root/plan.POPULATION_FILE,pop)
        m['execution']['sealed_inputs'][plan.POPULATION_FILE]=sha(root/plan.POPULATION_FILE);c['manifest']()
    if change=='writable_prior':(root/'plans/PRIOR_EVALUATION_OUTCOMES.json').chmod(0o600)
    if change=='missing_prior':(root/'plans/PRIOR_EVALUATION_OUTCOMES.json').unlink()
    if change=='changed_plan':
        p=json.loads((root/plan.PLAN).read_text());p['regular_conditions']=1;put(root/plan.PLAN,p)
        m['execution']['sealed_inputs'][plan.PLAN]=sha(root/plan.PLAN);c['manifest']()
    if change=='stop':put(root/'SOURCE_SAFETY_STOP_FROM_CORE.json',{})
    with guards(c):
        if change=='helper_outside':
            with patch.object(e,'__file__',str(c['helper'].parent.parent/'foreign.py')),pytest.raises(PermissionError):scope.validate(root)
        else:
            with pytest.raises((PermissionError,ValueError,KeyError,FileNotFoundError)):scope.validate(root)


def test_metadata_closeout_does_not_convert_a_safety_stop_to_new_execution(cold):
    c=cold;put(c['root']/'SOURCE_SAFETY_STOP_FROM_CORE.json',{})
    with guards(c):
        assert scope.validate(c['root'],closeout=True)[0]['scope']==e.ROLE
        with pytest.raises(PermissionError):scope.validate(c['root'])
        with pytest.raises(PermissionError):scope.validate(c['root'],closeout='yes')


@pytest.mark.parametrize('candidate',['A_CONTROL','B_COVERAGE'])
def test_saved_endpoint_metadata_is_bound_but_not_a_B_reuse_or_native_release(cold,candidate):
    c=cold;m=c['m'];root=c['root'];checkpoint=root/'fixture-checkpoint'
    put(checkpoint/'CHECKPOINT_COMPLETE.json',dict(global_step=4096,full_resume=True))
    put(root/'plans/EVALUATION_CHECKPOINT.private.json',dict(candidate=candidate,global_step=4096,
        checkpoint=str(checkpoint),checkpoint_manifest_sha256=e.ENDPOINTS[candidate]))
    put(root/'plans/INITIAL_CORR_PARAMETER_INVENTORY.json',dict(fixture_only=True))
    groups=json.loads((root/plan.POPULATION_FILE).read_text())['groups']
    put(root/plan.PLAN,plan.development(groups,route=c['stage']['route'],candidate=candidate,
        new_bank_manifest_sha256=c['stage']['new_bank_manifest_sha256']))
    for name in (plan.PLAN,'plans/EVALUATION_CHECKPOINT.private.json','plans/INITIAL_CORR_PARAMETER_INVENTORY.json'):
        m['execution']['sealed_inputs'][name]=sha(root/name)
    c['manifest']()
    from scripts.pi0294 import native_dataset
    original=native_dataset.sha
    # Only the synthetic complete-file digest is substituted. No native
    # endpoint, file audit, resource proof or reuse permission is produced.
    def synthetic_complete_digest(path):
        return e.ENDPOINTS[candidate] if Path(path)==checkpoint/'CHECKPOINT_COMPLETE.json' else original(path)
    with patch.object(native_dataset,'sha',side_effect=synthetic_complete_digest),\
         patch.object(e,'POPULATION',c['population_fixture_sha256']):
        result=e.cold_plan(root,m,c['stage'])
        assert result['candidate']==candidate and result['resource_admission'] is False
        assert result['model_loaded'] is False and result['physics_authorized'] is False
    with patch.object(e,'POPULATION',c['population_fixture_sha256']),pytest.raises(PermissionError,match='SAVED4096'):
        e.cold_plan(root,m,c['stage'])
