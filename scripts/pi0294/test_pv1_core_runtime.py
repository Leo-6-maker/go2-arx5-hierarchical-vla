"""CPU routing and pre-effect fences, never a native GPU qualification."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from scripts.pi0294.pv1 import evaluation_core_runtime as runtime,evaluation_core_identity as identity
from scripts.pi0294.pv1 import neural,evaluation_plan as plan
from scripts.pi0294.pv1 import evaluation_core_native as native
from scripts.pi0294.pv1 import evaluation_admission as resources
from scripts.pi0294.test_pv1_core_evaluation_admission import captured_inputs
from scripts.pi0294.test_pv1_core_premodel_join import admission
from scripts.pi0294.test_pv1_core_evaluation_accounting import NOW
from scripts.pi0294.test_pv1_evaluation_plan import groups
from scripts.pi0294.test_mp1_transport_fixture import put


def test_existing_dispatcher_routes_only_the_explicit_Core_request(tmp_path):
    request=tmp_path/'REQUEST.json';request.write_text(json.dumps(dict(platform_evaluation_version=identity.c.VERSION)))
    with patch.object(runtime,'run',return_value={'CPU_fixture_routed':True}) as hook:
        assert neural.run(tmp_path,request)=={'CPU_fixture_routed':True}
        hook.assert_called_once_with(tmp_path,request)
    request.write_text(json.dumps(dict(stage='P4_DEV',platform_evaluation_version='unknown-Core-name')))
    with patch.object(runtime,'run') as hook,pytest.raises(PermissionError):neural.run(tmp_path,request)
    hook.assert_not_called()


def test_whole_original_first_noise_panel_only_for_existing_Core_AB(captured_inputs):
    inputs,_=captured_inputs;m,r,a=admission(inputs)
    p=plan.development(groups(),route='COVERAGE_FIRST',candidate=r['evaluation_candidate'],
        new_bank_manifest_sha256=r['new_bank_manifest_sha256'])
    labels=neural.evaluation_labels(r,p,[]);assert len(labels)==1
    spec=p['panels'][labels[0]];assert spec['noise_bank']==294101 and len(spec['cards'])==24
    for fault,value in [('noise_bank',294102),('remaining_conditions',8),('optimizer_updates',1),
            ('evaluation_candidate','B0_CORR'),('statistics_sha256','f'*64),('checkpoint_manifest_sha256','f'*64)]:
        bad=dict(r,**{fault:value})
        with pytest.raises(PermissionError):neural.evaluation_labels(bad,p,[])


@pytest.mark.parametrize('fault',['wrong_memory','wrong_device','two_GPU','index_alias','other_endpoint','training','wrong_status'])
def test_cuda_metadata_cannot_relabel_a_smaller_or_foreign_allocation(captured_inputs,fault):
    inputs,_=captured_inputs;m,r,a=admission(inputs);opts=dict(visible=r['GPU_uuid'],count=1,
        name='NVIDIA GeForce RTX 4080 SUPER',memory_bytes=32760*1024**2)
    assert identity.require_device(r,a,**opts)
    if fault=='wrong_memory':opts['memory_bytes']=16*1024**3
    if fault=='wrong_device':opts['name']='NVIDIA A40'
    if fault=='two_GPU':opts['count']=2
    if fault=='index_alias':opts['visible']='0'
    if fault=='other_endpoint':a['checkpoint_manifest_sha256']='f'*64
    if fault=='training':r['optimizer_updates']=1
    if fault=='wrong_status':a['status']=identity.c.old.STATUS
    with pytest.raises(PermissionError):identity.require_device(r,a,**opts)


def test_release_needs_one_same_owner_immutable_current_phase(captured_inputs,tmp_path):
    inputs,_=captured_inputs;m,r,a=admission(inputs);root=tmp_path/r['run_uuid'];directory=root/'pbs/alloc-01'
    p=a['native_process'];name='CORE_MODEL_RELEASE.json'
    value=dict(schema='PV1_CORE_EXISTING_NATIVE_PHASE_RELEASE_V1',run_uuid=root.name,request_id=r['request_id'],
        request_sha256=a['request_sha256'],source_sha=r['source_sha'],job_id=a['job_id'],phase=name,
        physics_authorized=False,new_optimizer_updates=0,observed_epoch=NOW)
    directory.mkdir(parents=True);(directory/'REQUEST.json').write_bytes(inputs['request_bytes'])
    guard_calls=[]
    for fault in ('good','wrong_handle','wrong_phase','old_observation','request_changed','writable'):
        v=deepcopy(value)
        if fault=='wrong_handle':v['job_id']='1480000.stdct-mgmt-02'
        if fault=='wrong_phase':v['phase']='START_NEURAL.json'
        if fault=='old_observation':v['observed_epoch']=NOW-121
        if fault=='request_changed':v['request_sha256']='f'*64
        put(directory/name,v,fault!='writable')
        with patch.object(runtime.time,'time',return_value=NOW):
            if fault=='good':assert runtime.release(root,directory,r,p,name,phase_started=NOW-1,
                guard=lambda:guard_calls.append(True))==v
            else:
                with pytest.raises(PermissionError):runtime.release(root,directory,r,p,name,phase_started=NOW-1,guard=lambda:None)
    assert guard_calls==[True]


def test_invalid_core_root_fails_before_importing_neural_or_physics_libraries(tmp_path):
    root=Path(__file__).resolve().parents[2]
    program='''import importlib.abc,sys
class DenyNative(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','mujoco','rclpy'):raise AssertionError('NATIVE_IMPORT_BEFORE_ADMISSION:'+fullname)
sys.meta_path.insert(0,DenyNative())
from scripts.pi0294.pv1.evaluation_core_runtime import run
try:run(sys.argv[1],sys.argv[1]+'/REQUEST.json')
except (PermissionError,FileNotFoundError):print('REFUSED_BEFORE_NATIVE_IMPORT')
else:raise AssertionError('invalid root admitted')
'''
    result=subprocess.run([sys.executable,'-c',program,str(tmp_path/'missing')],capture_output=True,text=True,
        timeout=15,env=dict(os.environ,PYTHONPATH=str(root)+':'+str(root/'src'),CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=='REFUSED_BEFORE_NATIVE_IMPORT'


def test_native_loader_preserves_every_original_actual_history_file(captured_inputs,tmp_path):
    import hashlib
    inputs,_=captured_inputs;files={plan.POPULATION_FILE:inputs['population_bytes']};index={}
    for candidate,items in inputs['evidence_by_candidate'].items():
        index[candidate]=[]
        for i,item in enumerate(items):
            paths={kind:'plans/history/'+candidate+'/'+str(i)+'/'+kind+'.json' for kind in item}
            files.update({paths[k]:v for k,v in item.items()});index[candidate].append(paths)
    files[resources.INDEX]=(json.dumps(dict(schema='PI0294_V4_PRIMARY_HISTORY_INPUT_INDEX_V1',
        campaign_run_uuid=identity.c.scope.CAMPAIGN,authority_release=identity.c.AUTHORITY,
        population=plan.POPULATION_FILE,evidence=index))+'\n').encode()
    for name,raw in files.items():
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw);path.chmod(0o400)
    pins={n:hashlib.sha256(v).hexdigest() for n,v in files.items()}
    m=dict(execution=dict(sealed_inputs=pins));r=dict(sealed_inputs=dict(pins))
    pop,history=native.static_history(tmp_path,m,r)
    assert pop==inputs['population_bytes'] and history==inputs['evidence_by_candidate']
    assert len(history['B0_CORR'])==17 and history['A_CONTROL']==history['B_COVERAGE']==[]
    name=index['B0_CORR'][0]['model_ready'];path=tmp_path/name;path.chmod(0o600);path.write_bytes(path.read_bytes()+b' ');path.chmod(0o400)
    with pytest.raises(PermissionError,match='SEALED_HISTORY'):native.static_history(tmp_path,m,r)


def test_Core_processor_pins_are_exact_original_deployed_entry_baseline():
    import base64,hashlib
    from scripts.pi0294.pv1.evaluation_core_scope import COSMOS_PROCESSOR_PINS
    folder=Path(__file__).resolve().parents[4]/'work/private/science-next/wdr1-target-identity/v4a-core-eval-preparation-01'
    p=folder/'ACTUAL_CORE_EXISTING_STANDARD_ENTRY_FOR_GPU_PROFILE_1792840426467566678.private.json'
    if not p.is_file():pytest.skip('private original baseline capture absent')
    raw=base64.b64decode(json.loads(p.read_bytes())['value']['members']['ENTRY_HASH_BASELINE.json']['base64'])
    assert hashlib.sha256(raw).hexdigest()=='fa08051700d9320ce003ab0749f8ab25e64b18a7333ed2fed1376881314b4423'
    rows=[r for r in json.loads(raw)['4080_files'] if r['role'].startswith('COSMOS_PROCESSOR_')]
    assert {r['role'].removeprefix('COSMOS_PROCESSOR_'):r['sha256'] for r in rows}==COSMOS_PROCESSOR_PINS
