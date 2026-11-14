"""Corrected labels have separate identity but retain the exact raw consumer."""
import copy,json
from pathlib import Path
import pytest
from go2arx5_vla.world_model import sla_dense_native_v5 as dense,sla_reference_native_v6 as ref
from go2arx5_vla.world_model import task_learning_grasp_v1 as task
from test_sla_named_native_v5 import named_admission
from test_sla_dense_native_v5 import save,panel


def reference_admission(tmp_path,monkeypatch):
    q,tr,m=named_admission(tmp_path,monkeypatch)
    audit=save(Path(q['target_audit']['path']),dict(schema=ref.AUDIT_SCHEMA,passes=True,
        read_windows=1220,episodes=20,full_H40_envelope_pass=1220,FP32_decoded_envelope_pass=1220,
        original_manifest_sha256=ref.ORIGINAL_MANIFEST_SHA,label_semantics=ref.VERSION,
        original_observations_and_states_unchanged=True,all40_target_knots_actually_consumed=True,
        model_loads=0,optimizer_updates=0,physics_integrations=0,DEV_TEST_reads=0,RGB_requalified=False))
    b=m['bindings'];b.update(recipe=ref.RECIPE,dataset_id=ref.DATASET_ID,target_audit=audit['sha256'],
        reference_conversion_audit=audit['sha256'],reference_label_semantics=ref.VERSION,
        original_observation_manifest_sha256=ref.ORIGINAL_MANIFEST_SHA,
        current_anchor_equivalent_consumed_reference_labels=True)
    state=save(Path(q['complete_state']['path']),dict(schema='SLA_COMPLETE_PEFT_STATE_V1',
        step=3000,cursor=96000,arrays=[{}]*87,bindings=b))
    m.update(original_complete_sha256=state['sha256']);q['package']['sha256']=save(Path(q['package']['path'])/'MANIFEST.private.json',m)['sha256']
    tr.update(schema='SLA_N20_REFERENCE_LABEL_FINITE_FM3K_ACTUAL_V6',operation=ref.OPERATION,
        reference_label_semantics=ref.VERSION,same_original20_observations=True,original_3K_control_retained=True,
        last_checkpoint={'sha256':state['sha256']})
    trpin=save(Path(q['training_result']['path']),tr)
    common=save(Path(trpin['path']).parent/'COMMON_INITIAL_ACTUAL.private.json',dict(
        passes=True,bit_exact=True,arrays_compared=87,optimizer_updates=0))
    owner=json.loads(Path(q['candidate_owner']['path']).read_bytes());owner.update(
        schema='SLA_CORE_FRESH_REFERENCE_N20_FM3K_RGB_ACTUAL_V6',common_initial_actual=common,
        dataset_manifest_sha256='d'*64)
    ownerpin=save(Path(q['candidate_owner']['path']),owner)
    rs=json.loads(Path(q['ordinary_RGB_spec']['path']).read_bytes());rs['package']=q['package']
    q.update(schema=ref.ADMISSION,method=ref.METHOD,training_result=trpin,complete_state=state,
        target_audit=audit,candidate_owner=ownerpin,ordinary_RGB_owner=ownerpin,
        ordinary_RGB_spec=save(Path(q['ordinary_RGB_spec']['path']),rs))
    return q


def test_complete_reference_endpoint_uses_original_provider_and87_arrays(tmp_path,monkeypatch):
    q=reference_admission(tmp_path,monkeypatch)
    assert dense.qualification(q,verify_package=False)==q
    from go2arx5_vla.world_model.sla_live_v1 import apply_inference_spec
    spec={};apply_inference_spec(spec,q)
    assert spec['neural_owner_lock']==q['neural_owner_lock']
    assert spec['GPU_uuid']==q['GPU_uuid']


@pytest.mark.parametrize('failure',['old_identity','old_recipe','old_training','missing_FP32',
    'unconsumed','changed_observations','extra_model','not_same_initial','old_owner_manifest'])
def test_new_endpoint_cannot_alias_old_labels_or_partial_proofs(tmp_path,monkeypatch,failure):
    q=reference_admission(tmp_path,monkeypatch)
    if failure=='old_identity':q.update(schema=dense.NAMED_ADMISSION,method=dense.NAMED_METHOD)
    elif failure=='old_recipe':
        p=Path(q['package']['path'])/'MANIFEST.private.json';v=json.loads(p.read_bytes());v['bindings']['recipe']=dense.NAMED_RECIPE;q['package']['sha256']=save(p,v)['sha256']
    elif failure=='old_training':
        p=Path(q['training_result']['path']);v=json.loads(p.read_bytes());v.update(schema='SLA_FRESH_N20_FINITE_FM3K_ACTUAL_V5',operation='NAMED_PICK_HOLD_N20_FM_3K');q['training_result']=save(p,v)
    elif failure in ('not_same_initial','old_owner_manifest'):
        p=Path(q['candidate_owner']['path']);v=json.loads(p.read_bytes())
        if failure=='not_same_initial':
            cp=Path(v['common_initial_actual']['path']);cv=json.loads(cp.read_bytes());cv['bit_exact']=False;v['common_initial_actual']=save(cp,cv)
        else:v['dataset_manifest_sha256']=ref.ORIGINAL_MANIFEST_SHA
        pin=save(p,v);q['candidate_owner']=q['ordinary_RGB_owner']=pin
    else:
        p=Path(q['target_audit']['path']);v=json.loads(p.read_bytes())
        key,value={'missing_FP32':('FP32_decoded_envelope_pass',1219),'unconsumed':('all40_target_knots_actually_consumed',False),'changed_observations':('original_observations_and_states_unchanged',False),'extra_model':('model_loads',1)}[failure];v[key]=value
        # Exercise semantic checks after keeping the byte joins consistent.
        pin=save(p,v);q['target_audit']=pin
        statepath=Path(q['complete_state']['path']);state=json.loads(statepath.read_bytes());state['bindings'].update(target_audit=pin['sha256'],reference_conversion_audit=pin['sha256']);q['complete_state']=save(statepath,state)
        mp=Path(q['package']['path'])/'MANIFEST.private.json';m=json.loads(mp.read_bytes());m.update(bindings=state['bindings'],original_complete_sha256=q['complete_state']['sha256']);q['package']['sha256']=save(mp,m)['sha256']
        tp=Path(q['training_result']['path']);tr=json.loads(tp.read_bytes());tr['last_checkpoint']['sha256']=q['complete_state']['sha256'];q['training_result']=save(tp,tr)
    with pytest.raises(ValueError):dense.qualification(q,verify_package=False)


def test_new_train_panel_keeps_all_original_eight_conditions_and_no_old_suffix(tmp_path):
    p=panel();d,pin=dense.declaration(corrected=True);old,_=dense.declaration(named=True)
    assert d['conditions']==old['conditions']and d['dataset_id']==ref.DATASET_ID
    p.update(schema=ref.SCHEMA,method=ref.METHOD,policies={ref.METHOD:dict(arm=ref.METHOD)},
        preregistration=dict(path=str(Path(dense.__file__).resolve().parents[3]/ref.DECLARATION),sha256=pin))
    for row in p['assignments']:row.update(method=ref.METHOD,purpose=ref.EXPERIMENT)
    assert len(dense.validate_panel(p))==8
    inp=dict(schema=ref.INPUT_SCHEMA,source_sha=p['source_sha'],assignment_slice=[0,8])
    assert dense.selection(inp,p)==p['assignments']
    with pytest.raises(ValueError):dense.selection(dict(inp,assignment_slice=[3,8]),p)
    from go2arx5_vla.world_model.task_learning_student_batch_v1 import selection
    assert selection(inp,p)==p['assignments']
    from go2arx5_vla.world_model.task_learning_student_registration_v2 import group_input
    p['registration']={k:'fixture'for k in ('source_root','source_files','native_base','gpu_base','scenes','scene_allowlist','scene_allowlist_sha256','memory_execution','memory_execution_reference')}
    p['registration']['native_pause_read_wait']=dict(path=str(Path(dense.__file__).resolve().parents[3]/'experiments/sla_go2_posttrain/native_pause_read_wait_v1.json'),sha256='413d96e953a11d790136cd64e3a6f28a2fb13d89fe73ea4b6f55b896c50e6eb8')
    import time
    binding=save(tmp_path/'binding.json',dict(deadline_epoch=time.time()+10000))
    save(tmp_path/'LEASE.json',dict(status='ACTIVE'))
    q=group_input(dict(assignment_slice=[0,8],panel=dict(path='/fixture',sha256='a'*64),pilot1_binding=binding,fusion_binding=binding,max_wall_s=3600),p)
    assert q['schema']==ref.INPUT_SCHEMA
    from go2arx5_vla.world_model.sla_native_continuation_v2 import wait_seconds
    assert wait_seconds(dict(q,task_learning_role=task.SLA_DENSE_ROLE))==90


def test_native_metadata_entry_imports_without_training_scipy_or_jax():
    import os,subprocess,sys
    code="""
import sys,importlib.abc
class NativeOnly(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]in('scipy','jax','torch','flax'):
   raise ImportError('NATIVE_METADATA_MUST_NOT_IMPORT_'+fullname)
sys.meta_path.insert(0,NativeOnly())
from scripts.wm02.sla_n20_endpoint_followup_v5 import run
from scripts.wm02.admit_dense_sla_native_v5 import build
from go2arx5_vla.world_model.sla_reference_native_v6 import METHOD
assert METHOD=='N20_REFERENCE_FM3K_RAW_P10'
"""
    subprocess.run([sys.executable,'-c',code],env=dict(os.environ),check=True,capture_output=True,text=True)


@pytest.mark.parametrize('kind',['dense','named','reference'])
def test_registration_builds_all_eight_scene_rows_before_mutation(tmp_path,monkeypatch,kind):
    """Exercise the real registration path, including its module/scene names."""
    from go2arx5_vla.world_model import task_learning_student_registration_v2 as registration
    from go2arx5_vla.world_model import sla_native_v1
    from scripts.pi0294.e1 import journal
    source=tmp_path/'source';source.mkdir()
    method={'dense':dense.METHODS[0],'named':dense.NAMED_METHOD,'reference':ref.METHOD}[kind]
    declaration,pin=dense.declaration(named=kind=='named',corrected=kind=='reference')
    declaration_path=Path(dense.__file__).resolve().parents[3]/{'dense':dense.DECLARATION,
        'named':dense.NAMED_DECLARATION,'reference':ref.DECLARATION}[kind]
    seal=save(source/'SOURCE_STAGED.private.json',dict(source_sha='c'*40,files={}))
    admission=save(tmp_path/'admission.json',dict(method=method,package={}))
    scenes={}
    for row in declaration['conditions']:
        if row['layout'] in scenes:continue
        path=tmp_path/row['layout'];path.mkdir()
        (path/'SCENE.private.mjb').write_bytes(b'scene fixture')
        scenes[row['layout']]=save(path/'proof.json',dict(parent='TASK_LEARNING_UNDER_HAND_'+row['layout']))
    monkeypatch.setattr(registration,'closed_base',lambda _:dict(campaign_root=str(tmp_path/'campaign'),
        registered_root=str(tmp_path/'a/b/c/base')))
    monkeypatch.setattr(registration,'actual_site',lambda _:None)
    monkeypatch.setattr(dense,'qualification',lambda q:q)
    monkeypatch.setattr(sla_native_v1,'scene_custody',lambda *_:None)
    request=dict(root=str(tmp_path/'a/new-panel'),source_root=str(source),source_sha='c'*40,
        source_files={},source_seal=seal,Native_body_qualification=admission,
        native_base={},gpu_base={},preregistration=dict(path=str(declaration_path),sha256=pin),
        scenes=scenes,scene_producers={k:{}for k in scenes},scene_allowlist='',scene_allowlist_sha256='',
        memory_execution='',memory_execution_reference='')
    if kind=='reference':
        wait=Path(dense.__file__).resolve().parents[3]/'experiments/sla_go2_posttrain/native_pause_read_wait_v1.json'
        request['native_pause_read_wait']=dict(path=str(wait),sha256=dense.digest(wait))
    seen=[];validate=dense.validate_panel
    def validated(panel):
        rows=validate(panel);seen.extend(rows);return rows
    monkeypatch.setattr(dense,'validate_panel',validated)
    class BeforeMutation(Exception):pass
    def stop_before_mutation(*_):raise BeforeMutation
    monkeypatch.setattr(journal,'Journal',stop_before_mutation)
    with pytest.raises(BeforeMutation):dense.register_panel(request)
    assert len(seen)==8 and [r['layout']for r in seen]==[r['layout']for r in declaration['conditions']]
    assert all(r['method']==method for r in seen)
    assert not Path(request['root']).exists()
