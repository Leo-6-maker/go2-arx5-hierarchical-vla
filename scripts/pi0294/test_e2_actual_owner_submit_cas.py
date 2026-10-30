"""Actual E2 owner entry and real onceIO; queue/auth are explicit CPU fixtures."""
import base64,gzip,json,os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts.pi0294.pv1 import mix_accounting as a
from scripts.pi0294.generalization import minimal_native_owner as owner
from scripts.pi0294.generalization import minimal_e2only_execution as execution

@pytest.fixture(scope='module')
def captured():
 name=os.environ.get('D25_NODE_LOCAL05_PRIVATE_CAPTURE_DIR')
 if not name:pytest.skip('actual E2 request and latest07 ledger captures required')
 root=Path(name);D=root/'MAINTENANCE30_CONTROL_IO'
 p=json.loads((D/'NODE_LOCAL07FIX_RUNTIME_ASSEMBLY_FINAL_PREPARED.private.json').read_bytes());g=json.loads(gzip.decompress(base64.b64decode(p['proposed_GLOBAL_gzip_base64'])))
 previous=json.loads((root/'E2_PRE_NATIVE30_OWNER_STATUS_1793299991244465669.private.json').read_bytes())
 request=previous['records']['control/resident-owner/SOURCE_REQUEST.private.json']['value']['request']
 assert request['request_id']=='5a2c08f1-4d4d-41b0-bad0-e8f99257a229'
 assert g[execution.PREPAID][request['request_id']]['source_sha']==request['source_sha']
 return g,request


def test_actual_submit_constructs_r_and_passes_human_unlimited_count_before_registration(captured,tmp_path,monkeypatch):
 from scripts.pi0294.pv1 import evaluation_account_jobs
 from scripts.pi0294.generalization import minimal_native_launch
 g,request=captured;template=deepcopy(request)
 for key in('run_manifest_sha256','cold_input_audit_sha256','effective_queue_proof_sha256','created_epoch','submit_count'):template.pop(key,None)
 root=tmp_path/'role';(root/'control').mkdir(parents=True);manifest=dict(code=dict(vla_commit=template['source_sha']));(root/'RUN_MANIFEST.json').write_bytes(a.raw(manifest))
 audit=dict(status='PASS_ACTUAL_MINIMAL_FIRST512_COLD_INPUTS_NO_MODEL',source_sha=template['source_sha'],run_manifest_sha256=a.sha(a.raw(manifest)))
 (root/'control/COLD_INPUT_AUDIT.private.json').write_bytes(a.raw(audit));gp=tmp_path/'HOPPER_RESOURCE_LEDGER.json';gp.write_bytes(a.raw(g))
 fixture_queue=dict(explicit_CPU_fixture_only=True)
 monkeypatch.setattr(minimal_native_launch,'collect_effective_queue',lambda **_:fixture_queue)
 monkeypatch.setattr(minimal_native_launch,'effective_queue',lambda *args,**kwargs:fixture_queue)
 monkeypatch.setattr(owner.queue,'queue_rows',lambda *args,**kwargs:[])
 monkeypatch.setattr(evaluation_account_jobs,'checked_outside',lambda *args,**kwargs:[])
 calls=[]
 class StopBeforeRegistration(Exception):pass
 def registration(global_bytes,request_bytes,capture):
  calls.append(json.loads(request_bytes));assert global_bytes==gp.read_bytes();raise StopBeforeRegistration
 def read_only_call(phase,effect,invoke):
  assert phase=='FRESH_EFFECTIVE_QUEUE'and effect=='READ_ONLY';return invoke({},1)
 v=owner.Owner.__new__(owner.Owner);v.root=root;v.manifest=manifest;v.global_path=gp;v.CONFIG=dict(outside_confirmation_base64=base64.b64encode(b'{}').decode());v.request_template=template;v.adapter=execution
 v.auth=lambda secret:dict(explicit_CPU_fixture_only=True);v.program=lambda phase:dict(report_age_s=0,available_bytes_lower=10**15)
 v.fresh=lambda:dict(view=dict(campaign_jobs=[],complete_account_queue={},outside_registered_campaign_jobs=[]));v.setup=SimpleNamespace(call=read_only_call);v.record=lambda *args:None;v.registration=registration
 (root/'plans').mkdir();(root/'plans/MINIMAL_ACTUAL_CORE_MEASUREMENT.private.json').write_bytes(b'{}')
 with pytest.raises(StopBeforeRegistration):v.submit('CPU_FIXTURE_NOT_A_CREDENTIAL')
 assert len(calls)==1 and calls[0]['submit_count']==len(g['PBS_submissions'])+1
 assert calls[0]['session_plan']==request['session_plan']and calls[0]['optimizer_update_limit']==512
 assert json.loads(gp.read_bytes())==g


def test_actual_prepared_global_writer_uses_same_parent_original_once_and_rejects_replay(captured,tmp_path):
 g,_=captured;gp=tmp_path/'HOPPER_RESOURCE_LEDGER.json';before=a.raw(g);gp.write_bytes(before);target=deepcopy(g);target['revision']+=1
 v=owner.Owner.__new__(owner.Owner);v.global_path=gp;v.adapter=execution
 apply=v.prepared_global_write('REGISTER_REQUEST',before,target);result=apply({},40)
 assert gp.read_bytes()==a.raw(target)
 assert len(list(tmp_path.glob('V5_CD_*_INTENT.json')))==len(list(tmp_path.glob('V5_CD_*_RECEIPT.json')))==1
 with pytest.raises((PermissionError,FileExistsError)):apply({},40)
 assert gp.read_bytes()==a.raw(target)


def test_submission_count_preserves_every_finite_limit_and_requires_unlimited_authority(captured,monkeypatch):
 g,_=captured;owner.check_submission_count(g,1000)
 no=deepcopy(g);no.pop('V6_human_GPU_PBS_count_unlimited')
 monkeypatch.setattr(owner.e.authority,'effective_caps',lambda *args:{'PBS_submissions':44})
 owner.check_submission_count(no,44)
 with pytest.raises(PermissionError):owner.check_submission_count(no,45)
 monkeypatch.setattr(owner.e.authority,'effective_caps',lambda *args:{'PBS_submissions':None})
 with pytest.raises(PermissionError):owner.check_submission_count(no,45)
