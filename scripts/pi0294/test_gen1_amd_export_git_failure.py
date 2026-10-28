"""Actual captured failed PBS and source bytes; no native settlement."""
import base64,json
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.generalization import amd_export_git_failure as f,amd_export_accounting as e,amd_pilot_scope as scope

B=Path(__file__).resolve().parents[4]/'work/private/v5-execution-20261025'
@pytest.fixture
def captured():
 p=B/'GEN1_VANDA_OFFICIAL_EXPORT_CPU1497552_FAILED_FULL_INPUTS_READONLY.private.json'
 if not p.exists():pytest.skip('actual private failed PBS capture unavailable')
 return json.loads(p.read_bytes())

def replace_file(c,name,v):
 data=(json.dumps(v,indent=2)+'\n').encode();c['files'][name]=dict(base64=base64.b64encode(data).decode(),bytes=len(data),sha256=f.sha(data))

def test_exact_actual_failure_known_zero_derivative_does_not_clear_original_UNKNOWN(captured):
 original=deepcopy(captured);result=f.audit(captured)
 assert result['native']['Exit_status']==1 and result['native']['conservative_CPU_s']==12
 assert result['audit']['unknown_effects']==[]and not result['audit']['export_finished']
 assert result['audit']['original_failed_counter_and_UNKNOWN_unchanged']and original==captured
 g=json.loads(base64.b64decode(captured['GLOBAL_base64']));out=e.settle(g,request_id=f.REQUEST,**result)
 row=e.rows(out)[0];assert row['status']==e.STATES[-1]and not row['export_finished']
 assert out['CPU_s']==g['CPU_s']+12 and out['used']==g['used']and out['reserved']==g['reserved']
 assert out['storage_reserved_bytes']==g['storage_reserved_bytes']-10*1024**3
 assert out['new_storage_current_bytes']==g['new_storage_current_bytes']+10*1024**3
 assert out[scope.a.PACKAGE]==g[scope.a.PACKAGE]
 with pytest.raises(PermissionError):e.settle(out,request_id=f.REQUEST,**result)

@pytest.mark.parametrize('bad',['not_F','other_job','GPU','partial','output','active_owner','model','different_error','source','counter_unknown'])
def test_other_failure_or_uncertain_effect_never_zero_settled(captured,bad):
 if bad=='not_F':captured['native'][f.JOB]['job_state']='R'
 elif bad=='other_job':captured['native'][f.JOB]['Variable_List']['GEN1_REQUEST_ID']='other'
 elif bad=='GPU':captured['native'][f.JOB]['Resource_List']['ngpus']=1
 elif bad=='partial':
  name='evidence/GEN1_EXPORT_FAILURE.private.json';v=json.loads(base64.b64decode(captured['files'][name]['base64']));v['completed']=1;replace_file(captured,name,v)
 elif bad=='output':captured['datasets_members']=['datasets/partial-output']
 elif bad=='active_owner':captured['actors'][0]['actual_procstart']=captured['actors'][0]['expected_procstart']
 elif bad in ['model','different_error','counter_unknown']:
  name='RESOURCE_LEDGER.json';v=json.loads(base64.b64decode(captured['files'][name]['base64']))
  if bad=='model':v['model_loads']=1
  elif bad=='different_error':v['error']='UnrelatedFailure'
  else:v['unknown_effects'].append(dict(error='other unknown'))
  replace_file(captured,name,v)
 else:captured['modules']['scripts/pi0292/train_adapter.py']['base64']=base64.b64encode(b'changed source').decode()
 with pytest.raises(PermissionError):f.audit(captured)
