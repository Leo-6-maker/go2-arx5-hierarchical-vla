"""Captured full ledger and complete queue fixtures, never native admission."""
import base64,json
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.generalization import amd_export_queue as q,amd_export_accounting as e
from scripts.pi0294.dec1 import vanda_campaign as original

B=Path(__file__).resolve().parents[4]/'work/private/v5-execution-20261025'
@pytest.fixture
def ledger():
 p=B/'GEN1_VANDA_OFFICIAL_EXPORT_ROLE_INDEPENDENT_READONLY.private.json'
 if not p.exists():pytest.skip('private captured registered CPU ledger unavailable')
 return json.loads(base64.b64decode(json.loads(p.read_bytes())['GLOBAL_base64']))

def native(item,**change):
 row=dict(platform=item['native_platform'],job_id=item['job_id'],state='Q',
  project=item['project'],name=item['job_name'],ngpus=0);row.update(change);return row

def test_exact_unbound_intent_preserved_and_no_actual_job_invented(ledger):
 item=e.rows(ledger)[0];before=deepcopy(ledger)
 assert original.classify(ledger,[])[2][0]['job_id']is None
 v=q.view(ledger,[],request_id=item['request_id'])
 assert v['own_CPU_unbound']and v['native_job_id']is None and not v['own']and not v['unknown']
 assert not v['native_CPU_release']and not v['model_permission']and ledger==before

def test_bound_zero_GPU_CPU_Q_R_and_unknown_other_intents(ledger):
 item=e.rows(ledger)[0];g=e.bind(ledger,request_id=item['request_id'],job_id='123.stdct-mgmt-02');item=e.rows(g)[0]
 for state in ['Q','R']:
  rows=[native(item,state=state),dict(platform=item['native_platform'],job_id='124.stdct-mgmt-02',state='R',project='independent',name='human-other',ngpus=2)]
  v=q.view(g,rows,request_id=item['request_id']);assert len(v['own'])==len(v['outside'])==1 and not v['unknown']and v['complete_selected_rows_retained']==rows
 g['PBS_submissions'].append(dict(request_id='other-unknown',job_id=None,status=e.STATES[0]))
 assert q.view(g,[native(item)],request_id=item['request_id'])['unknown']==[dict(job_id=None,reason='ACTIVE_LEDGER_WITHOUT_CURRENT_NATIVE_JOB_PROOF')]

@pytest.mark.parametrize('bad',['GPU','project','name','duplicate','other_campaign','absent','wrong_request'])
def test_wrong_native_identity_or_other_unknown_never_releases(ledger,bad):
 request=e.rows(ledger)[0]['request_id'];g=e.bind(ledger,request_id=request,job_id='123.stdct-mgmt-02');item=e.rows(g)[0];rows=[native(item)]
 if bad=='GPU':rows[0]['ngpus']=1
 elif bad=='project':rows[0]['project']='wrong'
 elif bad=='name':rows[0]['name']='wrong'
 elif bad=='duplicate':rows+=rows
 elif bad=='other_campaign':rows.append(native(item,job_id='124.stdct-mgmt-02',name='pi0294_unknown'))
 elif bad=='absent':rows=[]
 else:request='other-request'
 if bad in ['other_campaign','absent']:assert q.view(g,rows,request_id=request)['unknown']
 else:
  with pytest.raises(PermissionError):q.view(g,rows,request_id=request)
