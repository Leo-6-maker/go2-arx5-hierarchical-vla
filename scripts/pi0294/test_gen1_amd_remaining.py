"""Captured real saved audit; no native Teacher or model effects in tests."""
import base64,json,os
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.generalization import amd_capture_terminal as t,amd_remaining as r,amd_capture as c,amd_adoption as a

W=Path(os.environ.get('PI0294_PRIVATE_WORKSPACE',str(Path(__file__).resolve().parents[4])))
B=W/'work/private/v5-execution-20261025'

@pytest.fixture
def inputs():
 file=B/'GEN1_AMD_SAVED_R1_TERMINAL_INDEPENDENT_READONLY.private.json'
 if not file.exists():pytest.skip('actual private saved audit absent')
 audit=json.loads(file.read_bytes());g=json.loads(base64.b64decode(json.loads((B/'GEN1_AMD_SAVED_R1_GLOBAL_RESERVE.private.json').read_bytes())['value']['GLOBAL_base64']))
 p=json.loads((B/'GEN1_AMD_SAVED_R1_PREPARED.private.json').read_bytes());packet=base64.b64decode(p['common']['plans/GEN1_V5_PREVIEW_V2.private.json'])
 return g,audit,packet

def test_actual_reaudit_classification_once_and_late_financial_close(inputs):
 g,audit,_=inputs;v=t.reaudit_settlement(g,audit,now_epoch=audit['epoch']+1);row=v['proposed_global'][a.FIELD]['teacher_pilot']['engineering_recovery']['saved_data_reaudit']
 assert row['actual']['engine_starts']==4 and row['actual']['physics_ticks']==row['actual']['episode_starts']==0
 assert len(row['raw_admitted_cases'])==2 and not row['native_training_admitted']
 assert all(v['proposed_global'][k]==x for k,x in g.items()if k not in(a.FIELD,'revision'))
 with pytest.raises(PermissionError):t.reaudit_settlement(v['proposed_global'],audit,now_epoch=audit['epoch']+2)
 assert t.reaudit_settlement(g,audit,now_epoch=1793537550.)['new_PBS']==0

@pytest.mark.parametrize('bad',['CPU','steps','OS','raw','unknown','identity','old_failed_audit'])
def test_reaudit_bad_effect_or_custody_cannot_settle(inputs,bad):
 g,audit,_=deepcopy(inputs)
 if bad=='CPU':audit['Core_CPU_s']=float('nan')
 elif bad=='steps':audit['integrations']=1
 elif bad=='OS':audit['actual_OS_returncode']=1
 elif bad=='raw':audit['cases']['0']['raw_admitted']=False
 elif bad=='unknown':audit['unknown_native_effects']=['unknown']
 elif bad=='identity':audit['manifest_sha256']='0'*64
 else:audit['cases']['0']['original_failed_audit_sha256']='0'*64
 with pytest.raises(PermissionError):t.reaudit_settlement(g,audit,now_epoch=audit['epoch']+1)

@pytest.mark.parametrize('bad',[None,'saved_replay','active','audit','duplicate','late'])
def test_only_original_unattempted_ten_can_start_after_reaudit_closes(inputs,bad):
 g,audit,packet=inputs;g=t.reaudit_settlement(g,audit,now_epoch=audit['epoch']+1)['proposed_global'];saved=g[a.FIELD]['teacher_pilot']['engineering_recovery']['saved_data_reaudit']
 lease=dict(status='CLOSED',manifest_sha256=saved['manifest_sha256'],independent_terminal_audit_sha256=saved['independent_terminal_audit_sha256'])
 workers={str(i):dict(run_uuid='gen1-amd-teacher-r2-'+str(i).zfill(2),root='/fixture/children/gen1-amd-teacher-r2-'+str(i).zfill(2),manifest_sha256='a'*64,attempt_ids=[x['id']for x in r.cards(packet,i)],counter_policy=c.COUNTER_POLICY)for i in (0,1)}
 now=audit['epoch']+2
 if bad=='saved_replay':workers['0']['attempt_ids'][0]=c.cards(packet,0)[0]['id']
 elif bad=='active':lease['status']='ACTIVE'
 elif bad=='audit':lease['independent_terminal_audit_sha256']='0'*64
 elif bad=='duplicate':g[a.FIELD]['teacher_pilot']['engineering_recovery']['remaining_conditions']={}
 elif bad=='late':now=1793537550.
 call=lambda:r.allocation(g,packet_bytes=packet,workers=workers,closed_reaudit_lease=lease,now_epoch=now)
 if bad is None:
  v=call();assert v['row']['episode_starts']==10 and v['row']['engine_starts']==880 and v['row']['Core_CPU_s']==36000
  assert set(workers['0']['attempt_ids']).isdisjoint(workers['1']['attempt_ids'])and v['new_GPU']==v['new_PBS']==0
 else:
  with pytest.raises(PermissionError):call()
