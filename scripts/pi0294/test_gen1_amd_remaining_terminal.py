"""Future terminal fixtures seeded by an actual saved T2 negative witness.

Only the first negative is actual. Other rows/OS endings are explicit fixtures;
none of these tests claims that the running native batch has completed.
"""
import base64,json,os
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.generalization import amd_remaining as r,amd_adoption as a

W=Path(os.environ.get('PI0294_PRIVATE_WORKSPACE',str(Path(__file__).resolve().parents[4])))
B=W/'work/private/v5-execution-20261025'
@pytest.fixture
def inputs():
 file=B/'GEN1_AMD_REMAINING10_COMPLETED_CASE_0_0.private.json'
 if not file.exists():pytest.skip('actual private T2 partial-negative witness absent')
 real=json.loads(file.read_bytes());g=json.loads(base64.b64decode(json.loads((B/'GEN1_AMD_REMAINING10_GLOBAL_RESERVE.private.json').read_bytes())['value']['GLOBAL_base64']))
 pool=g[a.FIELD]['teacher_pilot']['engineering_recovery']['remaining_conditions'];workers={}
 for key,bound in pool['workers'].items():
  rows=[]
  for i,identifier in enumerate(bound['attempt_ids']):
   negative=key=='0'and i==0
   row=dict(attempt_id=identifier,terminal=True,unknown_effects=[],raw_and_custody_verified=True,
    source_sha=r.SOURCE,full_task_success=not negative,
    live_ticks=real['live_ticks']if negative else 16000,shadow_ticks=real['shadow_ticks']if negative else 16000,
    native_safety_failures=0,S1_rejections=real['S1_rejections']if negative else 0,
    raw_manifest_sha256=real['raw_manifest_sha256']if negative else str(i+int(key)*5).zfill(64),
    audit_sha256=real['capture_row']['original_capture_receipt']['audit_sha256']if negative else'a'*64,
    card_sha256='b'*64,worker_receipt_sha256='c'*64,paired_frames=225 if negative else 641,
    independent_raw_audit='PASS_INDEPENDENT_PARTIAL_NEGATIVE_WITNESS'if negative else'PASS_INDEPENDENT_RAW_AUDIT')
   rows.append(row)
  workers[key]=dict(manifest_sha256=bound['manifest_sha256'],run_root=bound['root'],expected_ids=bound['attempt_ids'],attempted_ids=bound['attempt_ids'],cases=rows,unknown_effects=[],source_sha=r.SOURCE,engine_starts=400,live_ticks=sum(x['live_ticks']for x in rows),shadow_ticks=sum(x['shadow_ticks']for x in rows))
 audit=dict(status='PASS_INDEPENDENT_TEN_KNOWN_NATIVE_TERMINAL_EFFECTS',source=r.SOURCE,
  all_processes_terminal=True,worker_OS_returncodes=[0,0],unknown_native_effects=[],
  old_costs_and_raw_and_valid_failures_preserved=True,epoch=real['epoch'],workers=workers,
  models=0,queries=0,updates=0,Core_CPU_s=5000.,fixture=True)
 return g,audit

def test_full_future_fixture_keeps_actual_first_negative_and_no_training_admission(inputs):
 g,audit=inputs;v=r.settlement(g,audit,now_epoch=audit['epoch']+1);row=v['proposed_global'][a.FIELD]['teacher_pilot']['engineering_recovery']['remaining_conditions']
 assert row['actual']['episode_starts']==10 and row['actual']['independently_raw_admitted']==9
 assert row['actual']['S1_rejections']==1 and not row['native_training_admitted']
 assert row['outcomes'][0]['full_task_success']is False and row['outcomes'][0]['live_ticks']==5600
 assert all(v['proposed_global'][k]==x for k,x in g.items()if k not in(a.FIELD,'revision'))
 with pytest.raises(PermissionError):r.settlement(v['proposed_global'],audit,now_epoch=audit['epoch']+2)
 assert r.settlement(g,audit,now_epoch=1793537550.)['new_GPU']==0

@pytest.mark.parametrize('bad',['OS','source','unknown','missing','replay','clock','CPU','ticks','score_only'])
def test_terminal_requires_known_actual_custody_and_exact_remaining_conditions(inputs,bad):
 g,audit=deepcopy(inputs);w=audit['workers']['0'];case=w['cases'][0]
 if bad=='OS':audit['all_processes_terminal']=False
 elif bad=='source':audit['source']='0'*40
 elif bad=='unknown':w['unknown_effects']=['unknown']
 elif bad=='missing':w['cases'].pop()
 elif bad=='replay':case['attempt_id']='old-saved-full-T1'
 elif bad=='clock':audit['epoch']=0
 elif bad=='CPU':audit['Core_CPU_s']=float('nan')
 elif bad=='ticks':w['live_ticks']+=1
 else:case['full_task_success']=True
 with pytest.raises(PermissionError):r.settlement(g,audit,now_epoch=1793054000.)
