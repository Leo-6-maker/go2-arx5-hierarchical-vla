"""Captured real1503589 failure, exact old source and temp original arithmetic."""
import json,os,gzip,base64
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.generalization import minimal_entry_profile_failure as f,minimal_execution as e,minimal_retry_execution as retry
from scripts.pi0294.dec1.a40_training_context import minimal_training_adapter
from scripts.pi0294.generalization import minimal_followup_scope as s

@pytest.fixture
def actual():
 p=Path(os.environ.get('ARX5_ENTRY_FAILURE_CAPTURE',''))
 if not p.is_file():pytest.skip('full actual1503589 capture required')
 return json.loads(p.read_bytes())

def test_real_causal_source_traceback_terminal_and_exact_costs(actual):
 out=f.audit(actual)
 assert out['native']['conservative_GPU_s']==133 and out['native']['conservative_CPU_s']==240
 assert out['effects']['optimizer_updates_committed']==out['effects']['model_load_attempts']==out['effects']['query_attempts']==0
 assert out['original_missing_final_witness']['original_process_final_created']is False
 g=json.loads(gzip.decompress(base64.b64decode(actual['GLOBAL_gzip_base64'])));t=deepcopy(actual['physical']);now=actual['observed_epoch']+1
 after=retry.settle(g,f.REQUEST,native=out['native'],effects=out['effects'],audit=out['audit'],physical_proof=t,physical_observed_epoch=actual['physical_observed_epoch'],now_epoch=now)
 from scripts.pi0294.pv1.neural_ledger import check
 check(after);assert after['PBS_submissions'][:-1]==g['PBS_submissions'][:-1]and after[retry.FIELD]==g[retry.FIELD]
 assert after['new_storage_current_bytes']==g['new_storage_current_bytes']and after['new_storage_peak_bytes']==g['new_storage_peak_bytes']
 assert after['used']['H200_s']==g['used']['H200_s']+133 and after['CPU_s']==g['CPU_s']+240
 assert retry._selected(after)[1]['scientific_endpoint_complete']is False
 with pytest.raises(PermissionError):retry.settle(after,f.REQUEST,native=out['native'],effects=out['effects'],audit=out['audit'],physical_proof=t,physical_observed_epoch=actual['physical_observed_epoch'],now_epoch=now)

@pytest.mark.parametrize('kind',['source','traceback','native','actors','missing','science','inventory','wrapper'])
def test_no_absence_only_or_relabelled_other_failure(actual,kind):
 v=deepcopy(actual)
 if kind=='source':v['modules']['scripts/pi0294/dec1/a40_training_context.py']['sha256']='0'*64
 elif kind=='native':v['native']['Exit_status']=0
 elif kind=='actors':v['actors'][0]['actual_procstart']=v['actors'][0]['expected_procstart']
 elif kind=='missing':v['absent_PBS'].remove('PBS_PROCESS_FINAL.json')
 elif kind=='science':v['science_inventory']['training']=['SAVED_MODEL']
 elif kind=='inventory':v['inventory']['files'].append({'unexpected':True})
 else:
  name='PBS/pbs.log'if kind=='traceback'else'PBS/PBS_WRAPPER_FINAL.json'
  body=b'no causal traceback'if kind=='traceback'else e.raw({'returncode':0})
  v['files'][name]=dict(base64=base64.b64encode(body).decode(),bytes=len(body),sha256=e.sha(body))
 with pytest.raises((PermissionError,KeyError,AssertionError)):f.audit(v)

def test_actual_original_entry_dispatch_accepts_declared_new_profile_only():
 assert minimal_training_adapter(retry.VERSION,s.PROFILE)is retry
 assert minimal_training_adapter(retry.VERSION,retry.PROFILE)is retry
 assert minimal_training_adapter(e.VERSION,e.PROFILE)is e
 for version,profile in[(e.VERSION,s.PROFILE),('UNKNOWN',s.PROFILE),(retry.VERSION,'another')]:
  with pytest.raises(PermissionError):minimal_training_adapter(version,profile)
