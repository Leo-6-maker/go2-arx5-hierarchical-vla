"""Explicit GPU0 sharing only; malformed reads and GPU1 memory still fail."""
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.pv1 import evaluation_core_shared_B24 as s,evaluation_core_B24_1000 as late,evaluation_core_identity as n
from scripts.pi0294.test_pv1_core_evaluation_identity import DEVICE
ROOT=Path(__file__).resolve().parents[2]
def request():return dict(request_id=s.RID,run_uuid=s.RUN,control_id=s.CONTROL,submission_nonce=s.NONCE,evaluation_candidate='B_COVERAGE',stage='P4_DEV',noise_bank=294101,GPU_uuid=s.GPU,shared_GPU_grant_sha256=s.GRANT_SHA,late_start_grant_sha256=late.GRANT_SHA)
def test_explicit_shared_human_request_accepts_complete_foreign_inventory_only_here():
 r=request();raw=(ROOT/'reports/PI0294/design/parallel-science-20261022/V4A_B24_EXACT_SHARED_GPU_HUMAN_AUTHORIZATION_20261025.json').read_bytes();s.grant(raw)
 with pytest.raises(PermissionError):s.grant(raw+b' ')
 processes=s.GPU+', 123, python, 6500\n'+s.GPU+', 456, python, 5800\n'
 with pytest.raises(PermissionError):n.parse_GPU_rows(DEVICE,processes,GPU_uuid=s.GPU)
 v=n.parse_GPU_rows(DEVICE,processes,GPU_uuid=s.GPU,shared_request=r)
 assert v['human_authorized_shared_GPU_grant_sha256']==s.GRANT_SHA and late.start_limit(r)==late.LATEST
@pytest.mark.parametrize('key,value',[('request_id',late.RID),('evaluation_candidate','A_CONTROL'),('noise_bank',294102),('control_id',s.RID),('submission_nonce',s.RID),('shared_GPU_grant_sha256','f'*64),('GPU_uuid','GPU-845348f1-1511-dc19-0877-cc97658e3edd'),('queue_id',s.RID)])
def test_no_other_request_model_noise_or_gpu_receives_sharing(key,value):
 r=request();r[key]=value
 with pytest.raises(PermissionError):s.allowed(r)
def test_GPU1_and_truncated_or_nonfinite_reads_not_qualified_by_shared_approval():
 for devices,processes,gpu in [(DEVICE,'truncated',s.GPU),(DEVICE.replace('32760','N/A'),'',s.GPU),(DEVICE,'','GPU-845348f1-1511-dc19-0877-cc97658e3edd')]:
  with pytest.raises(PermissionError):n.parse_GPU_rows(devices,processes,GPU_uuid=gpu,shared_request=request())
def test_shared_requires_both_sealed_grants_and_actual_pair_before_runtime_admission():
 r=request();r['sealed_inputs']={s.PLAN:s.GRANT_SHA,late.PLAN:late.GRANT_SHA};m=dict(execution=dict(sealed_inputs=deepcopy(r['sealed_inputs'])))
 assert s.check_request(m,r)==late.LATEST
 del m['execution']['sealed_inputs'][s.PLAN]
 with pytest.raises(PermissionError):s.check_request(m,r)

def test_actual_closed_prebind_bytes_only_grant_once_no_cost_or_history_erasure():
 import base64,json
 from scripts.pi0294.pv1 import evaluation_core_accounting as c
 folder=ROOT.parent.parent/'work/private/science-next/wdr1-target-identity/v4a-core-eval-preparation-01/B24-1000-prebind-failure-terminal-01'
 paths=sorted(folder.glob('readback-for-recovery.REPLY_*.private.json'))
 if not paths:pytest.skip('private actual independent failed-unit dual closure unavailable')
 x=json.loads(paths[-1].read_bytes())['value'];g=json.loads(base64.b64decode(x['global_base64']));l=json.loads(base64.b64decode(x['LOCAL_base64']));raw=(ROOT/'reports/PI0294/design/parallel-science-20261022/V4A_B24_EXACT_SHARED_GPU_HUMAN_AUTHORIZATION_20261025.json').read_bytes()
 a,b=s.install(g,l,raw,source_sha='f'*40,closure=x['closure'],now_epoch=late.LATEST-1200)
 assert s.validate_pair(a,b)==a[s.FIELD] and c.FIELD in a and a[c.FIELD]==g[c.FIELD]
 for k in ('used','reserved','CPU_s','CPU_reserved_s','PBS_submissions','optimizer_updates','new_storage_current_bytes'):assert a[k]==g[k]
 with pytest.raises(PermissionError):s.install(a,b,raw,source_sha='f'*40,closure=x['closure'],now_epoch=late.LATEST-1100)
 bad=deepcopy(x['closure']);bad['GPU_lease_status']='ACTIVE'
 with pytest.raises(PermissionError):s.install(g,l,raw,source_sha='f'*40,closure=bad,now_epoch=late.LATEST-1200)
 with pytest.raises(PermissionError):s.install(g,l,raw,source_sha='f'*40,closure=x['closure'],now_epoch=late.LATEST)
