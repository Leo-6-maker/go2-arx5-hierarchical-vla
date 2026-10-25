"""CPU check of the actual private forwarding seam that failed, no native GO."""
import ast,base64,json
from pathlib import Path
from copy import deepcopy
import pytest
from scripts.pi0294.pv1 import evaluation_core_shared_repair as s,evaluation_core_shared_B24 as parent,evaluation_core_B24_1000 as late,evaluation_core_identity as n
from scripts.pi0294.test_pv1_core_evaluation_identity import DEVICE
ROOT=Path(__file__).resolve().parents[2]
def request():return dict(request_id=s.RID,run_uuid=s.RUN,control_id=s.CONTROL,submission_nonce=s.NONCE,evaluation_candidate='B_COVERAGE',stage='P4_DEV',noise_bank=294101,GPU_uuid=parent.GPU,shared_GPU_grant_sha256=parent.GRANT_SHA,late_start_grant_sha256=late.GRANT_SHA,shared_owner_repair_scope_sha256=s.SCOPE_SHA)
def test_same_scope_new_cold_request_is_not_a_failed_request_restart():
 raw=(ROOT/'reports/PI0294/design/parallel-science-20261022/V4A_SHARED_GPU0_OWNER_FACTS_COLD_ENGINEERING_REPAIR_20261025.json').read_bytes();s.scope(raw);r=request();assert s.allowed(r) and late.start_limit(r)==late.LATEST
 with pytest.raises(PermissionError):s.scope(raw+b' ')
@pytest.mark.parametrize('key,value',[('request_id',parent.RID),('GPU_uuid','GPU-845348f1-1511-dc19-0877-cc97658e3edd'),('noise_bank',294102),('shared_owner_repair_scope_sha256','f'*64)])
def test_no_other_scope_or_unqualified_GPU1(key,value):
 r=request();r[key]=value
 with pytest.raises(PermissionError):s.allowed(r)
def test_actual_private_owner_parser_call_forwards_exact_shared_request():
 p=ROOT.parent.parent/'work/private/science-next/wdr1-target-identity/v4a-core-eval-preparation-01/prepared-B-shared-gpu0-repair/v4a_core_resident_owner_v2.py'
 if not p.exists():pytest.skip('private actual cold owner absent')
 tree=ast.parse(p.read_bytes());f=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='fresh_process');calls=[x for x in ast.walk(f) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='parse_GPU_rows'];assert len(calls)==1 and any(k.arg=='shared_request' for k in calls[0].keywords)
 raw={'devices':{'stdout':DEVICE},'processes':{'stdout':parent.GPU+', 4109280, python, 5804\n'+parent.GPU+', 249150, python, 6490\n'}};r=request();expression=ast.Expression(calls[0]);code=compile(ast.fix_missing_locations(expression),'<exact-private-owner-parser-call>','eval');v=eval(code,dict(parse_GPU_rows=n.parse_GPU_rows,raw=raw,r=r))
 assert v['human_authorized_shared_GPU_grant_sha256']==parent.GRANT_SHA
 # The old missing-forwarding call is reproduced and actually rejected.
 with pytest.raises(PermissionError):n.parse_GPU_rows(raw['devices']['stdout'],raw['processes']['stdout'],GPU_uuid=r['GPU_uuid'])
