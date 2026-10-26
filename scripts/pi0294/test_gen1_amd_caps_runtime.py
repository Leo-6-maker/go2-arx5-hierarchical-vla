"""Captured whole ledger cap resolution, not new request/native admission."""
import base64,json,os
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.pv1 import mix_accounting as original
from scripts.pi0294.generalization import amd_adoption as a

W=Path(os.environ.get('PI0294_PRIVATE_WORKSPACE',str(Path(__file__).resolve().parents[4])))
P=W/'work/private/v5-execution-20261025/GEN1_AMD_REMAINING10_INDEPENDENT_GLOBAL_RESERVE_READBACK.private.json'
@pytest.fixture
def actual():
 if not P.exists():pytest.skip('private captured current GLOBAL unavailable')
 return json.loads(base64.b64decode(json.loads(P.read_bytes())['GLOBAL_base64']))

def test_original_current_ledger_and_exact_granted_limits(actual):
 before=deepcopy(actual);original.validate_global(actual);assert actual==before
 # A proposal only: the extra approved PBS count is not a real job. This
 # verifies the old hard-coded34 seam without creating/registering any row.
 actual['used']['PBS_submissions']+=2;original.validate_global(actual)
 actual['used']['PBS_submissions']+=1
 with pytest.raises(PermissionError):original.validate_global(actual)

def test_old_scope_keeps_34_without_new_grant(actual):
 actual.pop(a.FIELD);original.validate_global(actual)
 actual['used']['PBS_submissions']+=1
 with pytest.raises(PermissionError):original.validate_global(actual)

@pytest.mark.parametrize('bad',['cap','grant','clock','LOCAL'])
def test_no_cap_from_unadopted_or_modified_marker(actual,bad):
 m=actual[a.FIELD]
 if bad=='cap':m['caps']['PBS_submissions']=99
 elif bad=='grant':m['grant_sha256']='0'*64
 elif bad=='clock':m['T_CD']-=1
 else:m['LOCAL_sync']='FAKE_FRESH'
 with pytest.raises(PermissionError):original.validate_global(actual)
