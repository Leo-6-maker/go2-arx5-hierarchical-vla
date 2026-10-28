"""Actual C08 historical bytes; current-process calls remain CPU mocked, never native GO."""
import base64,json
from pathlib import Path
from unittest.mock import patch
from copy import deepcopy
import pytest
from scripts.pi0294.pv1 import mix_core_identity as cd,mix_core_desktop as desktop,evaluation_core_identity as original

@pytest.fixture
def actual():
 root=Path(__file__).resolve().parents[2].parent/'private/minimal-adaptation-20261027'
 paths=sorted(root.glob('CORE_FAILED08_ORIGINAL_OWNER_CLOSED_READONLY_*private.json'))
 if not paths:pytest.skip('private actual historical C08 capture absent')
 v=json.loads(paths[-1].read_bytes())['value'];docs={k:base64.b64decode(f['base64'])for k,f in v['files'].items()}
 return docs,json.loads(docs['MANIFEST']),json.loads(docs['REQUEST']),json.loads(docs['ALLOCATION'])

def sealed(m,r,a):
 return cd.sealed_identity(m,r,a,run_uuid=r['run_uuid'],manifest_sha256=a['manifest_sha256'],request_sha256=a['request_sha256'])

@pytest.mark.parametrize('age',[601,1201,3601])
def test_original_saved_identity_reparses_at_bound_epoch_only(actual,age):
 docs,m,r,a=actual;p=a['native_process']
 with patch('time.time',return_value=p['observed_epoch']+age):
  assert sealed(m,r,a)==a
  with pytest.raises(PermissionError,match='FRESH_ACTUAL_KERNEL_READ'):
   original.parse_GPU_rows(p['native_GPU_raw']['devices']['stdout'],p['native_GPU_raw']['processes']['stdout'],GPU_uuid=r['GPU_uuid'],desktop_request=r,desktop_proof=p['device']['exact_desktop_coexistence']['kernel_read'])


def test_current_identity_cannot_use_saved_proof_instead_of_new_read(actual,tmp_path):
 docs,m,r,a=actual;root=tmp_path/r['run_uuid'];directory=root/'pbs/alloc-01';directory.mkdir(parents=True)
 (root/'RUN_MANIFEST.json').write_bytes(docs['MANIFEST']);(directory/'REQUEST.json').write_bytes(docs['REQUEST']);(directory/'ALLOCATION_ADMISSION.json').write_bytes(docs['ALLOCATION']);(directory/'ALLOCATION_ADMISSION.json').chmod(0o400)
 stat=Path.stat
 def own_stat(path,*args,**kw):
  v=stat(path,*args,**kw)
  if path==directory/'ALLOCATION_ADMISSION.json':
   from types import SimpleNamespace
   return SimpleNamespace(st_mode=v.st_mode,st_uid=2009)
  return v
 with patch.object(Path,'stat',own_stat),patch.object(original,'observe_current',side_effect=PermissionError('ACTUAL_NEW_KERNEL_READ_REQUIRED'))as fresh,pytest.raises(PermissionError,match='ACTUAL_NEW_KERNEL_READ_REQUIRED'):
  original.current_identity(root,m,r,a)
 assert fresh.call_count==1


def test_actual_81_raw_samples_have_600s_but_no_qualification_receipt(actual):
 docs,m,r,a=actual;rows=[json.loads(x)for x in docs['READS'].splitlines()]
 assert len(rows)==81 and rows[-1]['monotonic_s']-rows[0]['monotonic_s']>=600
 assert all(v['proof']['heartbeat']['run_uuid']==r['run_uuid']and v['proof']['heartbeat_written']is False for v in rows)
 assert json.loads(docs['HISTORY'])['history']['valid_conditions']==0
