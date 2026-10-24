"""Exact approved two-cap grant and pure adoption; no native writes."""
import base64,json
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.pv1 import evaluation_core_zero_start_storage as s,evaluation_scope as scope
from scripts.pi0294.test_pv1_core_evaluation_accounting import captured_global
ROOT=Path(__file__).resolve().parents[2]
GRANT=ROOT/'reports/PI0294/design/parallel-science-20261022/V4A_B_ZERO_START_TWO_STORAGE_CAP_AUTHORIZATION_20261025.json'
def test_published_grant_is_exact_and_only_two_storage_dimensions():
 raw=GRANT.read_bytes();v=s.grant(raw);assert v['new_training'] is False
 assert set(v['approved'])=={'GPU_site_storage_GiB','Core_storage_GiB'}
 assert s.CAPS=={'Hopper_storage_bytes':811*1024**3,'Core_storage_bytes':329*1024**3}
 with pytest.raises(PermissionError):s.grant(raw+b' ')
def test_caps_require_real_exact_adoption_marker(captured_global):
 g=deepcopy(captured_global);assert scope.caps(g)['Hopper_storage_bytes']==810*1024**3
 marker=dict(schema=s.SCHEMA,grant_sha256=s.GRANT_SHA,before_sha256={'GLOBAL':'a'*64,'LOCAL':'b'*64},adopted_epoch=s.STOP_NEW-100,caps=dict(s.CAPS));g[s.FIELD]=marker
 assert scope.caps(g)['Hopper_storage_bytes']==811*1024**3 and scope.caps(g)['Core_storage_bytes']==329*1024**3
 for key,value in [('grant_sha256','f'*64),('caps',dict(s.CAPS,Core_storage_bytes=330*1024**3)),('adopted_epoch',s.STOP_NEW)]:
  bad=deepcopy(g);bad[s.FIELD][key]=value
  with pytest.raises(PermissionError):scope.caps(bad)
def test_actual_closed_captured_bytes_once_adoption_and_replay_rejected():
 p=ROOT.parent.parent/'work/private/science-next/wdr1-target-identity/v4a-core-eval-preparation-01/B-known-zero-start-failure-close-01/readback.REPLY_1792862635959165988.private.json'
 if not p.exists():pytest.skip('private exact actual dual closure bytes absent')
 v=json.loads(p.read_bytes())['value'];gb=base64.b64decode(v['global_base64']);lb=base64.b64decode(v['LOCAL_base64']);g,l=json.loads(gb),json.loads(lb)
 z=s.adopt(gb,lb,GRANT.read_bytes(),now_epoch=s.STOP_NEW-100);assert z['IO_performed'] is False
 for k in ('used','reserved','CPU_s','CPU_reserved_s','PBS_submissions','optimizer_updates','training_runs'):assert z['proposed_global'][k]==g[k]
 for k in ('GPU_allocated_s','GPU_reserved_s','CPU_s','CPU_reserved_s','neural_queries','physics_ticks'):assert z['proposed_LOCAL'][k]==l[k]
 with pytest.raises(PermissionError):s.adopt(json.dumps(z['proposed_global']).encode(),json.dumps(z['proposed_LOCAL']).encode(),GRANT.read_bytes(),now_epoch=s.STOP_NEW-99)
