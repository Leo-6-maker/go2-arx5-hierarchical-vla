"""C09 captured whole inputs; later live process readings are explicit CPU fixtures."""
import base64,json
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
import pytest
from scripts.pi0294.pv1 import mix_core_admission as admission,mix_core_identity as identity,mix_core_accounting as c,mix_accounting as a,mix_core_desktop as desktop
from scripts.pi0294.test_v5_core_actual_caps_admission import args

@pytest.fixture
def later():
 root=Path(__file__).resolve().parents[2].parent/'private/minimal-adaptation-20261027';files=sorted(root.glob('CORE_FAILED09_TERMINAL_READONLY_*private.json'))
 if not files:pytest.skip('private C09 captured complete native inputs absent')
 cap=json.loads(files[-1].read_bytes())['value'];b={k:base64.b64decode(v['base64'])for k,v in cap['files'].items()};kw=args(b);g=json.loads(kw['global_bytes']);l=json.loads(kw['local_bytes']);rid=cap['request_id'];prepared=json.loads((root/'CORE_RETRY09_C50_PEERS_PREPARED.private.json').read_bytes())['proposal']['proposed_global'];g[c.FIELD][rid]=deepcopy(l[c.FIELD][rid]);g['CPU_s']=prepared['CPU_s'];g['CPU_reserved_s']=prepared['CPU_reserved_s'];g['revision']-=1;kw['global_bytes']=a.raw(g);assert a.sha(kw['global_bytes'])==json.loads(kw['snapshot_bytes'])['GLOBAL_sha256'];p=json.loads(kw['native_facts_bytes']);now=kw['now_epoch'];assert now-p['started_epoch']>600
 p['observed_epoch']=now-.0001;p['device']['exact_desktop_coexistence']['kernel_read']['observed_epoch']=now-.0002
 kw['native_facts_bytes']=a.raw(p)
 return kw


def test_captured_full_post600_resource_recompile_with_same_live_bound_fixture(later):
 with patch('time.time',return_value=later['now_epoch']):v=admission.recompute(**later)
 assert v['shortfalls']=={}and not v['model_permission']and not v['physics_authorized']


def test_same_later_process_is_not_an_initial_binding(later):
 m=json.loads(later['manifest_bytes']);r=json.loads(later['request_bytes']);p=json.loads(later['native_facts_bytes']);base=json.loads(later['global_bytes']);row=base[c.FIELD][r['request_id']]
 # Preserve the real original request and receipt identities; only the fresh
 # observed process is the explicit later-read CPU fixture above.
 root=Path(__file__).resolve().parents[2].parent/'private/minimal-adaptation-20261027';cap=json.loads(sorted(root.glob('CORE_FAILED09_TERMINAL_READONLY_*private.json'))[-1].read_bytes())['value'];v=json.loads(base64.b64decode(cap['files']['ALLOCATION']['base64']));v['native_process']=p;v['native_process_sha256']=c.digest(p)
 with pytest.raises(PermissionError,match='INITIAL_PROCESS_BIND_WITHIN120S'):
  identity.sealed_identity(m,r,v,run_uuid=r['run_uuid'],manifest_sha256=a.sha(later['manifest_bytes']),request_sha256=a.sha(later['request_bytes']))
 assert identity.sealed_identity(m,r,v,run_uuid=r['run_uuid'],manifest_sha256=a.sha(later['manifest_bytes']),request_sha256=a.sha(later['request_bytes']),same_bound_process=row)==v


@pytest.mark.parametrize('kind',['stale-process','stale-desktop','new-pid','new-procstart','new-boot','new-start','foreign-GPU','outside-wall'])
def test_post600_does_not_allow_stale_changed_or_unbound_process(later,kind):
 p=json.loads(later['native_facts_bytes']);now=later['now_epoch']
 if kind=='stale-process':p['observed_epoch']=now-121;p['device']['exact_desktop_coexistence']['kernel_read']['observed_epoch']=now-121.1
 elif kind=='stale-desktop':p['device']['exact_desktop_coexistence']['kernel_read']['observed_epoch']=now-121
 elif kind=='new-pid':p['pid']+=1
 elif kind=='new-procstart':p['proc_start']+=1
 elif kind=='new-boot':p['boot_id']='8a193079-46c9-4698-9f72-8b24e06c4ed6'
 elif kind=='new-start':p['started_epoch']+=1
 elif kind=='foreign-GPU':p['native_GPU_raw']['processes']['stdout']+=f"{desktop.GPU}, 1234, /usr/bin/python, 10\n"
 else:p['observed_epoch']=p['started_epoch']+28801
 later['native_facts_bytes']=a.raw(p)
 with patch('time.time',return_value=now),pytest.raises(PermissionError):admission.recompute(**later)
