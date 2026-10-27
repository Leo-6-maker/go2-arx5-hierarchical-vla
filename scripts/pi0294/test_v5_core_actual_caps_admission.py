"""Full historical native input CPU replay; no live model or physics GO."""
import base64,json
from pathlib import Path
from unittest.mock import patch
import pytest
from scripts.pi0294.pv1 import mix_core_admission as admission,mix_accounting as a
from scripts.pi0294.pv1.evaluation_admission import adopted_caps
@pytest.fixture
def actual():
    root=Path(__file__).resolve().parents[2].parent/'private/minimal-adaptation-20261027'
    files=sorted(root.glob('CORE_FAILED05_TERMINAL_READONLY_*private.json'))
    if not files:pytest.skip('private native capture absent')
    p=json.loads(files[-1].read_bytes())['value'];b={k:base64.b64decode(v['base64'],validate=True)for k,v in p['files'].items()}
    for k,v in b.items():assert a.sha(v)==p['files'][k]['sha256']
    return b

def args(b):
    snap=json.loads(b['SNAPSHOT'])
    return dict(manifest_bytes=b['MANIFEST'],request_bytes=b['REQUEST'],global_bytes=b['GLOBAL'],local_bytes=b['LOCAL'],snapshot_bytes=b['SNAPSHOT'],training_capture_bytes=b['CAPTURE'],population_bytes=b['POPULATION'],plan_bytes=b['PLAN'],prior_bytes=b['PRIOR'],native_facts_bytes=b['NATIVE_STARTED'],now_epoch=snap['observed_epoch']+.001)

def test_full_actual_caps_native_input_replay(actual):
    v=args(actual)
    with patch('time.time',return_value=v['now_epoch']):p=admission.recompute(**v)
    assert p['shortfalls']=={}and not p['model_permission']and not p['physics_authorized']
    assert p['dimensions']['Hopper_storage_bytes']['cap']==1399*1024**3
    assert p['dimensions']['Core_storage_bytes']['cap']==550*1024**3
    assert p['dimensions']['H200_s']['final']==72000
    assert p['dimensions']['PBS_submissions']['matrix']==3

def test_latest_adopted_caps_win_and_old_markers_unchanged(actual):
    g=json.loads(actual['GLOBAL']);before=a.raw(g);caps=adopted_caps(g)
    assert caps['H200_s']==147*3600 and caps['PBS_submissions']==39
    assert caps['Hopper_CPU_s']==265*3600 and caps['Core_CPU_s']==380*3600
    assert caps['transfer_bytes']==455*1024**3 and a.raw(g)==before

@pytest.mark.parametrize('field',['GLOBAL_sha256','LOCAL_sha256'])
def test_actual_raw_counterpart_mutation_rejected(actual,field):
    v=args(actual);s=json.loads(v['snapshot_bytes']);s[field]='0'*64;v['snapshot_bytes']=a.raw(s)
    with patch('time.time',return_value=v['now_epoch']),pytest.raises(PermissionError):admission.recompute(**v)
