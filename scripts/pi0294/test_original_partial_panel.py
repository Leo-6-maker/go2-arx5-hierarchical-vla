"""Use captured actual C12 bytes, never model/raw-physics construction."""
import base64,json,os
from pathlib import Path
from copy import deepcopy
import pytest
from scripts.pi0294.pv1 import partial_panel as p

@pytest.fixture
def captured():
 path=os.environ.get('V5_C12_HISTORY_CAPTURE')
 if not path:pytest.skip('Private original twelve-attempt capture required')
 value=json.loads(Path(path).read_bytes());value['evidence_decoded']=[{k:base64.b64decode(v)for k,v in row.items()}for row in value['evidence']]
 return value

def actual_history(v):return p.history(base64.b64decode(v['population_base64']),v['evidence_decoded'],checkpoint_manifest_sha256=v['history']['checkpoint_manifest_sha256'])
def forecast(v,h=None):
 return p.forecast(base64.b64decode(v['GLOBAL_base64']),base64.b64decode(v['LOCAL_base64']),validated_history=h or actual_history(v),native_proof=v['native_proof'],effects=v['effects'],closed_leases={k:x['value']for k,x in v['closed_readback']['leases'].items()})

def test_actual_eleven_outcomes_and_one_partial_only_thirteen_remaining(captured):
 h=actual_history(captured);assert h['successes']=={'T1':1,'T2':0}
 f=forecast(captured,h);assert f['shortfalls']=={} and f['native_hold_restored']is False
 assert f['dimensions']['PBS_submissions']['projected']==39
 assert f['resources']['optimizer_updates']==f['resources']['PBS_submissions']==0
 assert f['aggregate_paid_panel_IO']['native_all_retained_outputs_and_model_inode_join']=='NOT_RUN'

def test_replaying_original_valid_failure_or_dropping_attempt_rejected(captured):
 v=deepcopy(captured);v['evidence_decoded'].append(v['evidence_decoded'][0])
 with pytest.raises(PermissionError):actual_history(v)
 v=deepcopy(captured);v['evidence_decoded'].pop(0)
 with pytest.raises(PermissionError):actual_history(v)

def test_unknown_or_not_closed_cannot_be_a_zero_effect_retry(captured):
 v=deepcopy(captured);v['effects']['unknown_query_intents']=[573]
 with pytest.raises(PermissionError):forecast(v)
 v=deepcopy(captured);v['closed_readback']['leases']['control']['value']['status']='ACTIVE'
 with pytest.raises(PermissionError):forecast(v)
