"""Cold metadata-only completion; runtime/GPU inputs are explicit fixtures."""
import runpy
from copy import deepcopy
from pathlib import Path
import pytest
s=runpy.run_path(str(Path(__file__).resolve().parents[2]/'hpc/pi0294/saved512_unstarted_manifest_finalize_v1.py'))

def packet():
    m=dict(platform_training_version=s['OLD'],code=dict(vla_commit=s['SOURCE']),run_uuid='arx5-minimal-saved-e1-inference-e2-fresh512-03',execution=dict(profile='vanda-arx5-minimal-saved-e1-inference-fresh-e2-fixed512-cold03',agent_liveness_required=False,sealed_inputs={'one':'a'*64}))
    pin=s['sha'](s['raw'](m));b=dict(run_manifest_sha256=pin);l=dict(status='ACTIVE',run_uuid=m['run_uuid'],manifest_sha256=pin)
    a=dict(run_manifest_sha256=pin,status='PASS_ACTUAL_MINIMAL_FIRST512_COLD_INPUTS_NO_MODEL',source_sha=s['SOURCE'],actual_twelve_reader_rows=7212,new_GPU_PBS_model_query_update_physics=0)
    return m,b,l,a

def test_only_top_type_changes_and_all_old_bytes_are_inputs_not_rewritten_by_proposal():
    old=packet();before=deepcopy(old);new=s['proposal'](*old)
    assert old==before and {k for k in old[0]if old[0][k]!=new[0][k]}=={'platform_training_version'}
    assert new[0]['code']==old[0]['code']and new[0]['execution']==old[0]['execution']and new[0]['platform_training_version']==s['NEW']
    pin=s['sha'](s['raw'](new[0]));assert new[1]['run_manifest_sha256']==new[2]['manifest_sha256']==new[3]['run_manifest_sha256']==pin
    assert new[3]['original_actual_cold_data_audit_sha256']==s['sha'](s['raw'](old[3]))
    assert new[3]['cold_top_type_completion']['new_model_query_update_permission']is False
    with pytest.raises(PermissionError):s['proposal'](*new)

@pytest.mark.parametrize('index,key,value',[(0,'platform_training_version',s['NEW']),(1,'run_manifest_sha256','0'*64),(2,'status','CLOSED'),(3,'actual_twelve_reader_rows',0),(3,'new_GPU_PBS_model_query_update_physics',1)])
def test_wrong_or_already_used_identity_is_refused(index,key,value):
    v=list(packet());v[index][key]=value
    with pytest.raises(PermissionError):s['proposal'](*v)
