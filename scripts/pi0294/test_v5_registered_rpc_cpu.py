"""Exact captured RPC restriction plus actual local kernel child; no physics."""
import base64,json,os,subprocess,sys
from pathlib import Path
import pytest
from scripts.pi0294.pv1.successor_core_context import validate_cpu_affinity

def test_saved_actual_rpc_is_separate_from_declared_physical_cpu():
    root=Path(__file__).resolve().parents[2].parent/'private/minimal-adaptation-20261027';files=sorted(root.glob('CORE_FAILED06_TERMINAL_READONLY_*private.json'))
    if not files:pytest.skip('explicit private native capture absent')
    v=json.loads(files[-1].read_bytes())['value'];m=json.loads(base64.b64decode(v['files']['PHYSICAL_LEDGER']['base64']));assert m['engine_starts']==m['live_and_recovery_integrations']==0
    # The sealed private resident bootstrap pins core transport to CPU6; the
    # physical manifest keeps its original physics5/rpc6 unchanged.
    owner=json.loads((root/'CORE_RETRY06_C50_PEERS_PREPARED.private.json').read_bytes());manifest=json.loads(base64.b64decode(owner['assembly']['rows']['core']['manifest_bytes_base64']));limits=manifest['limits'];assert limits['cpus']==[5]and limits['rpc_cpu']==6
    assert not set(limits['cpus'])<={6}
    assert validate_cpu_affinity(limits,{6})['registered_RPC_singleton']
    assert not validate_cpu_affinity(limits,{6})['new_model_or_physics_permission']

@pytest.mark.parametrize('cpus',[set(),{7},{8},{6,8}])
def test_arbitrary_partial_or_foreign_cpu_rejected(cpus):
    with pytest.raises(PermissionError):validate_cpu_affinity(dict(cpus=[6,7],rpc_cpu=6),cpus)

def test_original_parent_grant_and_GEN_without_rpc_stay_valid():
    assert not validate_cpu_affinity(dict(cpus=[6,7]),set(range(28)))['registered_RPC_singleton']
    assert not validate_cpu_affinity(dict(cpus=[6,7],rpc_cpu=6),{6,7})['registered_RPC_singleton']

@pytest.mark.parametrize('declared',[[6,6],[],[True,7],[-1,7],[1,2,3,4]])
def test_bad_manifest_cpu_list_rejected(declared):
    with pytest.raises(PermissionError):validate_cpu_affinity(dict(cpus=declared,rpc_cpu=6),set(range(28)))

def test_actual_kernel_child_pin_to_one_registered_cpu():
    allowed=sorted(os.sched_getaffinity(0))
    if len(allowed)<2:pytest.skip('needs two actual allowed CPU identifiers')
    selected=allowed[:2];program='import os,json;from scripts.pi0294.pv1.successor_core_context import validate_cpu_affinity;os.sched_setaffinity(0,{'+str(selected[0])+'});print(json.dumps(validate_cpu_affinity('+repr(dict(cpus=selected[1:],rpc_cpu=selected[0]))+',os.sched_getaffinity(0))))'
    result=subprocess.run([sys.executable,'-c',program],capture_output=True,text=True,timeout=10,check=True)
    v=json.loads(result.stdout);assert v['actual_cpus']==selected[:1]and v['registered_RPC_singleton']and not v['new_model_or_physics_permission']
