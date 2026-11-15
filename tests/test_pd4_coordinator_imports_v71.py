"""The coordinator's real identity chain must work without numerical workers."""
import os
from pathlib import Path
import subprocess
import sys


def test_coordinator_identity_and_package_contract_do_not_import_worker_dependencies():
    code = r'''
import importlib.abc,sys
class BlockWorkerDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'scipy','jax','flax','torch','mujoco'}:
            raise AssertionError('Coordinator imported worker dependency: '+fullname)
sys.meta_path.insert(0,BlockWorkerDependencies())
from go2arx5_vla.sla_adapter.pd4_config_v71 import MANIFESTS,STATISTICS
from go2arx5_vla.sla_adapter.provider_v3 import contract,SCHEMA,DECODE_ORDER,VERSION,EPSILON
from go2arx5_vla.world_model.pd4_student_v71 import policy_contract,protocol
from scripts.wm02.sla_pd4_student_queue_v71 import terminal_training
from scripts.wm02.sla_pd4_sampling_queue_v71 import dependency_cold
contract(dict(schema=SCHEMA,codec=VERSION,decode_order=DECODE_ORDER,label_epsilon=EPSILON,
    bindings=dict(codec_version=VERSION),state_gripper_transformed=False,
    teacher_required=False,training_cache_required=False,optimizer_required=False))
try:
    policy_contract({'arm':'OLD_UNKNOWN_POLICY','package':{},'qualification':{}})
except PermissionError as e:
    assert str(e)=='PD4_ONLY_TWO_FIXED500_RAW_POLICIES'
else:
    raise AssertionError('Invalid policy was accepted')
assert len(MANIFESTS)==len(STATISTICS)==2 and protocol()['policy_queries']==60
'''
    root=Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable,'-B','-c',code],cwd=root,
        env=dict(os.environ,PYTHONPATH=str(root/'src')+':'+str(root)),check=True)
