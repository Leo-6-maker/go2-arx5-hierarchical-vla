"""Closed Vanda reader provenance fixtures; no payload/model qualification."""
from pathlib import Path
from unittest.mock import patch
import pytest
from scripts.pi0292.durable import read
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.dec1 import a40_training_context as c,a40_allocation as a
from scripts.pi0294.dec1.test_a40_training_context import consumers_fixture
from scripts.pi0294.dec1.test_a40_training_runtime import put
from scripts.pi0294.pv1 import bank_reuse as b,evaluation_platform as e

@pytest.mark.parametrize('fault',[None,'lease','registry','receipt','sealed','delivery','target','profile'])
def test_closed_CPU_readers_remain_mandatory_before_bank_reuse(tmp_path,fault):
    root,records=consumers_fixture(tmp_path)
    # Existing fixture models two sibling readers; retain the same relationship
    # as real children without weakening the original reader validator.
    source=Path(records['new139']['root']);target=source/'datasets/FROZEN_TRAIN_BANK'
    delivery=dict(revision='b'*40)
    put(source/'plans/PV1_HF_BANK_DELIVERY.json',delivery);put(root/'plans/PV1_HF_BANK_DELIVERY.json',delivery)
    put(target/'READY.json',{});put(target/'statistics.json',{})
    m=read(root/'RUN_MANIFEST.json');m.update(platform_evaluation_version=e.VERSION,scope=e.ROLE,
      execution=dict(profile=e.PROFILE,sealed_inputs={c.CONSUMERS:sha(root/c.CONSUMERS)}))
    put(root/'RUN_MANIFEST.json',m)
    stage=dict(new_bank_manifest_sha256=a.BANK,new_bank_ready_sha256=sha(target/'READY.json'))
    if fault=='lease':
      v=read(source/'lease.json');v['status']='ACTIVE';put(source/'lease.json',v)
    if fault=='registry':put(source/'registry.json',dict(entries=[]))
    if fault=='receipt':put(source/'RESULT.private.json',dict(status='PASS'))
    if fault=='sealed':(root/c.CONSUMERS).chmod(0o600)
    if fault=='delivery':put(source/'plans/PV1_HF_BANK_DELIVERY.json',dict(revision='c'*40))
    if fault=='target':target=Path(records['original126_C1']['root'])/'datasets/original'
    if fault=='profile':
      m['execution']['profile']='vanda-neural-training';put(root/'RUN_MANIFEST.json',m)
    real=c.sha
    def identity(p):
      p=Path(p)
      if p.name=='DATASET_MANIFEST.json':
        from scripts.pi0294.e1.scope import DATA
        from scripts.pi0294.dec1.source_data_original import C1_MANIFEST
        return a.BANK if p.parent.name=='FROZEN_TRAIN_BANK' else DATA if p.parent.name=='original' else C1_MANIFEST
      return real(p)
    with patch.object(c,'sha',side_effect=identity),patch.object(b,'STATS',sha(source/'datasets/FROZEN_TRAIN_BANK/statistics.json')):
      if fault is None:
        v=b.validate(root,target,delivery,stage)
        assert v['prior_consumption_does_not_replace_current_hash_verification'] and set(v['closed_CPU_readers'])==set(records)
      else:
        with pytest.raises(PermissionError):b.validate(root,target,delivery,stage)
