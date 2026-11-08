"""Recorded-packet identity regressions; no model or physics is constructed."""
import json
from pathlib import Path
import pytest
from go2arx5_vla.world_model import target_gr00t_baseline_v1 as baseline
from go2arx5_vla.world_model.storage import digest
from scripts.wm02.infer_fusion_vla import recorded_payload
from tests.world_model.test_target_gr00t_dataset_v1 import dataset_fixture


def test_closed_dataset_and_same_original_packet_are_required(tmp_path):
    root=dataset_fixture(tmp_path);parent=root.parent
    (parent/'control').mkdir()
    (parent/'control/LEASE.json').write_text(json.dumps(dict(status='CLOSED',worker_exit_code=0)))
    (parent/'WORKER_EXIT.private.json').write_text(json.dumps(dict(exit_code=0)))
    (parent/'RESULT.json').write_text(json.dumps(dict(status='ACTUAL_FOUR_DEMOS_PROCESSOR_AND_PRIVATE_ARCHIVE_SEALED',dataset_sha256=digest(root/'DATASET_MANIFEST.json'))))
    (parent/'EXECUTION_SPEC.json').write_text('{}')
    (parent/'control/LEASE.json').write_text(json.dumps(dict(status='CLOSED',worker_exit_code=0,spec_sha256=digest(parent/'EXECUTION_SPEC.json'),result_sha256=digest(parent/'RESULT.json'))))
    spec=dict(observation_mode=baseline.MODE,baseline_dataset=str(root),baseline_dataset_sha256=digest(root/'DATASET_MANIFEST.json'),neural_bindings=dict(source='unused',cosmos='unused'),baseline_episode='SYNTHETIC_NOT_NATIVE')
    # Fixture has no physical capture proof, so its numeric arrays are insufficient.
    with pytest.raises(KeyError):recorded_payload(spec)
    p=parent/'control/LEASE.json';p.write_text(json.dumps(dict(status='ACTIVE',worker_exit_code=0)))
    with pytest.raises(ValueError,match='ACTUAL_CORE_EXPORT_CLOSED0'):recorded_payload(spec)
    with pytest.raises(ValueError,match='SAME_SEALED_DATASET'):
        recorded_payload(dict(spec,baseline_dataset_sha256='0'*64))
