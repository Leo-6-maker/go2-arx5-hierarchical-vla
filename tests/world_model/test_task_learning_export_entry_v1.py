"""The CPU entry preserves the full-population gate and existing outputs."""
from argparse import Namespace
import pytest
from test_task_learning_collection_v1 import plan, save
from test_task_learning_export_v1 import group
from go2arx5_vla.world_model import task_learning_grasp_v1 as task
from scripts.wm02.export_task_learning_dataset_v1 import run


def test_partial_collection_cannot_create_export_or_admission(tmp_path):
    panel = plan(tmp_path / 'proofs', task.COLLECTION_ROLE)
    path = tmp_path / 'PANEL.json'
    pin = save(path, panel)
    root = tmp_path / 'group'
    group(root, panel, pin['sha256'], [0, 50])
    output = tmp_path / 'export'
    with pytest.raises(ValueError, match='ALL100_ATTEMPTS'):
        run(Namespace(operation='train-base', panel=path, group=[root], output=output))
    assert not output.exists()


def test_existing_receipt_is_preserved_before_any_export(tmp_path):
    output = tmp_path / 'export'
    output.mkdir()
    receipt = output / 'EXPORT_RESULT.private.json'
    receipt.write_text('original evidence')
    with pytest.raises(ValueError, match='PRESERVE_EXISTING_DATA_EXPORT_RECEIPT'):
        run(Namespace(operation='train-base', output=output))
    assert receipt.read_text() == 'original evidence'


def test_formal_requires_actual_base_before_creating_output(tmp_path):
    base = tmp_path / 'base'
    base.mkdir()
    (base / 'DATASET_MANIFEST.json').write_text('{}')
    output = tmp_path / 'formal'
    with pytest.raises(ValueError, match='DECLARED_SEALED_TRAIN_BASE'):
        run(Namespace(operation='formal', train_base=base,
                      train_base_sha256='a' * 64, output=output))
    assert not output.exists()
