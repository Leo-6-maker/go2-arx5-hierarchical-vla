"""Serialization fixtures do not establish native or Processor admission."""
import json
import numpy as np
import pytest
from PIL import Image
from test_task_learning_native_dataset_v1 import manifest
from test_task_learning_dataset_v1 import packet
from go2arx5_vla.world_model.task_learning_bundle_v1 import member,finalize
from go2arx5_vla.world_model.task_learning_export_v1 import write_episode
from go2arx5_vla.world_model.task_learning_dataset_v1 import fit_statistics
from go2arx5_vla.world_model.storage import digest


def test_claimed_population_without_actual_data_cannot_create_a_formal_bundle(tmp_path):
    base=tmp_path/'base';manifest(base);out=tmp_path/'formal'
    with pytest.raises(ValueError,match='SEALED_BASE_REGULAR_MEMBER'):
        finalize(base,digest(base/'DATASET_MANIFEST.json'),tmp_path/'missing-dev',out)
    assert not out.exists()


def test_base_transfer_rejects_path_escape_and_symlinks(tmp_path):
    base=tmp_path/'base';base.mkdir();p=tmp_path/'outside';p.write_text('bytes')
    with pytest.raises(ValueError):member(base,'../outside',digest(p))
    (base/'link').symlink_to(p)
    with pytest.raises(ValueError):member(base,'link',digest(p))


def test_dev_serialization_keeps_all_true_decisions_without_optimizer_exposure(tmp_path):
    output=tmp_path/'output';output.mkdir();(output/'episodes').mkdir();(output/'media').mkdir()
    img=tmp_path/'real-format.png';Image.new('RGB',(256,256)).save(img)
    p=packet();p['valid_window'][:]=True
    e=dict(row=dict(native_run='dev-fixture',split='DEV',parent='DEV01',command='task',purpose='DEV_REFERENCE_NOT_BC'),
        captures=[dict(images=[img,img])for _ in range(15)],packet=p,
        optimization_anchor_schedule=[0]*32,genuine_recovery_success=False,proof=dict(fixture=True))
    row=write_episode(e,output)
    assert row['split']=='DEV'and row['purpose']=='DEV_REFERENCE_NOT_BC'
    assert row['optimization_anchors']==[]and row['supervised_anchors']==list(range(15))
    with np.load(output/row['file'],allow_pickle=False)as z:
        assert np.array_equal(z['actual_state10'],p['actual_state10'])
        assert np.array_equal(z['committed_absolute10'],p['committed_absolute10'])
        assert z['rgb_paths'].shape==(15,2)
    with pytest.raises(ValueError):fit_statistics([e])
