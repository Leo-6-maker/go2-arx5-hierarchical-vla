"""Transfer inventory fixtures check bytes and routing, not native admission."""
import copy
import json
import numpy as np
import pytest
from PIL import Image
from test_task_learning_native_dataset_v1 import manifest, save_manifest
from test_task_learning_dataset_v1 import packet
from go2arx5_vla.world_model.storage import digest
from go2arx5_vla.world_model.task_learning_native_dataset_v1 import sample_rows
from go2arx5_vla.world_model.task_learning_transfer_v1 import inventory


def bundle(root):
    m = manifest(root)
    m['bundle_kind'] = 'FORMAL_TASK_LEARNING_V1'
    for i in range(4):
        r = copy.deepcopy(m['episodes'][i])
        r.update(id=f'dev-{i}', parent=f'DEV{i}', split='DEV', purpose='DEV_REFERENCE_NOT_BC',
                 file=f'episodes/dev-{i}.npz', optimization_anchors=[])
        m['episodes'].append(r)
    image = root / 'rgb.png'
    Image.new('RGB', (256, 256), (10, 20, 30)).save(image)
    (root / 'episodes').mkdir()
    for r in m['episodes']:
        p = packet()
        p.update(rgb_paths=np.full((15, 2), 'rgb.png'),
                 rgb_sha256=np.full((15, 2), digest(image)))
        np.savez_compressed(root / r['file'], **p)
        r['sha256'] = digest(root / r['file'])
    save_manifest(root, m)
    (root / 'EXPORT_RESULT.private.json').write_text(json.dumps(dict(
        status='FORMAL_CONTINUOUS_BUNDLE_CREATED_WAIT_ACTUAL_PROCESSOR',
        manifest_sha256=digest(root / 'DATASET_MANIFEST.json'),
        statistics_sha256=m['statistics_sha256'])))
    processor = root / 'target-processor-v1'
    processor.mkdir()
    (processor / 'processor_config.json').write_text('{}')
    observed = sample_rows(m['episodes'], 'TRAIN', False)
    proof = dict(status='PASS_ACTUAL_OFFICIAL_PROCESSOR_ROUNDTRIP',
        dataset_family='TASK_LEARNING_CONTINUOUS_V1',
        manifest_sha256=digest(root / 'DATASET_MANIFEST.json'),
        statistics_sha256=m['statistics_sha256'], processed_TRAIN_windows=len(observed),
        records=[dict(episode=r['id'], anchor=a) for r, a in observed],
        fixed_component_tolerance=1e-6, maximum_abs_error=0., TRAIN_only_statistics=True,
        processor_files={'processor_config.json': digest(processor / 'processor_config.json')})
    (root / 'PROCESSOR_TRAINING_ADMISSION.json').write_text(json.dumps(proof))
    return proof


def test_explicit_inventory_preserves_all_images_episodes_and_core_processor(tmp_path):
    bundle(tmp_path)
    (tmp_path / 'unrelated-private-file').write_text('excluded')
    result = inventory(tmp_path)
    assert result['files']['rgb.png'] == digest(tmp_path / 'rgb.png')
    assert sum(name.endswith('.npz') for name in result['files']) == 104
    assert 'target-processor-v1/processor_config.json' in result['files']
    assert 'unrelated-private-file' not in result['files']
    assert result['bytes'] == sum((tmp_path / n).stat().st_size for n in result['files'])


@pytest.mark.parametrize('change', ['image', 'processor', 'family', 'anchor'])
def test_changed_media_processor_or_actual_roundtrip_cannot_be_transferred(tmp_path, change):
    proof = bundle(tmp_path)
    if change == 'image':
        (tmp_path / 'rgb.png').write_bytes(b'changed')
    elif change == 'processor':
        (tmp_path / 'target-processor-v1/processor_config.json').write_text('changed')
    else:
        if change == 'family': proof['dataset_family'] = 'TARGET_SHORT_V1'
        else: proof['records'][0]['anchor'] = 14
        (tmp_path / 'PROCESSOR_TRAINING_ADMISSION.json').write_text(json.dumps(proof))
    with pytest.raises(ValueError): inventory(tmp_path)
