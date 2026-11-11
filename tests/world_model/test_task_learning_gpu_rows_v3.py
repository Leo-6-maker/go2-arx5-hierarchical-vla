import pytest
import torch

from go2arx5_vla.world_model.task_learning_gpu_rows_v3 import compact_row,row_bytes


def test_compact_row_keeps_every_head_tensor_exact_and_does_not_modify_source():
    inputs={k:torch.arange(5,dtype=torch.float32)for k in ('state','embodiment_id','action','action_mask')}
    inputs['pixel_values']=torch.ones(200)
    row=dict(identity=dict(episode_id='e',split='TRAIN',anchor=1),inputs=inputs,
        features={k:torch.arange(8,dtype=torch.float32)for k in ('backbone_features','backbone_attention_mask','image_mask')})
    compact=compact_row(row,device='cpu')
    assert set(compact['inputs'])=={'state','embodiment_id','action','action_mask'}
    assert 'pixel_values'in row['inputs']
    assert row_bytes(compact)==4*(5*4+8*3)
    for group in ('inputs','features'):
        for key,value in compact[group].items():
            assert torch.equal(value,row[group][key])and value.data_ptr()!=row[group][key].data_ptr()


def test_unknown_backbone_feature_is_not_silently_dropped():
    row=dict(inputs={k:torch.zeros(1)for k in ('state','embodiment_id','action','action_mask')},
        features=dict(backbone_features=torch.zeros(1),backbone_attention_mask=torch.zeros(1),
                      image_mask=torch.zeros(1),unknown=torch.zeros(1)))
    with pytest.raises(ValueError,match='EXACT_ORIGINAL_HEAD_AND_FEATURE_KEYS'):compact_row(row,device='cpu')
