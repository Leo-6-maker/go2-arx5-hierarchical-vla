import base64
import hashlib
import io
from types import SimpleNamespace
from unittest.mock import patch
import uuid

from PIL import Image
import pytest

torch=pytest.importorskip('torch')
from scripts.pi0292 import live_inputs
from scripts.pi0292.eval_contract import FRAME,TCP
from go2arx5_vla.flowdagger.native import context_from_observation


class ToyContextModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.eval();self.backbone_calls=0;self.feature_calls=0
        self.action_head=SimpleNamespace(_encode_features=self.encode)

    def prepare_input(self,inputs):
        assert set(inputs)=={'state','embodiment_id'}
        return inputs,SimpleNamespace(**inputs)

    def backbone(self,inputs):
        self.backbone_calls+=1
        return SimpleNamespace(backbone_features=torch.ones(1,2,3))

    def encode(self,bb,action):
        self.feature_calls+=1
        return SimpleNamespace(backbone_features=bb.backbone_features,state_features=action.state)

    def get_action(self,*args,**kwargs):
        raise AssertionError('NO_DISCARDED_POLICY_DRAW_FOR_FEATURE_CAPTURE')


def payload():
    stream=io.BytesIO();Image.new('RGB',(256,256)).save(stream,format='PNG');raw=stream.getvalue()
    return dict(snapshot_uuid=str(uuid.uuid4()),language='push the object',
        state_eef10=[.3,.1,.5,1,0,0,0,1,0,.8],frame=FRAME,tcp=TCP,
        images=[dict(camera=name,png_base64=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest())
                for name in ('external_rgb','wrist_rgb')])


def test_context_one_backbone_no_default_action_draw():
    model=ToyContextModel();p=payload()
    batch=dict(inputs=dict(state=torch.ones(1,1,2),embodiment_id=torch.zeros(1,dtype=torch.long)))
    with patch.object(live_inputs,'prepare',return_value=batch) as prepare:
        context=context_from_observation(model,object(),p)
    assert context.observation_id==p['snapshot_uuid']
    assert model.backbone_calls==model.feature_calls==prepare.call_count==1


def test_privileged_teacher_fields_rejected_before_backbone_or_processor():
    model=ToyContextModel();p=payload();p['object_pose']=[0,0,0]
    with patch.object(live_inputs,'prepare') as prepare:
        with pytest.raises(ValueError):context_from_observation(model,object(),p)
    assert model.backbone_calls==prepare.call_count==0
