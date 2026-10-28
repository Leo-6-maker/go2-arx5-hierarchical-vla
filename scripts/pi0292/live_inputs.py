"""Official processing of an observation-only relay payload, never labels."""
import base64
import io

import numpy as np
from PIL import Image

from .relay_protocol import validate_observation


def observation_arrays(payload):
    validate_observation(payload)
    images=[]
    for row in payload['images']:
        raw=base64.b64decode(row['png_base64'],validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            images.append(np.asarray(image).copy())
    return np.asarray(payload['state_eef10'],dtype=np.float64),images,payload['language']


def prepare(processor,payload):
    """Match native inference inputs without constructing supervised actions."""
    from gr00t.data.types import VLAStepData,MessageType
    from gr00t.data.embodiment_tags import EmbodimentTag
    state,images,text=observation_arrays(payload)
    value=VLAStepData(images={'external_rgb':[images[0]],'wrist_rgb':[images[1]]},
        states={'eef_9d':state[None,:9],'gripper_position':state[None,9:]},
        actions=None,masks=None,text=text,embodiment=EmbodimentTag.NEW_EMBODIMENT,metadata={})
    processed=processor([{'type':MessageType.EPISODE_STEP.value,'content':value}])
    batch=processor.collator([processed])
    forbidden={'action','action_mask','actions','labels','object_pose','goal','teacher_stage'}
    if set(batch)!= {'inputs'} or forbidden.intersection(batch['inputs']):
        raise ValueError('OBSERVATION_ONLY_NATIVE_INPUTS_REQUIRED')
    return batch
