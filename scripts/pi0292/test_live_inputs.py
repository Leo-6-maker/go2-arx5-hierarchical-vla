import base64
import hashlib
import io
import unittest
import uuid

import numpy as np
from PIL import Image

from .eval_contract import FRAME,TCP
from .live_inputs import observation_arrays


class TestObservationOnlyArrays(unittest.TestCase):
    def payload(self):
        rows=[]
        for camera,value in [('external_rgb',11),('wrist_rgb',22)]:
            output=io.BytesIO();Image.fromarray(np.full((256,256,3),value,dtype=np.uint8)).save(output,format='PNG')
            raw=output.getvalue();rows.append(dict(camera=camera,png_base64=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest()))
        return dict(snapshot_uuid=str(uuid.uuid4()),language='original task',state_eef10=[.01,.02,.03,1.,0.,0.,0.,1.,0.,1.],
            images=rows,frame=FRAME,tcp=TCP)

    def test_original_pixels_state_and_language_without_labels(self):
        payload=self.payload();state,images,text=observation_arrays(payload)
        self.assertEqual(state.dtype,np.float64)
        self.assertEqual(state.tolist(),payload['state_eef10'])
        self.assertTrue(np.all(images[0]==11));self.assertTrue(np.all(images[1]==22))
        self.assertEqual(text,'original task')

    def test_privileged_or_future_input_fields_rejected(self):
        for field in ('action','future_action','goal','object_pose','teacher_stage','history'):
            with self.subTest(field=field):
                payload=self.payload();payload[field]=[]
                with self.assertRaises(ValueError):observation_arrays(payload)

    def test_camera_order_hash_and_nonfinite_state_rejected(self):
        for mutation in ('camera','hash','nan'):
            with self.subTest(mutation=mutation):
                payload=self.payload()
                if mutation=='camera':payload['images'].reverse()
                if mutation=='hash':payload['images'][0]['sha256']='0'*64
                if mutation=='nan':payload['state_eef10'][0]=float('nan')
                with self.assertRaises(ValueError):observation_arrays(payload)
