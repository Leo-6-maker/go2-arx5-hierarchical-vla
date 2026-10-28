"""CPU packet fixtures only, never a physical or double-SSH qualification."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
import uuid
from unittest.mock import patch

import numpy as np
from PIL import Image

from . import relay_teacher as module
from scripts.pi0292.durable import read,write
from scripts.pi0292.relay_protocol import RelayExchange,sealed,unseal
from scripts.pi0294.run_admission import sha
from go2arx5_vla.task_data.codec import pose10


class RelayTeacherTests(unittest.TestCase):
    def test_fresh_prefix_reserved_and_acknowledged_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'episode';folder.mkdir()
            (root/'RUN_MANIFEST.json').write_text('{}')
            for camera in ('body_rgb','wrist_rgb'):
                (folder/camera).mkdir()
                Image.fromarray(np.zeros((256,256,3),dtype=np.uint8)).save(folder/camera/'0000.png')
            identity=dict(profile='TEACHER_REFERENCE',model_revision='NOT_APPLICABLE',source_sha='a'*40,
                config_sha256='b'*64,processor_sha256='c'*64,statistics_sha256='d'*64)
            request=dict(run_uuid=root.name,attempt_id='fixture',source_sha='a'*40,
                run_manifest_sha256=sha(root/'RUN_MANIFEST.json'),initial_state_sha256='e'*64,
                profile='TEACHER_REFERENCE',category='RELAY_QUALIFICATION',identity=identity,
                episode_uuid=str(uuid.uuid4()),deadline_epoch=time.time()+300)
            write(root/'REQUEST.json',request)
            card=dict(id='fixture',initial_state_sha256='e'*64,instruction='Fixture hold',
                relay_request_relative='REQUEST.json',relay_request_sha256=sha(root/'REQUEST.json'))
            tcp=np.eye(4);tcp[:3,:3]=np.diag([1.,-1.,-1.]);tcp[:3,3]=[.45,-.08,.41]
            plan=dict(poses=np.repeat(tcp[None],40,axis=0),opening=np.ones(40),offset_s=np.arange(40)/20,stage='hold')
            teacher=SimpleNamespace(plan=lambda state:plan,stage='hold',observe=lambda state:None)
            with patch.object(module,'validate',return_value=dict(code=dict(vla_commit='a'*40))):
                relay=module.RelayTeacher(teacher,root,folder,card)
            exchange=RelayExchange(root/'exchange',run_uuid=root.name,episode_uuid=request['episode_uuid'],identity=identity,now=time.time())
            def install(_delay):
                observation=read(folder/'relay/observations/000.json')
                query=exchange.observe(observation,time.time());body=unseal(query)
                action=sealed(dict(protocol=body['protocol'],kind='action',run_uuid=root.name,
                    episode_uuid=body['episode_uuid'],sequence=0,tick=0,
                    payload={k:body[k] for k in ('query_uuid','reference_uuid','observation_sha256','anchor_sha256','identity')}|
                    dict(trajectory_eef10=[pose10(tcp,1.).tolist()]*40,offsets_ns=[i*50000000 for i in range(40)])))
                command=exchange.action(action,time.time())
                write(folder/'relay/commands/000.json',dict(command=command,action=action,request=query))
            with patch.object(module.time,'sleep',side_effect=install):
                result=relay.plan(dict(time_s=0.,tcp=tcp,gripper_fraction=1.))
            self.assertEqual(result['poses'].shape,(40,4,4))
            self.assertTrue((folder/'relay/intents/000.json').exists())
            with self.assertRaises(ValueError):relay.plan(dict(time_s=0.,tcp=tcp,gripper_fraction=1.))
            relay.acknowledge(SimpleNamespace(steps=200),np.zeros((200,1)))
            exchange.acknowledge(read(folder/'relay/acks/000.json'),time.time())
            self.assertEqual(exchange.state['tick'],200)
            self.assertEqual(relay.acknowledged,1)
            with self.assertRaises(ValueError):relay.acknowledge(SimpleNamespace(steps=200),np.zeros((200,1)))
            relay.finish('VALID_COMPLETE',None);exchange.close()

    def test_path_escape_rejected_before_any_physics(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(module,'validate',return_value={}):
                with self.assertRaises(PermissionError):
                    module.admit_request(root,root,dict(relay_request_relative='../missing',relay_request_sha256='0'*64))


if __name__=='__main__':unittest.main()
