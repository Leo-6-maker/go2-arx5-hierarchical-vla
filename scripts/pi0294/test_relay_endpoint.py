"""Endpoint file fixtures, not SSH or physical qualification."""
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts.pi0292.durable import read, write
from scripts.pi0292.relay_protocol import RelayExchange, unseal
from .native_dataset import sha
from . import relay_endpoint as endpoint_module
from . import test_relay_transport as fixtures

Endpoint, inside = endpoint_module.Endpoint, endpoint_module.inside


class EndpointTests(unittest.TestCase):
    def prepare(self, root):
        case, core, _, _, _, _ = fixtures.RelayTransportTests().fixture()
        case['run_uuid'] = root.name
        # The existing packet fixture's fixed run name is retained here.
        root = root/'fixture';root.mkdir()
        case['run_uuid'] = root.name
        episode = case['episode_uuid']
        folder = root/'episodes'/'attempt'/'relay'
        for name in ('observations','commands','acks','intents'):
            (folder/name).mkdir(parents=True)
        for name in ('inbox','outbox'):
            (root/'session'/name).mkdir(parents=True)
        write(root/'RUN_MANIFEST.json', {})
        request = dict(episode_uuid=episode,identity=case['identity'],attempt_id='attempt',
            source_sha='a'*40,run_manifest_sha256=sha(root/'RUN_MANIFEST.json'))
        write(root/'request.json',request)
        registered = dict(episode_uuid=episode,identity=case['identity'],attempt_id='attempt',
            core_request_relative='request.json',core_request_sha256=sha(root/'request.json'),
            hopper_session_relative='session')
        registry = dict(run_uuid=root.name,core_source_sha='a'*40,hopper_source_sha='a'*40,
            core_manifest_sha256=sha(root/'RUN_MANIFEST.json'),hopper_manifest_sha256=sha(root/'RUN_MANIFEST.json'),
            cases={episode:registered},deadline_epoch=time.time()+900)
        write(root/'registry.json',registry)
        with patch.object(endpoint_module,'validate',return_value=dict(code=dict(vla_commit='a'*40))):
            endpoints = [Endpoint(root,'registry.json',sha(root/'registry.json'),role) for role in ('core','hopper')]
        packet = core.observation(case,0);write(folder/'observations/000.json',packet)
        teacher = core.teacher(case,0);write(folder/'teacher-000.json',teacher)
        return root,case,folder,packet,teacher,endpoints

    def test_teacher_cross_endpoint_and_one_shot_install(self):
        with tempfile.TemporaryDirectory() as directory:
            root,case,folder,packet,teacher,(core,hopper)=self.prepare(Path(directory))
            exchange=RelayExchange(root/'exchange',run_uuid=root.name,episode_uuid=case['episode_uuid'],identity=case['identity'],now=time.time())
            try:
                self.assertEqual(core.core('observation',case['episode_uuid'],0),packet)
                query=exchange.observe(packet,time.time())
                action=hopper.hopper('query',case['episode_uuid'],query=query,teacher=teacher)
                command=exchange.action(action,time.time())
                result=core.core('install',case['episode_uuid'],0,dict(command=command,action=action,request=query))
                self.assertEqual(result['status'],'BUNDLE_INSTALLED_NOT_PHYSICAL_ACK')
                self.assertIsNone(core.core('ack',case['episode_uuid'],0))
                self.assertEqual(hopper.hopper('response',case['episode_uuid'],query_uuid=unseal(query)['query_uuid']),action)
                with self.assertRaises(FileExistsError):
                    core.core('install',case['episode_uuid'],0,dict(command=command,action=action,request=query))
                with self.assertRaises(FileExistsError):
                    hopper.hopper('query',case['episode_uuid'],query=query,teacher=teacher)
            finally:exchange.close()

    def test_failed_teacher_intent_preserved_without_resampling(self):
        with tempfile.TemporaryDirectory() as directory:
            root,case,folder,packet,teacher,(_,hopper)=self.prepare(Path(directory))
            exchange=RelayExchange(root/'exchange',run_uuid=root.name,episode_uuid=case['episode_uuid'],identity=case['identity'],now=time.time())
            try:
                query=exchange.observe(packet,time.time())
                with self.assertRaises(ValueError):
                    hopper.hopper('query',case['episode_uuid'],query=query,teacher=dict(teacher,observation_sha256='f'*64))
                self.assertIsNotNone(read(root/'session/QUERY_STATE.json')['pending'])
                with self.assertRaises(FileExistsError):
                    hopper.hopper('query',case['episode_uuid'],query=query,teacher=teacher)
            finally:exchange.close()

    def test_escape_and_changed_source_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for relative in ('../outside','/tmp/outside'):
                with self.assertRaises(PermissionError):inside(root,relative)
            (root/'escape').symlink_to('/tmp')
            with self.assertRaises(PermissionError):inside(root,'escape/member')
            write(root/'RUN_MANIFEST.json',{})
            write(root/'registry.json',dict(run_uuid=root.name,core_source_sha='a'*40,
                core_manifest_sha256=sha(root/'RUN_MANIFEST.json'),deadline_epoch=time.time()+60))
            with patch.object(endpoint_module,'validate',return_value=dict(code=dict(vla_commit='b'*40))):
                with self.assertRaises(PermissionError):Endpoint(root,'registry.json',sha(root/'registry.json'),'core')


if __name__=='__main__':unittest.main()
