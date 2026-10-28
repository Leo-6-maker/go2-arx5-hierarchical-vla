"""In-memory packet fixtures, not dual-SSH or actual physics qualification."""
import base64
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
import uuid
from types import SimpleNamespace

import numpy as np
from PIL import Image

from scripts.pi0292.durable import read
from scripts.pi0292.relay_protocol import sealed,unseal,PROTOCOL
from scripts.pi0292.eval_contract import FRAME,TCP
from .relay_transport import run_episode,teacher_action


class RelayTransportTests(unittest.TestCase):
    def fixture(self,unknown=False):
        stream=io.BytesIO();Image.fromarray(np.zeros((256,256,3),dtype=np.uint8)).save(stream,format='PNG')
        raw=stream.getvalue();eef=[.45,-.08,.41,1.,0.,0.,0.,-1.,0.,1.]
        case=dict(run_uuid='fixture',episode_uuid=str(uuid.uuid4()),deadline_epoch=1000.,
            identity=dict(profile='TEACHER_REFERENCE',model_revision='NOT_APPLICABLE',source_sha='a'*40,
                config_sha256='b'*64,processor_sha256='c'*64,statistics_sha256='d'*64))
        clock=[1.];events=[];packets={};acks={};installs=[]
        def now():clock[0]+=.01;return clock[0]
        def observation(case,seq):
            if seq and seq-1 not in acks:raise AssertionError('Fresh observation before previous physical ACK')
            events.append(('obs',seq))
            packet=sealed(dict(protocol=PROTOCOL,kind='observation',run_uuid='fixture',episode_uuid=case['episode_uuid'],
                sequence=seq,tick=seq*200,payload=dict(snapshot_uuid=str(uuid.uuid4()),language='Fixture hold',
                    state_eef10=eef,frame=FRAME,tcp=TCP,images=[dict(camera=c,png_base64=base64.b64encode(raw).decode(),
                        sha256=hashlib.sha256(raw).hexdigest()) for c in ('external_rgb','wrist_rgb')])))
            packets[seq]=packet;return packet
        def teacher(case,seq):return dict(observation_sha256=packets[seq]['sha256'],trajectory_eef10=[eef]*40,
            source='ACTUAL_CURRENT_FEEDBACK_TEACHER_NOT_MODEL')
        def install(case,seq,bundle):
            installs.append(seq);events.append(('install',seq))
            if unknown:raise ConnectionError('Unknown physical dispatch outcome')
            cmd=unseal(bundle['command'])
            acks[seq]=sealed(dict(protocol=PROTOCOL,kind='ack',run_uuid='fixture',episode_uuid=case['episode_uuid'],
                sequence=seq,tick=seq*200,payload={k:cmd[k] for k in ('query_uuid','reference_uuid','prefix_sha256')}|
                dict(command_sha256=bundle['command']['sha256'],tick_after=(seq+1)*200,integrations=200,physics_receipt_sha256='e'*64)))
        core=SimpleNamespace(observation=observation,teacher=teacher,install=install,ack=lambda c,s:acks.get(s))
        hopper=SimpleNamespace(query=lambda c,q,t:teacher_action(q,t),response=lambda c,q:None)
        return case,core,hopper,now,events,installs

    def test_all_eighty_fresh_observations_follow_previous_ack(self):
        case,core,hopper,now,events,installs=self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            result=run_episode(directory,case,core,hopper,now=now)
            self.assertEqual(result['acknowledgements'],80);self.assertEqual(installs,list(range(80)))
            self.assertEqual(read(Path(directory)/'STATE.json')['tick'],16000)
            self.assertEqual(result['model_calls_local'],0);self.assertEqual(result['physics_integrations_local'],0)

    def test_unknown_install_is_not_replayed_or_called_success(self):
        case,core,hopper,now,events,installs=self.fixture(unknown=True)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ConnectionError):run_episode(directory,case,core,hopper,now=now)
            self.assertEqual(installs,[0]);self.assertEqual(events,[('obs',0),('install',0)])
            state=read(Path(directory)/'ROUTE.json')
            self.assertTrue(state['network_outcome_unknown']);self.assertEqual(state['acknowledgements'],0)
            self.assertEqual(state['status'],'STOPPED_PRESERVED_NO_AUTOMATIC_RETRY')

    def test_versioned_read_polling_keeps_eighty_one_shot_queries_and_fresh_ACK_chain(self):
        case,core,hopper,now,events,installs=self.fixture();sleeps=[];pending={};queries=[]
        observation=core.observation;ack=core.ack;first_observation=[True];ack_reads={};responses={}
        def observe(c,s):
            if first_observation[0]:first_observation[0]=False;return None
            return observation(c,s)
        def query(c,q,t):
            key=unseal(q)['query_uuid'];queries.append(key);pending[key]=teacher_action(q,t);return None
        def response(c,key):
            responses[key]=responses.get(key,0)+1
            return None if responses[key]==1 else pending[key]
        def read_ack(c,s):
            ack_reads[s]=ack_reads.get(s,0)+1
            return None if ack_reads[s]==1 else ack(c,s)
        core.observation=observe;core.ack=read_ack;hopper.query=query;hopper.response=response
        with tempfile.TemporaryDirectory() as directory:
            result=run_episode(directory,case,core,hopper,now=now,sleep=sleeps.append,poll_interval=.25)
            self.assertEqual(result['acknowledgements'],80);self.assertEqual(installs,list(range(80)))
            self.assertEqual(len(queries),80);self.assertEqual(len(set(queries)),80)
            self.assertEqual(sleeps,[.25]*161)

    def test_arbitrary_poll_mode_is_rejected_before_any_dispatch(self):
        case,core,hopper,now,events,installs=self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            for interval in (0,.01,.5,True,float('nan')):
                with self.subTest(interval=interval),self.assertRaises(PermissionError):
                    run_episode(directory,case,core,hopper,now=now,poll_interval=interval)
        self.assertEqual(installs,[]);self.assertEqual(events,[])

    def test_teacher_cannot_supply_model_or_stale_observation_answer(self):
        case,core,hopper,now,events,installs=self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            from scripts.pi0292.relay_protocol import RelayExchange
            exchange=RelayExchange(directory,run_uuid='fixture',episode_uuid=case['episode_uuid'],identity=case['identity'],now=now())
            try:
                query=exchange.observe(core.observation(case,0),now());teacher=core.teacher(case,0)
                with self.assertRaises(ValueError):teacher_action(query,dict(teacher,observation_sha256='f'*64))
                value=unseal(query);value['identity']['profile']='B1'
                with self.assertRaises(PermissionError):teacher_action(sealed(value),teacher)
            finally:exchange.close()


if __name__=='__main__':unittest.main()
