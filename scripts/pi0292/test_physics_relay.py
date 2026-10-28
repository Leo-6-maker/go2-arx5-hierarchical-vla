import copy
import tempfile
from pathlib import Path
import unittest
import uuid

from .physics_relay import accept_prefix,require_s6_admission
from .relay_protocol import PROTOCOL,RelayExchange,sealed
from . import test_live_inputs as packet_fixtures


class TestPhysicalPrefixContractWithoutPhysics(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.identity=dict(profile='B1',model_revision='1'*40,source_sha='2'*40,
            config_sha256='3'*64,processor_sha256='4'*64,statistics_sha256='5'*64)
        payload=packet_fixtures.TestObservationOnlyArrays().payload()
        self.observation=sealed(dict(protocol=PROTOCOL,kind='observation',run_uuid='run',episode_uuid=str(uuid.uuid4()),
            sequence=0,tick=0,payload=payload))
        self.exchange=RelayExchange(self.root,run_uuid='run',episode_uuid=self.observation['body']['episode_uuid'],identity=self.identity,now=100.)
        self.request=self.exchange.observe(self.observation,100.);request=self.request['body']
        trajectory=[list(payload['state_eef10']) for _ in range(40)]
        self.action=sealed(dict(protocol=PROTOCOL,kind='action',run_uuid='run',episode_uuid=request['episode_uuid'],sequence=0,tick=0,
            payload={key:request[key] for key in ('query_uuid','reference_uuid','observation_sha256','anchor_sha256','identity')}|
                dict(trajectory_eef10=trajectory,offsets_ns=[i*50000000 for i in range(40)])))
        self.command=self.exchange.action(self.action,101.)

    def tearDown(self):self.exchange.close();self.temp.cleanup()

    def check(self,command=None,action=None,now=102.):
        return accept_prefix(command or self.command,action or self.action,self.observation,
            request=self.request,identity=self.identity,now=now)

    def test_full_gate_then_exact_prefix200_not_physics_evidence(self):
        command,gate=self.check();self.assertEqual(command['advance_integrations'],200)
        self.assertEqual(len(command['prefix_eef10']),8)
        self.assertIn(gate['decision'],('ACCEPT','WARN'))

    def test_wrong_identity_expiry_clock_count_and_prefix_rejected(self):
        for key,value in [('tick',200),('sequence',True),('expires',161.),('advance_integrations',201),
                          ('prefix_sha256','0'*64),('anchor_sha256','0'*64),('frame','other')]:
            with self.subTest(key=key):
                body=copy.deepcopy(self.command['body']);body[key]=value
                with self.assertRaises(ValueError):self.check(sealed(body))
        with self.assertRaises(ValueError):self.check(now=160.)

    def test_h40_tail_still_enters_gate_not_only_installed_prefix(self):
        body=copy.deepcopy(self.action['body']);body['payload']['trajectory_eef10'][30][0]+=1.
        with self.assertRaisesRegex(ValueError,'SAFETY_GATE_REJECT'):self.check(action=sealed(body))
        body=copy.deepcopy(self.action['body']);body['payload']['offsets_ns'][0]=False
        with self.assertRaises(ValueError):self.check(action=sealed(body))

    def test_no_s6_physics_before_actual_main_and_offline(self):
        case=dict(profile='TEACHER_REFERENCE',kind='RELAY_QUALIFICATION',layout_id='T291000')
        value=dict(status='ALLOW_ORIGINAL_S6_GATED_PHASE',run_uuid='run',release='261bc5b37f20aaff1ad7d5742252d3dc1a91fa65',
            original_T0=1791860304,original_deadline=1791946704,main_status='MAIN_COMPLETED_FIXED_S4096',
            offline_status='PAIRED_OFFLINE_RECORDS_ANALYZED_NOT_TASK_SUCCESS',authority_kind='ACTUAL_HOPPER_RECEIPTS_READBACK_BY_EXISTING_LOCAL_PC')
        require_s6_admission(value,run='run',case=case)
        for field,change in [('main_status','NOT_RUN'),('offline_status','NOT_RUN'),('original_T0',1),('authority_kind','CPU_FIXTURE')]:
            with self.subTest(field=field):
                bad=copy.deepcopy(value);bad[field]=change
                with self.assertRaises(PermissionError):require_s6_admission(bad,run='run',case=case)
