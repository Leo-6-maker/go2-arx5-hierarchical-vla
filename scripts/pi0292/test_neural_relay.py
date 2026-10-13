import copy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

from .neural_relay import QueryReservations,live_seed,require_case_order,require_case_phase,require_teacher_proof,run,validate_request
from .relay_protocol import PROTOCOL,digest,sealed
from . import test_live_inputs as packet_fixtures


class TestNeuralRelayCPUOnly(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.episode=str(uuid.uuid4());self.run='existing-science-run'
        self.identity=dict(profile='B1',model_revision='1'*40,source_sha='2'*40,
            config_sha256='3'*64,processor_sha256='4'*64,statistics_sha256='5'*64)
        self.cases={self.episode:dict(profile='B1',kind='TRAIN_CANARY',layout_id='T291000')}
        payload=packet_fixtures.TestObservationOnlyArrays().payload()
        body=dict(protocol=PROTOCOL,kind='student_request',run_uuid=self.run,episode_uuid=self.episode,
            sequence=0,tick=0,query_uuid=str(uuid.uuid4()),reference_uuid=str(uuid.uuid4()),expires=160.,
            observation_sha256=sealed(dict(protocol=PROTOCOL,kind='observation',run_uuid=self.run,
                episode_uuid=self.episode,sequence=0,tick=0,payload=payload))['sha256'],
            anchor_sha256=digest(payload['state_eef10']),identity=self.identity,payload=payload)
        body['previous_ack']=None
        self.packet=sealed(body)

    def tearDown(self):self.temp.cleanup()

    def inspect(self,packet=None,now=100.):
        return validate_request(packet or self.packet,run=self.run,cases=self.cases,identities={'B1':self.identity},now=now)

    def test_registered_fresh_observation_and_noise_pairing(self):
        body,case=self.inspect();self.assertEqual(case,self.cases[self.episode])
        self.assertEqual(body['sequence'],0)
        self.assertEqual(live_seed('E292000',3),live_seed('E292000',3))
        self.assertNotEqual(live_seed('E292000',3),live_seed('E292000',4))

    def test_wrong_frame_identity_anchor_expiry_tick_and_privilege_rejected(self):
        for mutation in ('tick','booltick','anchor','hash','expiry','identity','future','episode'):
            with self.subTest(mutation=mutation):
                body=copy.deepcopy(self.packet['body'])
                if mutation=='tick':body['tick']=200
                if mutation=='booltick':body['tick']=False
                if mutation=='anchor':body['anchor_sha256']='0'*64
                if mutation=='hash':body['observation_sha256']='0'*64
                if mutation=='expiry':body['expires']=161.
                if mutation=='identity':body['identity']['model_revision']='0'*40
                if mutation=='future':body['payload']['action']=[1]
                if mutation=='episode':body['episode_uuid']=str(uuid.uuid4())
                with self.assertRaises(ValueError):self.inspect(sealed(body))
        with self.assertRaises(ValueError):self.inspect(now=160.)

    def test_intent_duplicate_reorder_lost_response_and_restart_refused(self):
        body,_=self.inspect();query=QueryReservations(self.root,self.run)
        query.begin(body,100.)
        with self.assertRaises(ValueError):query.begin(body,101.)
        query.complete(body,'a'*64,prefix_sha256='b'*64,command_sha256='c'*64)
        with self.assertRaises(ValueError):query.begin(body,102.)
        changed=copy.deepcopy(body);changed.update(sequence=1,tick=200,query_uuid=str(uuid.uuid4()))
        with self.assertRaises(ValueError):query.begin(changed,102.)
        changed['payload']['snapshot_uuid']=str(uuid.uuid4())
        changed['previous_ack']=sealed(dict(protocol=PROTOCOL,kind='ack',run_uuid=self.run,episode_uuid=self.episode,
            sequence=0,tick=0,payload=dict(query_uuid=body['query_uuid'],reference_uuid=body['reference_uuid'],
                command_sha256='c'*64,prefix_sha256='b'*64,tick_after=200,integrations=200,physics_receipt_sha256='d'*64)))
        query.begin(changed,102.)
        query.fail(changed,'CPU fixture failure',known_terminal=True)
        self.assertIsNone(query.state['pending'])
        changed.update(sequence=2,tick=400,query_uuid=str(uuid.uuid4()))
        changed['payload']['snapshot_uuid']=str(uuid.uuid4())
        with self.assertRaises(ValueError):query.begin(changed,103.)
        with self.assertRaises(FileExistsError):QueryReservations(self.root,self.run)

    def test_confirmation_requires_real_safe_canary_and_teacher_gates(self):
        case=dict(profile='B0',kind='CONFIRM',layout_id='E292000')
        with self.assertRaises(PermissionError):require_case_phase(case,{})
        row=dict(attempted=True,persistent_success=True,safety_valid=True,
            proof_kind='REAL_CORELAB_ATTEMPT_NOT_PACKET_FIXTURE',result_sha256='a'*64,score_sha256='b'*64,
            physics_steps=16000,paired_samples=641,acknowledged_commands=80)
        proofs=dict(B1_canaries=[dict(row,layout_id=name) for name in ('T291000','T291001')],
            confirm_teacher=[dict(row,layout_id=f'E292{i:03d}') for i in range(8)])
        require_case_phase(case,proofs)
        for mutation in ('six-teachers','no-canary-success','unsafe','fixture','stopped-arm','short-teacher'):
            with self.subTest(mutation=mutation):
                bad=copy.deepcopy(proofs)
                if mutation=='six-teachers':
                    for item in bad['confirm_teacher'][:2]:item['persistent_success']=False
                if mutation=='no-canary-success':
                    for item in bad['B1_canaries']:item['persistent_success']=False
                if mutation=='unsafe':bad['B1_canaries'][0]['safety_valid']=False
                if mutation=='fixture':bad['confirm_teacher'][0]['proof_kind']='CPU_FIXTURE'
                if mutation=='stopped-arm':bad['stopped_models']=['B0']
                if mutation=='short-teacher':bad['confirm_teacher'][0]['physics_steps']=200
                with self.assertRaises(PermissionError):require_case_phase(case,bad)

    def test_teacher_positive_metadata_is_not_packet_fixture_acceptance(self):
        row=dict(physics_steps=16000,paired_samples=641,acknowledged_commands=80,
            persistent_success=True,safety_valid=True,physics_sha256='a'*64,score_sha256='b'*64)
        value=dict(status='PASS_ACTUAL_CORELAB_TEACHER_REFERENCE_RELAY_2_OF_2',run_uuid=self.run,
            physics_source_sha='35290803cd82e10a164daaa53af5743dd43054ee',
            proof_kind='REAL_PHYSICS_AND_INDEPENDENT_SCORE_NOT_PACKET_FIXTURE',
            records=[dict(row,layout_id=name) for name in ('T291000','T291001')])
        require_teacher_proof(value,self.run)
        value['proof_kind']='CPU_PACKET_FIXTURE'
        with self.assertRaises(PermissionError):require_teacher_proof(value,self.run)

    def test_missing_pbs_denied_before_any_model_or_file_query(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(PermissionError,'NO_LIVE_MODELS'):
                run(SimpleNamespace(root=str(self.root),bindings='absent',request='absent'))

    def test_case_order_and_actual_prior_terminal_receipt_required(self):
        first=dict(episode_uuid='first',schedule_index=0,profile='B1')
        second=dict(episode_uuid='second',schedule_index=1,profile='B1')
        cases={'first':first,'second':second};state=dict(episodes={})
        with self.assertRaises(PermissionError):require_case_order(second,cases=cases,query_state=state,proofs={})
        require_case_order(first,cases=cases,query_state=state,proofs={})
        state['episodes']['first']=dict(next_sequence=1)
        with self.assertRaises(PermissionError):require_case_order(second,cases=cases,query_state=state,proofs={})
        require_case_order(second,cases=cases,query_state=state,
            proofs=dict(finished_episodes={'first':dict(proof_kind='REAL_CORELAB_ATTEMPT_NOT_PACKET_FIXTURE')}))
