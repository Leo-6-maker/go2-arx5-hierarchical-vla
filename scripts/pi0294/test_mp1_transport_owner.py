from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .mp1 import transport_fixture_owner as owner


class FakeSession:
    def __init__(self,role,label):self.role=role;self.label=label;self.closed=0;self.failed=False;self.sent=0
    def call(self,args):
        if self.failed:raise RuntimeError('POISONED_FILE_RPC_SESSION_NO_NEW_REQUEST_OR_REPLAY')
        self.sent+=1;self.failed=True;raise TimeoutError('ISOLATED_FIXTURE')
    def close(self):self.closed+=1


class OwnerFixturePhaseTests(unittest.TestCase):
    def setUp(self):
        label='N1-PREMODEL-ALLOCATION-06';self.registry=dict(label=label,run_uuid='source-pair-01',job_id='123.hopper-m-02',hopper_source_sha='a'*40)
        self.sessions={r:FakeSession(r,label) for r in ['core','hopper']}
        self.faults={name:FakeSession('hopper',label.replace('N1-PREMODEL','N1-PREMODEL-FAULT-'+suffix))
            for name,suffix in [('late_response','LATE'),('process_exit','EXIT')]}
        self.native=lambda name,session:dict(native_process_terminal_verified=True,actual_fault_requests=1,
            actual_install_count=80,actual_install_ACKs_sha256='a'*64,job_id=self.registry['job_id'],label=session.label,
            source_sha='a'*40,native_fault_intent_sha256='f'*64,actual_server_exit=72 if name=='process_exit' else 0)

    def test_one_owner_phase_never_resends_failed_fault_and_closes_all_own_channels(self):
        published=[];clock=iter([0.,20.25,30.,30.01])
        with patch.object(owner,'route',return_value=[dict(sequence=i) for i in range(80)]),patch.object(owner.time,'monotonic',side_effect=lambda:next(clock)):
            value=owner.drive(self.sessions,self.faults,{},self.registry,lambda row:None,self.native,self.native,published.append,
                lambda:None,registry_sha256='b'*64)
        self.assertEqual(published,[value]);self.assertFalse(value['heartbeat_writer'])
        self.assertTrue(all(s.sent==1 and s.closed==1 for s in self.faults.values()))
        self.assertTrue(all(s.closed==1 for s in self.sessions.values()))
        self.assertEqual(value['nominal_cycles'],80)

    def test_unknown_route_effect_is_not_retried_and_still_closes_fixture_channels(self):
        calls=[]
        def fail(*args,**kwargs):calls.append(True);raise ConnectionError('UNKNOWN_FIXTURE_EFFECT_NO_REPLAY')
        with patch.object(owner,'route',side_effect=fail),self.assertRaises(ConnectionError):
            owner.drive(self.sessions,self.faults,{},self.registry,lambda row:None,self.native,self.native,lambda value:None,
                lambda:None,registry_sha256='b'*64)
        self.assertEqual(len(calls),1);self.assertTrue(all(s.closed==1 for s in [*self.sessions.values(),*self.faults.values()]))

    def test_pv1_reuses_identical_wire_driver_with_exact_versioned_reference(self):
        self.registry['label']=self.registry['label'].replace('N1-PREMODEL','PV1-PREMODEL')
        for s in [*self.sessions.values(),*self.faults.values()]:s.label=s.label.replace('N1-PREMODEL','PV1-PREMODEL')
        clock=iter([0.,20.25,30.,30.01])
        with patch.object(owner,'route',return_value=[dict(sequence=i) for i in range(80)]),\
                patch.object(owner.time,'monotonic',side_effect=lambda:next(clock)):
            value=owner.drive(self.sessions,self.faults,{},self.registry,lambda row:None,self.native,self.native,
                lambda value:None,lambda:None,registry_sha256='b'*64,fixture_base='PV1-PREMODEL',reference_proof_sha256='c'*64)
        self.assertEqual(value['schema'],'PI0294_PV1_OWNER_TRANSPORT_FAULTS_V1')
        self.assertEqual(value['reference_proof_sha256'],'c'*64)
        self.assertTrue(all(s.sent==1 and s.closed==1 for s in self.faults.values()))

    def test_pv1_cannot_claim_reference_or_namespace_by_implicit_legacy_mode(self):
        with self.assertRaisesRegex(PermissionError,'REFERENCE_BINDING'):
            owner.drive(self.sessions,self.faults,{},self.registry,lambda row:None,self.native,self.native,
                lambda value:self.fail('NO_PUBLISH'),lambda:None,registry_sha256='b'*64,fixture_base='PV1-PREMODEL')
        self.assertTrue(all(s.sent==0 and s.closed==1 for s in [*self.sessions.values(),*self.faults.values()]))

    def test_alive_remote_fault_process_cannot_be_claimed_qualified(self):
        clock=iter([0.,20.25])
        with patch.object(owner,'route',return_value=[dict(sequence=i) for i in range(80)]),patch.object(owner.time,'monotonic',side_effect=lambda:next(clock)),self.assertRaises(PermissionError):
            owner.drive(self.sessions,self.faults,{},self.registry,lambda row:None,
                lambda *args:dict(native_process_terminal_verified=False,actual_fault_requests=1),self.native,
                lambda value:self.fail('MUST_NOT_PUBLISH'),lambda:None,registry_sha256='b'*64)

    def test_wrong_role_or_allocation_is_rejected_without_sending(self):
        self.sessions['core'].role='hopper'
        with self.assertRaises(PermissionError):owner.drive(self.sessions,self.faults,{},self.registry,lambda row:None,
            self.native,self.native,lambda value:None,lambda:None,registry_sha256='b'*64)
        self.assertTrue(all(s.sent==0 for s in [*self.sessions.values(),*self.faults.values()]))

    def test_logged_call_reads_exactly_one_new_mature_client_record(self):
        with tempfile.TemporaryDirectory() as folder:
            class Session:
                metrics=Path(folder)/'wire.jsonl'
                def call(self,args):
                    with self.metrics.open('a') as stream:stream.write(json.dumps(dict(operation=args['operation']))+'\n')
                    return dict(ok=True,rpc_id='fixture')
            s=Session();s.metrics.write_text('{}\n')
            value=owner.logged_call(s,dict(operation='observation'))
            self.assertEqual(value['wire'],dict(operation='observation'));self.assertEqual(value['packet']['rpc_id'],'fixture')


if __name__=='__main__':unittest.main()
