from pathlib import Path
import hashlib
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.pi0292.durable import write
from scripts.pi0292.relay_protocol import digest, sealed
from .mp1 import effect_closure as closure
from .mp1 import fixed_probes as probes
from .native_dataset import sha


class EffectClosure(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.folder=self.root/'evidence/test/resident'
        for kind in ['inbox','outbox','receipts']:(self.folder/kind).mkdir(parents=True)
        self.query=sealed(dict(query_uuid='query',episode_uuid='episode',sequence=0,
            observation_sha256='observed',payload={'state':1},previous_ack=None,identity={'model':'frozen'}))
        self.action=sealed(dict(sequence=0,payload={'query_uuid':'query','trajectory_eef10':[[1.0]*10]*40}))
        self.core=dict(rows=[dict(attempt_id='attempt',status='VALID_TASK_FAIL',S1_rejections=1,main_integrations=0)],
            installed={},observations={'episode/0':dict(attempt_id='attempt',observation_sha256='observed',
                payload_sha256=digest({'state':1}),previous_ACK_sha256=None)})
        self.core['rejected_installs']={'query':dict(request_sha256=self.query['sha256'],action_sha256=self.action['sha256'])}
        self.ledger=dict(GPU_reserved_s=0,pending_updates=0,CPU_reserved_s=0,
            PBS_submissions=[dict(status='NATIVE_TERMINAL')],generation_queries=433)
        self.draw=dict(query_uuid='query',episode_uuid='episode',sequence=0,response_sha256=self.action['sha256'])
        write(self.root/'HOPPER_RESOURCE_LEDGER.json',self.ledger)
        write(self.folder/'QUERY_STATE.json',dict(queries=[self.draw],pending=None))
        write(self.folder/'STATE.json',dict(queries=1,identity={'relay_identity':{'model':'frozen'}}))
        write(self.folder/'inbox/query.json',self.query);write(self.folder/'outbox/query.json',self.action)
        self.array=self.folder/'receipts/query.npz';np.savez(self.array,physical=np.ones((40,10)))
        write(self.folder/'receipts/query.json',dict(action_sha256=self.action['sha256'],arrays_sha256=sha(self.array)))
        for branch in ['S-BC','S-CORR']:
            for step in [512,2048]:
                path=self.root/'training'/branch/f'checkpoint-{step}'
                path.mkdir(parents=True)
                write(path/'CHECKPOINT_COMPLETE.json',dict(files={'config.json':'a','optimizer.pt':'not-inference'}))
        self.probe=patch.object(probes,'status',return_value=dict(complete=True,unknown_effects=0,
            new_queries_required=0,completed_draw_ids=list(range(108))))
        self.probe_mock=self.probe.start();self.addCleanup(self.probe.stop)

    def test_rejected_prefix_is_classified_not_executed_or_replayed(self):
        value=closure.hopper(self.root,self.core)
        self.assertEqual(value['cumulative_generation_queries'],433)
        self.assertEqual(value['classified'][0]['classification'],'ORIGINAL_S1_REJECTED_INSTALLED_ZERO_LIVE_PREFIX')
        self.assertEqual(value['actual_executed_prefixes'],0)
        self.assertEqual(self.probe_mock.call_args.args[2],{'config.json':'a'})

    def test_installed_prefix_must_join_exact_native_bytes(self):
        self.core['rejected_installs']={}
        self.core['installed']['query']=dict(request_sha256=self.query['sha256'],action_sha256=self.action['sha256'])
        value=closure.hopper(self.root,self.core)
        self.assertEqual(value['classified'][0]['classification'],'EXECUTED_ACKED_PREFIX')
        self.core['installed']['query']['action_sha256']='different'
        with self.assertRaises(PermissionError):closure.hopper(self.root,self.core)

    def test_inbox_only_terminal_orphan_is_not_an_issued_model_query(self):
        for kind in ['outbox','receipts']:
            for path in (self.folder/kind).iterdir():path.unlink()
        self.core['rows'][0]['status']='ENGINEERING_INCOMPLETE'
        self.core['rejected_installs']={}
        write(self.folder/'QUERY_STATE.json',dict(queries=[],pending=None))
        write(self.folder/'STATE.json',dict(queries=0,identity={'relay_identity':{'model':'frozen'}}))
        write(self.root/'HOPPER_RESOURCE_LEDGER.json',dict(self.ledger,generation_queries=432))
        value=closure.hopper(self.root,self.core)
        self.assertEqual(value['resident_generations'],0)
        self.assertEqual(value['classified'][0]['classification'],'TERMINAL_INBOX_ONLY_NO_GENERATION_NO_INSTALL')
        self.core['rows'][0]['status']='SUCCESS'
        with self.assertRaises(PermissionError):closure.hopper(self.root,self.core)

    def test_unknown_intents_incomplete_probes_or_corrupt_arrays_are_not_closed(self):
        write(self.folder/'QUERY_STATE.json',dict(queries=[self.draw],pending={'query_uuid':'query'}))
        with self.assertRaises(PermissionError):closure.hopper(self.root,self.core)
        write(self.folder/'QUERY_STATE.json',dict(queries=[self.draw],pending=None))
        self.probe_mock.return_value=dict(complete=False,unknown_effects=0,new_queries_required=1)
        with self.assertRaises(PermissionError):closure.hopper(self.root,self.core)
        self.array.write_bytes(b'corrupt')
        with self.assertRaises(PermissionError):closure.hopper(self.root,self.core)

    def test_core_closes_only_actual_command_ack_physics_and_saved_camera_chain(self):
        root=self.root/'core';episode=root/'episodes/attempt';relay=episode/'relay'
        for kind in ['observations','commands','acks','intents']:(relay/kind).mkdir(parents=True)
        for camera in ['body_rgb','wrist_rgb']:
            (episode/camera).mkdir();(episode/camera/'0000.png').write_bytes(b'CPU_SYNTHETIC_CAMERA_FIXTURE')
        write(root/'RESOURCE_LEDGER.json',dict(reservations={},CPU_reserved_s=0))
        (root/'worker_processes').mkdir()
        write(root/'worker_processes/attempt.json',dict(pid=2**30,proc_start='1'))
        write(episode/'MP1_ADJUDICATION.json',dict(attempt_id='attempt',main_integrations=200,
            live_native_failures=0,status='ENGINEERING_INCOMPLETE'))
        observation=sealed(dict(episode_uuid='episode',sequence=0,tick=0,payload={'images':[
            {'sha256':sha(episode/camera/'0000.png')} for camera in ['body_rgb','wrist_rgb']]}))
        write(relay/'observations/000.json',observation)
        query=sealed(dict(query_uuid='query',sequence=0));action=self.action
        command=sealed(dict(sequence=0,prefix_sha256=digest([[1.0]*10]*8)))
        (relay/'physics-000.npz').write_bytes(b'CPU_SYNTHETIC_PHYSICS_FIXTURE')
        ack=sealed(dict(payload=dict(command_sha256=command['sha256'],query_uuid='query',integrations=200,
            tick_after=200,physics_receipt_sha256=sha(relay/'physics-000.npz'))))
        write(relay/'commands/000.json',dict(request=query,action=action,command=command))
        write(relay/'intents/000.json',dict(known=True));write(relay/'acks/000.json',ack)
        value=closure.core(root)
        self.assertEqual(value['actual_executed_prefixes'],1)
        self.assertEqual(value['observations']['episode/0']['observation_sha256'],observation['sha256'])
        (relay/'acks/000.json').unlink()
        with self.assertRaisesRegex(PermissionError,'UNKNOWN_INSTALLED'):closure.core(root)

    def test_installed_rejection_needs_exact_frozen_S1_action_audit_and_zero_live_evidence(self):
        from scripts.pi0292.eval_contract import matrix10
        root=self.root/'rejected';folder=root/'episodes/attempt';relay=folder/'relay'
        for kind in ['intents','acks']:(relay/kind).mkdir(parents=True)
        values=np.tile([0.,0.,0.,1.,0.,0.,0.,1.,0.,.5],(40,1))
        query=sealed(dict(query_uuid='query',sequence=0))
        action=sealed(dict(sequence=0,payload={'trajectory_eef10':values.tolist()}))
        command=sealed(dict(sequence=0,prefix_sha256=digest(values[:8].tolist())))
        bundle=dict(request=query,action=action,command=command)
        row=dict(status='VALID_TASK_FAIL',failure_stage='S1_PRE_EXECUTION_REJECT',
            pipeline_trustworthy=True,S1_rejections=1,main_integrations=0,live_native_failures=0)
        intent=dict(tick_before=0,query_uuid='query',command_sha256=command['sha256'],requested_integrations=200)
        forecast=dict(decision='REJECT',live_integrations=0,live_state_unchanged=True,actions_modified=False,
            sequence=0,action_sha256=hashlib.sha256(matrix10(values)[:8].tobytes()+values[:8,9].tobytes()).hexdigest())
        audit=dict(status='PASS_INDEPENDENT_ACTUAL_EVIDENCE_AUDIT',adjudication=row,actual_native_safety_flag_rows=0)
        audit_path=root/'evidence/audits/attempt/AUDIT.private.json';audit_path.parent.mkdir(parents=True)
        write(relay/'intents/000.json',intent);write(folder/'S1_PREFLIGHTS.json',[forecast])
        write(folder/'S1_RESULT.json',dict(pre_execution_rejections=1,live_native_failures=0))
        write(folder/'WORKER_RECEIPT.json',dict(physics_steps=0));write(audit_path,audit)
        result=closure.rejected_prefix(folder,row,'000.json',bundle)
        self.assertEqual(result['classification'],'ORIGINAL_S1_REJECTED_INSTALLED_ZERO_LIVE_PREFIX')
        self.assertEqual(result['live_integrations'],0)
        self.assertFalse((relay/'acks/000.json').exists())
        for change in [dict(forecast,live_integrations=1),dict(forecast,action_sha256='different'),
                       dict(forecast,live_state_unchanged=False)]:
            write(folder/'S1_PREFLIGHTS.json',[change])
            with self.assertRaises(PermissionError):closure.rejected_prefix(folder,row,'000.json',bundle)
        write(folder/'S1_PREFLIGHTS.json',[forecast]);(relay/'physics-000.npz').write_bytes(b'nonzero-effect')
        with self.assertRaises(PermissionError):closure.rejected_prefix(folder,row,'000.json',bundle)


if __name__=='__main__':unittest.main()
