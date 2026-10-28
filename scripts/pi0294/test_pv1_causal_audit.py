"""Synthetic CPU packet fixtures, never actual native/model/physics evidence."""
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.pi0292.durable import read
from scripts.pi0292.eval_contract import gate_action
from scripts.pi0292.relay_protocol import digest,sealed
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.pv1 import causal_audit as audit
from scripts.pi0294.test_mp1_transport_fixture import put as write


class CausalAudit(unittest.TestCase):
    def test_cross_platform_metric_ulps_do_not_change_gate_semantics_or_thresholds(self):
        original=dict(decision='REJECT',issues=[dict(code='LINEAR_SPEED_REJECT',severity='REJECT',observed=2.102065784832548,threshold=2.)],
            metrics=dict(chunk_id='fixed',angular_speed_radps=dict(median=.09643620197621688),
                initial_rotation_discontinuity_rad=.0045055063981939755),provenance=dict(corrective_action='NONE'))
        recomputed=deepcopy(original);recomputed['metrics']['angular_speed_radps']['median']=.0964362019762169
        recomputed['metrics']['initial_rotation_discontinuity_rad']=.004505506398193975
        self.assertTrue(audit.same_gate(original,recomputed))
        for changed in (dict(recomputed,decision='PASS'),dict(recomputed,provenance={}),
                dict(recomputed,metrics=dict(chunk_id='OTHER')),
                dict(recomputed,issues=[dict(original['issues'][0],threshold=2.0000000000001)]),
                dict(recomputed,issues=[dict(original['issues'][0],observed=2.1)]),
                dict(recomputed,metrics=dict(chunk_id='fixed',bad=float('nan')))):
            with self.subTest(changed=changed):self.assertFalse(audit.same_gate(original,changed))

    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.base=Path(temp.name);self.core=self.base/'core';self.hopper=self.base/'other-site'/'core'
        self.identifier='CPU_FIXTURE';self.label='B0_CORR-DEV-N294101'
        self.folder=self.core/'episodes'/self.identifier;self.relay=self.folder/'relay'
        self.resident=self.hopper/'evidence'/self.label/'resident'
        for name in ('observations','commands','acks','intents'):(self.relay/name).mkdir(parents=True)
        for name in ('inbox','outbox','receipts'):(self.resident/name).mkdir(parents=True)
        for camera in ('body_rgb','wrist_rgb'):
            (self.folder/camera).mkdir()
            for index in (0,8):(self.folder/camera/f'{index:04}.png').write_bytes(b'CPU_CAMERA_NOT_REAL_RGB'+str(index).encode())
        self.identity=dict(label='B0_CORR',global_step=2048,checkpoint_manifest_sha256='f'*64,
            relay_identity=dict(source_sha='a'*40))
        self.card=dict(id=self.identifier,pv1_session_label=self.label,scene_group='SCENE',task='T1',noise_namespace='PI0294/PV1/DEV/N294101')
        self.row=dict(attempt_id=self.identifier,status='VALID_TASK_FAIL',failure_stage='ORIGINAL_REFERENCE_GATE_REJECT',
            pipeline_trustworthy=True,live_native_failures=0,teacher_takeover=False,
            main_integrations=200,S1_rejections=0,model='B0_CORR')
        write(self.folder/'RESULT.json',dict(status='CPU_FIXTURE'))
        write(self.folder/'PV1_ADJUDICATION.json',self.row);write(self.folder/'FILE_MANIFEST.json',[])
        write(self.core/'worker_processes'/(self.identifier+'.json'),dict(pid=2**30,proc_start='1'))
        write(self.core/'evidence/PV1_AUDIT_RECEIPTS'/(self.identifier+'.json'),
            dict(status='PASS_INDEPENDENT_ACTUAL_EVIDENCE_AUDIT',adjudication=self.row))
        write(self.folder/'CARD.json',self.card)
        admission=self.hopper/'pbs/alloc-03/ALLOCATION_ADMISSION.json';write(admission,dict(job_id='CPU_NOT_A_JOB'))
        qualification=admission.parent/'PV1_NATIVE_ENDURANCE_AND_TRANSPORT.json';write(qualification,dict(CPU_FIXTURE=True))
        ready=dict(status='PV1_NATIVE_MODEL_READY',run_uuid=self.hopper.name,job_id='CPU_NOT_A_JOB',identity=self.identity,
            native_pre_model_qualification=dict(status='PASS_SAME_JOB_NATIVE_QUALIFICATION_READBACK',receipt_sha256=sha(qualification)))
        write(self.resident/'STATE.json',ready);write(self.core/'plans'/self.label/'REMOTE_READY.json',ready)
        write(self.hopper/'plans'/self.label/'REGISTRY.json',dict(run_uuid=self.hopper.name,job_id='CPU_NOT_A_JOB',
            checkpoint_identity=self.identity,cases={'CPU_EPISODE':dict(attempt_id=self.identifier,card=self.card,layout_id='SCENE/T1')}))
        state=[0.,0.,0.,1.,0.,0.,0.,1.,0.,.5];previous=None;draws=[]
        for sequence in range(2):
            payload=dict(state_eef10=state,snapshot_uuid='CPU_SNAPSHOT_'+str(sequence),images=[
                dict(sha256=sha(self.folder/camera/f'{sequence*8:04}.png')) for camera in ('body_rgb','wrist_rgb')])
            observation=sealed(dict(run_uuid=self.core.name,episode_uuid='CPU_EPISODE',sequence=sequence,tick=sequence*200,payload=payload))
            write(self.relay/'observations'/f'{sequence:03}.json',observation)
            key='CPU_QUERY_'+str(sequence)
            query=sealed(dict(run_uuid=self.core.name,query_uuid=key,reference_uuid='CPU_REF',episode_uuid='CPU_EPISODE',sequence=sequence,
                tick=sequence*200,payload=payload,identity=self.identity['relay_identity'],previous_ack=previous,
                observation_sha256=observation['sha256'],anchor_sha256=digest(state)))
            physical=np.tile(state,(40,1));physical[:,0]=10. if sequence else 0.
            action=sealed(dict(run_uuid=self.core.name,episode_uuid='CPU_EPISODE',sequence=sequence,payload={k:query['body'][k] for k in
                ('query_uuid','reference_uuid','observation_sha256','anchor_sha256','identity')}|dict(trajectory_eef10=physical.tolist())))
            if sequence==0:
                command=sealed(dict(sequence=0,prefix_sha256=digest(physical[:8].tolist())))
                write(self.relay/'commands/000.json',dict(request=query,action=action,command=command))
                write(self.relay/'intents/000.json',dict(CPU_FIXTURE=True))
                physics=self.relay/'physics-000.npz';physics.write_bytes(b'CPU_NO_PHYSICS_EXECUTED')
                previous=sealed(dict(episode_uuid='CPU_EPISODE',sequence=0,tick=0,payload=dict(tick_after=200,integrations=200,
                    query_uuid=key,reference_uuid='CPU_REF',prefix_sha256=command['body']['prefix_sha256'],
                    command_sha256=command['sha256'],physics_receipt_sha256=sha(physics))))
                write(self.relay/'acks/000.json',previous)
            else:
                gate=gate_action(physical,state,.4,'CPU_REF');self.assertEqual(gate['decision'],'REJECT')
                write(self.folder/'MODEL_REJECT.json',dict(query_uuid=key,action_sha256=action['sha256'],original_gate=gate))
            write(self.resident/'inbox'/(key+'.json'),query);write(self.resident/'outbox'/(key+'.json'),action)
            array=self.resident/'receipts'/(key+'.npz');np.savez(array,physical=physical)
            seed=int.from_bytes(hashlib.sha256((self.card['noise_namespace']+'/SCENE/T1/'+str(sequence)).encode()).digest()[:8],'big')%(2**63)
            write(self.resident/'receipts'/(key+'.json'),dict(seed=seed,action_sha256=action['sha256'],arrays_sha256=sha(array),
                noise=dict(shape=[1,40,132],dtype='torch.float32',sha256='d'*64),
                inference_precision=dict(mode='ORIGINAL_E0_BF16_AUTOCAST_FP32_MATMUL_HIGHEST',bf16_autocast=True,cuda_matmul_allow_tf32=False)))
            draws.append(dict(query_uuid=key,episode_uuid='CPU_EPISODE',sequence=sequence,response_sha256=action['sha256']))
        write(self.resident/'QUERY_STATE.json',dict(queries=draws,episodes={'CPU_EPISODE':dict(next_sequence=2)}))

    def proof(self):return audit.core(self.core,self.identifier)

    def test_original_reference_rejection_is_a_valid_uninstalled_zero_live_draw(self):
        proof=self.proof();value=audit.hopper(self.hopper,proof)
        self.assertEqual((value['draws'],value['executed_ACKs']),(2,1))
        self.assertEqual(value['records'][-1]['classification'],'ORIGINAL_REFERENCE_GATE_UNINSTALLED_ZERO_LIVE')
        self.assertFalse((self.relay/'commands/001.json').exists())
        self.assertEqual(proof['raw_camera_hashes_matched'],4)
        self.assertEqual(audit.join(proof,value)['status'],'PASS_INDEPENDENT_ACTUAL_TWO_SITE_CAUSAL_CHAIN')
        with self.assertRaises(PermissionError):audit.join(proof,dict(value,draws=1))

    def test_forged_previous_ack_saved_camera_missing_ack_or_extra_install_are_rejected(self):
        for change in ('camera','ACK','install','snapshot'):
            with self.subTest(change=change):
                if change=='camera':
                    p=self.folder/'body_rgb/0008.png';before=p.read_bytes();p.write_bytes(b'WRONG')
                elif change=='ACK':
                    p=self.relay/'acks/000.json';before=p.read_bytes();p.unlink()
                elif change=='install':
                    p=self.relay/'commands/001.json';before=None;write(p,dict(WRONG=True))
                else:
                    p=self.relay/'observations/001.json';before=p.read_bytes();v=read(p)['body'];v['payload']['snapshot_uuid']='CPU_SNAPSHOT_0';write(p,sealed(v))
                with self.assertRaises(PermissionError):self.proof()
                if before is None:p.unlink()
                else:p.write_bytes(before)

    def test_native_bytes_checkpoint_previous_ack_noise_and_precision_cannot_drift(self):
        proof=self.proof()
        variants=[('STATE.json',lambda v:v['identity'].update(checkpoint_manifest_sha256='e'*64)),
            ('inbox/CPU_QUERY_1.json',lambda v:v['body'].update(previous_ack=None)),
            ('receipts/CPU_QUERY_1.json',lambda v:v.update(seed=1)),
            ('receipts/CPU_QUERY_1.json',lambda v:v['inference_precision'].update(cuda_matmul_allow_tf32=True)),
            ('receipts/CPU_QUERY_1.json',lambda v:v.update(arrays_sha256='e'*64))]
        for name,change in variants:
            p=self.resident/name;before=p.read_bytes();v=read(p);change(v)
            if 'body' in v:v=sealed(v['body'])
            write(p,v)
            with self.subTest(name=name),self.assertRaises(PermissionError):audit.hopper(self.hopper,proof)
            p.write_bytes(before)

    def test_zero_live_reference_veto_does_not_invent_an_execution_ack(self):
        # Read-side count contract rejects a fabricated ACK even if a row says zero.
        row=deepcopy(self.row);row['main_integrations']=0;write(self.folder/'PV1_ADJUDICATION.json',row)
        write(self.core/'evidence/PV1_AUDIT_RECEIPTS'/(self.identifier+'.json'),
            dict(status='PASS_INDEPENDENT_ACTUAL_EVIDENCE_AUDIT',adjudication=row))
        with self.assertRaisesRegex(PermissionError,'CONTIGUOUS'):self.proof()

    def engineering_fixture(self):
        self.row.update(status='ENGINEERING_INCOMPLETE',pipeline_trustworthy=False,failure_stage='ENGINEERING_OR_UNKNOWN')
        write(self.folder/'PV1_ADJUDICATION.json',self.row)
        write(self.core/'evidence/PV1_AUDIT_RECEIPTS'/(self.identifier+'.json'),
            dict(status='PASS_INDEPENDENT_ACTUAL_EVIDENCE_AUDIT',adjudication=self.row))
        for name in ('inbox/CPU_QUERY_1.json','outbox/CPU_QUERY_1.json','receipts/CPU_QUERY_1.json','receipts/CPU_QUERY_1.npz'):
            (self.resident/name).unlink()
        state=read(self.resident/'QUERY_STATE.json');state['queries']=state['queries'][:1]
        state['episodes']['CPU_EPISODE']['next_sequence']=1;write(self.resident/'QUERY_STATE.json',state)
        (self.folder/'MODEL_REJECT.json').unlink()

    def test_engineering_prefix_closure_is_separate_from_valid_outcome_and_preserves_unused_snapshot(self):
        self.engineering_fixture()
        with self.assertRaises(PermissionError):self.proof()
        core=audit.core(self.core,self.identifier,engineering=True);hopper=audit.hopper(self.hopper,core)
        self.assertEqual((core['draws'],core['executed_ACKs']),(1,1))
        self.assertEqual(core['terminal_unused_observation']['model_query_generated'],'UNKNOWN_UNTIL_NATIVE_READBACK')
        self.assertFalse(hopper['terminal_unused_observation_model_query_generated'])
        with self.assertRaises(PermissionError):audit.join(core,hopper)
        result=audit.join_engineering(core,hopper)
        self.assertEqual(result['status'],'PASS_INDEPENDENT_ACTUAL_ENGINEERING_EFFECT_CLOSURE_NOT_POLICY_RESULT')
        self.assertFalse(result['policy_outcome_valid']);self.assertFalse(result['replacement_admitted'])
        (self.relay/'acks/000.json').unlink()
        with self.assertRaisesRegex(PermissionError,'CONTIGUOUS'):
            audit.core(self.core,self.identifier,engineering=True)

    def test_engineering_unused_camera_or_unclassified_native_draw_cannot_be_hidden(self):
        self.engineering_fixture();core=audit.core(self.core,self.identifier,engineering=True)
        state=read(self.resident/'QUERY_STATE.json');state['queries'].append(dict(query_uuid='UNKNOWN',episode_uuid='CPU_EPISODE'))
        write(self.resident/'QUERY_STATE.json',state)
        with self.assertRaises(PermissionError):audit.hopper(self.hopper,core)
        (self.folder/'body_rgb/0008.png').write_bytes(b'WRONG_UNUSED_CAMERA')
        with self.assertRaisesRegex(PermissionError,'UNUSED_OBSERVATION_ACTUAL_CAMERA'):
            audit.core(self.core,self.identifier,engineering=True)

    def test_complete_generated_tail_is_accounted_without_inventing_an_install_or_ACK(self):
        saved={n:(self.resident/n).read_bytes() for n in ('inbox/CPU_QUERY_1.json','outbox/CPU_QUERY_1.json',
            'receipts/CPU_QUERY_1.json','receipts/CPU_QUERY_1.npz','QUERY_STATE.json')}
        self.engineering_fixture()
        for name,raw in saved.items():(self.resident/name).write_bytes(raw)
        core=audit.core(self.core,self.identifier,engineering=True);hopper=audit.hopper(self.hopper,core)
        self.assertEqual(hopper['draws'],2);self.assertEqual(hopper['executed_ACKs'],1)
        self.assertEqual(hopper['engineering_generated_uninstalled_draws'],1)
        self.assertEqual(hopper['records'][-1]['classification'],'ENGINEERING_GENERATED_UNINSTALLED_ZERO_LIVE_TAIL')
        result=audit.join_engineering(core,hopper)
        self.assertFalse(result['policy_outcome_valid']);self.assertFalse(result['replacement_admitted'])
        self.assertEqual(result['engineering_generated_uninstalled_draws'],1)
        self.assertFalse((self.relay/'commands/001.json').exists());self.assertFalse((self.relay/'acks/001.json').exists())
        with self.assertRaises(PermissionError):audit.join(core,hopper)
        bad=deepcopy(core);bad['terminal_unused_observation']['live_integrations']=1
        with self.assertRaises(PermissionError):audit.join_engineering(bad,hopper)

    def test_uninstalled_tail_wrong_previous_ack_or_wrong_saved_camera_is_not_effect_closure(self):
        saved={n:(self.resident/n).read_bytes() for n in ('inbox/CPU_QUERY_1.json','outbox/CPU_QUERY_1.json',
            'receipts/CPU_QUERY_1.json','receipts/CPU_QUERY_1.npz','QUERY_STATE.json')}
        self.engineering_fixture()
        for name,raw in saved.items():(self.resident/name).write_bytes(raw)
        core=audit.core(self.core,self.identifier,engineering=True)
        path=self.resident/'inbox/CPU_QUERY_1.json';v=read(path)['body'];v['previous_ack']=None;write(path,sealed(v))
        with self.assertRaises(PermissionError):audit.hopper(self.hopper,core)
