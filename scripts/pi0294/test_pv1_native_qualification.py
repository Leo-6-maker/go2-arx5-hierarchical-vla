"""CPU contract fixtures, explicitly not native PV1 qualification receipts."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.pi0292.durable import read
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.pv1 import heartbeat_capture as heartbeat
from scripts.pi0294.pv1 import pre_model_qualification as qualification
from scripts.pi0294.pv1 import native_transport_qualification as transport
from scripts.pi0294.pv1 import transport_fixture as fixture
from scripts.pi0294.pv1.decision import AUTHORITY
from scripts.pi0294.test_mp1_transport_fixture import put


class NativeHeartbeatContracts(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name)/'policy-01';self.directory=self.root/'pbs/alloc-24'
        self.directory.mkdir(parents=True)
        put(self.root/'RUN_MANIFEST.json',dict(code=dict(vla_commit='a'*40)),True)
        self.request=dict(submit_count=24,run_uuid=self.root.name,source_sha='a'*40,
            run_manifest_sha256=sha(self.root/'RUN_MANIFEST.json'),project='CFP05-CF-025',ncpus=12,
            job_name='pi0294_pv1_24',request_id='sealed-request',submission_nonce='sealed-nonce')
        put(self.directory/'REQUEST.json',self.request,True)
        self.allocation=dict(status='PASS_PV1_OWN_NATIVE_H200_ALLOCATION',run_uuid=self.root.name,
            job_id='123.hopper-m-02',hostname='hopper-h200-01',project=self.request['project'],
            allocation_start_epoch=1000,allocation_end_epoch=10000,request_sha256=sha(self.directory/'REQUEST.json'))
        put(self.directory/'ALLOCATION_ADMISSION.json',self.allocation,True)
        self.job=dict(job_state='R',Job_Owner='fixture@host',project=self.request['project'],
            Job_Name=self.request['job_name'],exec_host='hopper-h200-01/0',
            Rerunable='False',run_count=1,
            Variable_List=dict(PV1_REQUEST_ID='sealed-request',PV1_SUBMISSION_NONCE='sealed-nonce'),
            Resource_List=dict(ngpus=1,ncpus=12,select='1:ngpus=1:gpu_model=H200:ncpus=12:mem=128GB'))

    def test_actual_native_host_job_owner_project_gpu_and_no_model_before_endurance(self):
        with patch.dict(heartbeat.os.environ,{'PBS_JOBID':self.allocation['job_id'],'USER':'fixture'}),\
                patch.object(heartbeat.socket,'gethostname',return_value=self.allocation['hostname']),\
                patch.object(heartbeat.time,'time',return_value=2000):
            for change in ('good','foreign_project','foreign_user','foreign_node','not_running','two_gpu','A40','loaded','rerun','nonce'):
                job=deepcopy(self.job);model=self.root/'pbs/MODEL_STATE.json'
                if model.exists():model.unlink()
                if change=='foreign_project':job['project']='other'
                if change=='foreign_user':job['Job_Owner']='other@host'
                if change=='foreign_node':job['exec_host']='other/0'
                if change=='not_running':job['job_state']='Q'
                if change=='two_gpu':job['Resource_List']['ngpus']=2
                if change=='A40':job['Resource_List']['select']=job['Resource_List']['select'].replace('H200','A40')
                if change=='loaded':put(model,dict(job_id=self.allocation['job_id']))
                if change=='rerun':job['Rerunable']='True'
                if change=='nonce':job['Variable_List']['PV1_SUBMISSION_NONCE']='other'
                with self.subTest(change=change),patch.object(heartbeat,'owned_jobs',return_value={self.allocation['job_id']:job}):
                    if change=='good':self.assertEqual(heartbeat.identity(self.root,self.directory,self.allocation)['source_sha'],'a'*40)
                    else:
                        with self.assertRaises(PermissionError):heartbeat.identity(self.root,self.directory,self.allocation)

    def fragment(self):
        rows=[]
        for i in range(81):
            clock=1000+7.5*i
            rows.append(dict(sequence=i,monotonic_s=i*7.5,proof=dict(server_revalidation=True,
                revalidated_clock_epoch=clock,cached_clock_epoch=clock,clock_sampled_after_read=True,
                heartbeat_written=False,watchdog_s=300,heartbeat=dict(run_uuid=self.root.name,observed_epoch=clock-10))))
        journal=self.directory/'heartbeat-capture/READS.jsonl';journal.parent.mkdir(exist_ok=True)
        if journal.exists():journal.chmod(0o600)
        journal.write_text(''.join(json.dumps(r)+'\n' for r in rows));journal.chmod(0o400)
        value=dict(schema='PI0294_PV1_NATIVE_HEARTBEAT_CAPTURE_V1',status='PASS_NATIVE_HEARTBEAT_FRAGMENT_ONLY',
            authority_release=AUTHORITY,run_uuid=self.root.name,job_id=self.allocation['job_id'],hostname=self.allocation['hostname'],
            source_sha='a'*40,manifest_sha256=sha(self.root/'RUN_MANIFEST.json'),request_sha256=self.allocation['request_sha256'],
            clock_source=heartbeat.CLOCK,rows=rows,raw_journal_relative=str(journal.relative_to(self.root)),raw_journal_sha256=sha(journal),
            models_loaded=0,new_model_queries=0,new_physics_integrations=0,heartbeat_written=False,transport_qualified=False)
        put(self.directory/heartbeat.FILENAME,value,True)
        return value,journal

    def test_fragment_requires_raw_journal_original_clock_and_600_actual_seconds(self):
        value,_=self.fragment();self.assertEqual(heartbeat.verify(self.root,self.directory,self.allocation),value)
        for change in ('future','expired','wrong_run','before_read','short','wrong_job','old_authority','writable','raw_changed'):
            value,journal=self.fragment()
            p=value['rows'][20]['proof']
            if change=='future':p['heartbeat']['observed_epoch']=p['revalidated_clock_epoch']+.001
            if change=='expired':p['heartbeat']['observed_epoch']=p['revalidated_clock_epoch']-300.001
            if change=='wrong_run':p['heartbeat']['run_uuid']='other'
            if change=='before_read':p['clock_sampled_after_read']=False
            if change=='short':value['rows']=value['rows'][:-1]
            if change=='wrong_job':value['job_id']='previous.hopper-m-02'
            if change=='old_authority':value['authority_release']='b'*40
            if change not in ('writable','raw_changed'):
                journal.chmod(0o600);journal.write_text(''.join(json.dumps(r)+'\n' for r in value['rows']));journal.chmod(0o400)
                value['raw_journal_sha256']=sha(journal)
            if change=='raw_changed':journal.chmod(0o600);journal.write_text('{}\n');journal.chmod(0o400)
            put(self.directory/heartbeat.FILENAME,value,change!='writable')
            with self.subTest(change=change),self.assertRaises(PermissionError):heartbeat.verify(self.root,self.directory,self.allocation)

    def test_pre_model_receipt_cannot_be_absent_or_inherit_old_authority(self):
        with self.assertRaisesRegex(PermissionError,'NOT_RUN'):qualification.require(self.root,self.directory,self.allocation)
        v=dict(schema='MP1_N1_NATIVE_ENDURANCE_AND_TRANSPORT_V1',status='PASS_NATIVE_PREMODEL_HEARTBEAT_AND_REAL_TRANSPORT')
        put(self.directory/qualification.FILENAME,v,True)
        with self.assertRaisesRegex(PermissionError,'PV1_FRESH_SAME_NATIVE_JOB'):qualification.require(self.root,self.directory,self.allocation)

    def test_model_or_trainer_must_join_real_current_host_and_same_job_qualification(self):
        with patch.dict(qualification.os.environ,{'PBS_JOBID':self.allocation['job_id']}),\
                patch.object(qualification.socket,'gethostname',return_value=self.allocation['hostname']),\
                patch.object(qualification.time,'time',return_value=2000):
            self.assertEqual(qualification.current_directory(self.root,self.allocation),self.directory)
            with self.assertRaisesRegex(PermissionError,'NOT_RUN'):qualification.require_current(self.root,self.allocation)
            with patch.object(qualification.socket,'gethostname',return_value='hopper-login'),\
                    self.assertRaisesRegex(PermissionError,'NATIVE_CURRENT'):qualification.require_current(self.root,self.allocation)
            put(self.directory/'PBS_PROCESS_FINAL.json',dict(job_id=self.allocation['job_id']))
            with self.assertRaisesRegex(PermissionError,'TERMINAL_ALLOCATION'):qualification.current_directory(self.root,self.allocation)


class ReferenceContracts(unittest.TestCase):
    def test_existing_actual_audit_location_is_bound_without_copying_or_relabeling(self):
        from scripts.pi0294.test_mp1_transport_fixture import reference
        with tempfile.TemporaryDirectory() as tmp:
            old=Path(tmp)/'closed-old';root=Path(tmp)/'new';old.mkdir();root.mkdir()
            plan,_,_=reference(old,1000);episode=old/'episodes/fixture-closed'
            put(episode/'RESULT.json',dict(status='VALID_COMPLETE',physics_steps=16000))
            put(episode/'FILE_MANIFEST.json',[]);put(episode/'S1_RESULT.json',dict(live_native_failures=0))
            put(old/'lease.json',dict(status='CLOSED'));put(old/'OUTPUT_BINDING.private.json',dict(uid=fixture.os.getuid(),lease=str(old/'lease.json')))
            audit=old/'evidence/audits/fixture-closed/AUDIT.private.json'
            value=dict(status='PASS_INDEPENDENT_ACTUAL_EVIDENCE_AUDIT',checks=dict(full_raw=True),errors=[],
                actual_native_safety_flag_rows=0,adjudication=dict(attempt_id=episode.name,pipeline_trustworthy=True,
                    live_native_failures=0,result_sha256=sha(episode/'RESULT.json')))
            put(audit,value)
            binding=dict(original_root=str(old),original_observation_relative=plan['recorded_observation_relative'],
                original_audit_relative=str(audit.relative_to(old)),audit_sha256=sha(audit),raw_manifest_sha256=sha(episode/'FILE_MANIFEST.json'))
            p=root/'reference/RAW_REFERENCE_BINDING.private.json';put(p,binding,True)
            observed=root/'reference/RECORDED_OBSERVATION.json';observed.write_bytes((old/plan['recorded_observation_relative']).read_bytes());observed.chmod(0o400)
            plan.update(raw_reference_binding_relative=str(p.relative_to(root)),raw_reference_binding_sha256=sha(p),recorded_observation_relative=str(observed.relative_to(root)))
            m=dict(execution=dict(sealed_inputs={str(p.relative_to(root)):sha(p)}))
            self.assertEqual(fixture.reference_proof(root,plan,m)['original_audit_sha256'],sha(audit))
            self.assertFalse((episode/'AUDIT.json').exists())
            for field in ('checks','errors','actual_native_safety_flag_rows'):
                bad=deepcopy(value)
                bad[field]={'full_raw':False} if field=='checks' else ['wrong'] if field=='errors' else 1
                put(audit,bad);binding['audit_sha256']=sha(audit);put(p,binding,True)
                plan['raw_reference_binding_sha256']=sha(p);m['execution']['sealed_inputs'][str(p.relative_to(root))]=sha(p)
                with self.subTest(field=field),self.assertRaises(PermissionError):fixture.reference_proof(root,plan,m)

    def test_core_raw_reference_is_presealed_not_just_an_owner_pass_string(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);v=dict(status='PASS_ACTUAL_CLOSED_SOURCE_RECORDED_TWO_CAMERA_REFERENCE',
                recorded_observation_file_sha256='a'*64,recorded_payload_sha256='b'*64,
                raw_reference_binding_sha256='c'*64,actual_two_camera_pixels=dict(body_rgb='d'*64,wrist_rgb='e'*64),
                new_models=0,new_queries=0,new_physics=0)
            path=folder/transport.REFERENCE;put(path,v,True)
            plan={k:v[k] for k in ('recorded_observation_file_sha256','recorded_payload_sha256','raw_reference_binding_sha256')}
            plan['reference_proof_sha256']=sha(path);owner=dict(reference_proof_sha256=sha(path))
            self.assertEqual(transport.verify_reference(folder,plan,owner),v)
            for change in ('wrong_hash','wrong_pixels','models','writable'):
                bad=deepcopy(v);p=deepcopy(plan)
                if change=='wrong_hash':p['reference_proof_sha256']='f'*64
                if change=='wrong_pixels':bad['actual_two_camera_pixels'].pop('body_rgb')
                if change=='models':bad['new_models']=1
                put(path,bad,change!='writable')
                if change!='wrong_hash':p['reference_proof_sha256']=sha(path)
                with self.subTest(change=change),self.assertRaises(PermissionError):
                    transport.verify_reference(folder,p,dict(reference_proof_sha256=sha(path)))
