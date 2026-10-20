import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.pi0294.pv1 import pbs_owner as owner
from scripts.pi0294.test_pv1_pbs_owner import root_ledger,submitted
from scripts.pi0294.test_mp1_transport_fixture import put
from scripts.pi0294.native_dataset import sha


class ReleaseRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root,self.ledger=root_ledger(self.tmp.name);self.d=self.root/'pbs/alloc-01';self.d.mkdir(parents=True)
        self.now=1005.;self.r=dict(stage='B0_DEV',request_id='fixture-1',project='CFP04-CF-007',job_name='pi0294_pv1_01',
            submission_nonce='exact-nonce',authentication_context_fingerprint='same',authentication_principal='fixture')
        put(self.d/'REQUEST.json',self.r,True)
        put(self.d/'PBS_READY.json',dict(job_id='123.hopper-m-02',observed_epoch=1000.,request_sha256=sha(self.d/'REQUEST.json')),True)
        v=json.loads(self.ledger.path.read_text());v['PBS_submissions']=[submitted()];v['used']['PBS_submissions']=1;v['reserved']['H200_s']=7260
        put(self.ledger.path,v)
        self.job=dict(job_state='R',Job_Owner='fixture@host',project=self.r['project'],Job_Name=self.r['job_name'],
            Rerunable=False,run_count=1,Variable_List=dict(PV1_REQUEST_ID=self.r['request_id'],PV1_SUBMISSION_NONCE=self.r['submission_nonce']))

    def proof(self):
        return dict(source=owner.PROJECT_SOURCE,project=dict(name=self.r['project']),balance_su=1000,observed_epoch=self.now)

    def patches(self):
        stack=__import__('contextlib').ExitStack()
        stack.enter_context(patch.object(owner,'validate'))
        stack.enter_context(patch.object(owner,'native',return_value=self.job))
        stack.enter_context(patch.object(owner,'auth_context',return_value=dict(fingerprint='same',principal='fixture')))
        stack.enter_context(patch.object(owner.time,'time',side_effect=lambda:self.now))
        stack.enter_context(patch.dict(os.environ,USER='fixture'))
        return stack

    def test_duplicate_callback_reconciles_without_writing_or_reauth(self):
        with self.patches(),patch.object(owner,'project',side_effect=lambda name,**kw:self.proof()) as project:
            self.assertEqual(owner.release(self.root,1)['status'],'PV1_CURRENT_PROJECT_RELEASED_REQUEST_UNCHANGED')
            before={n:(self.d/n).read_bytes() for n in ('START_NEURAL.json','RUNNING_PROJECT_ATTESTATION.json')}
            self.now=1300.
            self.assertEqual(owner.release(self.root,1)['status'],'PV1_ALREADY_RELEASED_EXACT_NO_SECOND_WRITE')
            self.assertEqual(project.call_count,1)
            self.assertEqual(before,{n:(self.d/n).read_bytes() for n in before})

    def test_partial_proof_recovers_with_fresh_query_and_original_proof_unchanged(self):
        original=owner.publish_release_record
        def crash(p,v):
            if Path(p).name=='START_NEURAL.json':raise RuntimeError('CPU_FIXTURE_CRASH_BEFORE_START')
            return original(p,v)
        with self.patches(),patch.object(owner,'project',side_effect=lambda name,**kw:self.proof()) as project:
            with patch.object(owner,'publish_release_record',side_effect=crash):
                with self.assertRaises(RuntimeError):owner.release(self.root,1)
            before=(self.d/'RUNNING_PROJECT_ATTESTATION.json').read_bytes();self.now=1060.
            result=owner.release(self.root,1)
            self.assertTrue(result['recovered_existing_project_attestation']);self.assertEqual(project.call_count,2)
            self.assertEqual(before,(self.d/'RUNNING_PROJECT_ATTESTATION.json').read_bytes())

    def test_exact_deadline_and_slow_query_cannot_create_a_late_release(self):
        with self.patches(),patch.object(owner,'project',side_effect=lambda name,**kw:self.proof()):
            self.now=1175.
            with self.assertRaises(PermissionError):owner.release(self.root,1)
            self.assertFalse((self.d/'START_NEURAL.json').exists());self.now=1005.
            def slow(name,**kw):self.now=1175.;return self.proof()
            with patch.object(owner,'project',side_effect=slow):
                with self.assertRaises(PermissionError):owner.release(self.root,1)
            self.assertFalse((self.d/'RUNNING_PROJECT_ATTESTATION.json').exists())

    def test_context_and_native_nonce_mismatch_rejected_before_project_query(self):
        with self.patches(),patch.object(owner,'project') as project:
            with patch.object(owner,'auth_context',return_value=dict(fingerprint='other',principal='fixture')):
                with self.assertRaises(PermissionError):owner.release(self.root,1)
            self.job['Variable_List']['PV1_SUBMISSION_NONCE']='different'
            with self.assertRaises(PermissionError):owner.release(self.root,1)
            project.assert_not_called()

    def test_auth_failure_still_allows_exact_terminal_settlement(self):
        with self.patches(),patch.object(owner,'project',side_effect=PermissionError('PV1_PROJECT_AUTH_EXPIRED_15006')):
            with self.assertRaises(PermissionError):owner.release(self.root,1)
            self.assertFalse((self.d/'START_NEURAL.json').exists())
            self.job.update(job_state='F',Resource_List=dict(ngpus=1),resources_used=dict(walltime='00:03:08'),stime=1000,obittime=1198,Exit_status=1)
            with patch.object(owner.subprocess,'check_output',return_value='1024 campaign'):
                self.assertEqual(owner.settle(self.root)['rows'][0]['conservative_GPU_s'],198)
        self.assertEqual(json.loads(self.ledger.path.read_text())['reserved']['H200_s'],0)

    def test_start_inode_is_readonly_at_publication_and_existing_evidence_cannot_be_overwritten(self):
        link=os.link;seen=[]
        def observe(source,target):
            self.assertEqual(Path(source).stat().st_mode&0o777,0o400)
            seen.append(Path(target).name);return link(source,target)
        with patch.object(owner.os,'link',side_effect=observe):
            p=self.d/'atomic-fixture.json';owner.publish_release_record(p,dict(value=1))
            before=p.read_bytes()
            with self.assertRaises(FileExistsError):owner.publish_release_record(p,dict(value=2))
            self.assertEqual(before,p.read_bytes())
        self.assertEqual(seen,['atomic-fixture.json','atomic-fixture.json'])
        self.assertEqual(list(self.d.glob('.*.new-*')),[])

    def test_readback_has_native_pid_and_exact_publication_without_replaying(self):
        with self.patches(),patch.object(owner,'project',side_effect=lambda name,**kw:self.proof()):
            owner.release(self.root,1)
            result=owner.release_readback(self.root,1)
        self.assertEqual(result['status'],'TERMINAL_RELEASE_WRITER_EXACT_FILES_FOR_RECONCILIATION')
        self.assertEqual(result['active'],[]);self.assertEqual(len(result['attempts']),1)
        self.assertEqual(set(result['publications']),{'RUNNING_PROJECT_ATTESTATION.json','START_NEURAL.json'})
        self.assertFalse(result['new_release'])

    def test_unknown_reply_with_live_exact_writer_is_wait_not_replay(self):
        d=self.d/'release_attempts';d.mkdir()
        v=dict(run_uuid=self.root.name,request_sha256=sha(self.d/'REQUEST.json'),request_id=self.r['request_id'],
            submission_nonce=self.r['submission_nonce'],job_id='123.hopper-m-02',hostname=os.uname().nodename,
            pid=os.getpid(),proc_start=Path('/proc/self/stat').read_text().split()[21])
        put(d/'fixture.json',v,True)
        with self.patches():result=owner.release_readback(self.root,1)
        self.assertEqual(result['status'],'LIVE_RELEASE_PROCESS_DO_NOT_REPLAY')
        self.assertEqual(result['active'][0]['pid'],os.getpid())

    def test_lock_alone_is_unknown_not_a_terminal_or_verified_process(self):
        lock=owner.OwnerLock(self.d/'PV1_RELEASE_OWNER.lock')
        try:
            with self.patches():result=owner.release_readback(self.root,1)
        finally:lock.close()
        self.assertEqual(result['status'],'UNKNOWN_INFLIGHT_RELEASE_LOCK_NO_REPLAY')
        self.assertEqual(result['active'],[])
