import contextlib
import io
import os
from pathlib import Path
import sys
import unittest

from scripts.pi0294.pv1.auth_lifecycle import Maintainer,masked_login,HumanAuthRequired,SiteClock,existing_job_auth


class AuthLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.now=1000.;self.expired=False;self.fingerprint='same';self.renewals=0;self.queries=0

    def query(self):
        self.queries+=1
        if self.expired:raise PermissionError('PV1_PROJECT_AUTH_EXPIRED_15006')
        return dict(observed_epoch=self.now,native_fixture=True)

    def renew(self):self.renewals+=1;self.expired=False

    def manager(self):
        return Maintainer(self.query,self.renew,lambda:dict(fingerprint=self.fingerprint),clock=lambda:self.now)

    def test_queue_crosses_expiry_and_R_never_uses_Q_cached_proof(self):
        m=self.manager();m.bind();self.now+=3600;self.expired=True
        m.ensure(force=True);self.assertEqual(self.renewals,1)
        self.now+=1;ready=self.now
        proof=m.ensure(force=True,ready_epoch=ready)
        self.assertEqual(proof['proof']['observed_epoch'],ready)
        self.assertEqual(self.queries,4)

    def test_R_expiry_recovers_only_with_original_time_budget(self):
        m=self.manager();m.bind();self.expired=True;self.now+=10
        m.ensure(force=True,ready_epoch=self.now-10);self.assertEqual(self.renewals,1)
        self.expired=True;self.now+=200
        with self.assertRaises(TimeoutError):m.ensure(force=True,ready_epoch=self.now-150)
        self.assertEqual(self.renewals,1)

    def test_helper_release_context_mismatch_and_renewal_context_change_rejected(self):
        m=self.manager();m.bind();self.fingerprint='another'
        with self.assertRaisesRegex(PermissionError,'CONTEXT_MISMATCH'):m.ensure(force=True)
        self.fingerprint='same';self.expired=True
        def renew():self.expired=False;self.fingerprint='another'
        m.renew=renew
        with self.assertRaisesRegex(PermissionError,'CHANGED_PRINCIPAL'):m.ensure(force=True)

    def test_rights_revoked_does_not_attempt_password_renewal(self):
        m=self.manager();m.bind()
        def revoked():raise PermissionError('PV1_FRESH_PROJECT_ACTIVE_DATE_MEMBERSHIP_CLUSTER_AND_BALANCE')
        m.query=revoked
        with self.assertRaises(PermissionError):m.ensure(force=True)
        self.assertEqual(self.renewals,0)

    def test_failed_auth_does_not_destroy_owner_or_prevent_terminal_settlement(self):
        m=self.manager();m.bind();self.expired=True;settled=[]
        def failed():raise HumanAuthRequired('HUMAN_AUTH_ONCE_REQUIRED')
        m.renew=failed
        settle=lambda:settled.append('EXACT_NATIVE_TERMINAL_JOB')
        blocked=existing_job_auth(m,'R',settle,ready_epoch=self.now)
        self.assertEqual(blocked['status'],'AUTH_BLOCKED_KEEP_OWNER_AND_SETTLE_TERMINAL')
        self.assertEqual(settled,[])
        terminal=existing_job_auth(m,'F',settle)
        self.assertEqual(terminal['status'],'NATIVE_TERMINAL_SETTLED')
        self.assertEqual(settled,['EXACT_NATIVE_TERMINAL_JOB']);self.assertEqual(m.next_check,1030.)

    def test_job_auth_waits_for_native_ready_and_never_releases_twice(self):
        m=self.manager();m.bind();settle=lambda:self.fail('not terminal')
        self.assertEqual(existing_job_auth(m,'R',settle)['status'],'WAIT_EXACT_PBS_READY_NO_RELEASE')
        self.assertEqual(existing_job_auth(m,None,settle)['status'],'UNKNOWN_NATIVE_STATE_PRESERVE_HANDLE_NO_RELEASE')
        self.assertEqual(existing_job_auth(m,'R',settle,ready_epoch=self.now,released=True)['status'],
                         'EXISTING_RELEASE_NO_SECOND_PUBLICATION')
        self.assertEqual(self.queries,1)
        self.assertEqual(existing_job_auth(m,'R',settle,ready_epoch=self.now)['status'],
                         'R_AUTH_READY_NOT_YET_RELEASED')
        self.assertEqual(self.queries,2)

    def test_login_error_diagnostics_never_include_raw_output_or_secret(self):
        secret='CPU_FIXTURE_NOT_A_REAL_PASSWORD'
        program="import getpass;getpass.getpass('Password: ');print('Connection timed out; Code 15006');raise SystemExit(1)"
        with self.assertRaises(ConnectionError) as caught:
            masked_login([sys.executable,'-c',program],secret,timeout_s=3)
        diagnostic=caught.exception.diagnostic
        self.assertEqual(diagnostic['server_codes'],[15006])
        self.assertTrue(diagnostic['password_prompt_answered'])
        self.assertFalse(diagnostic['raw_output_logged'])
        self.assertNotIn(secret,str(diagnostic)+str(caught.exception))

    def test_stale_and_pre_R_proofs_rejected(self):
        m=self.manager();m.bind();m.query=lambda:dict(observed_epoch=self.now-31)
        with self.assertRaises(PermissionError):m.ensure(force=True)
        m.query=lambda:dict(observed_epoch=self.now-1)
        with self.assertRaises(TimeoutError):m.ensure(force=True,ready_epoch=self.now)

    def test_masked_native_style_password_prompt_without_plaintext_output_or_fd_leak(self):
        secret='CPU_FIXTURE_NOT_A_REAL_PASSWORD';before=len(list(Path('/proc/self/fd').iterdir()))
        argv=[sys.executable,'-c',"import getpass;getpass.getpass('Password: ');print('ok')"]
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            for unused in range(3):
                result=masked_login(argv,secret,timeout_s=3)
                self.assertFalse(result['credential_saved']);self.assertFalse(result['raw_output_logged'])
        self.assertNotIn(secret,output.getvalue())
        self.assertLessEqual(len(list(Path('/proc/self/fd').iterdir())),before+1)

    def test_mfa_is_not_bypassed_and_secret_not_put_in_command_arguments(self):
        with self.assertRaises(HumanAuthRequired):masked_login(
            [sys.executable,'-c',"print('OTP verification code: ',flush=True);input()"],'fixture',timeout_s=3)
        with self.assertRaises(ValueError):masked_login(['fake','--password','fixture'],'fixture')

    def test_site_clock_does_not_compare_PC_epoch_with_site_proof(self):
        monotonic=[100.3];clock=SiteClock(lambda:monotonic[0])
        with self.assertRaises(PermissionError):clock.now()
        clock.observe(5000.068,100.,100.3,site_processing_s=.2)
        self.assertAlmostEqual(clock.now(),5000.168)
        monotonic[0]+=.5;self.assertAlmostEqual(clock.now(),5000.668)
        with self.assertRaises(ValueError):clock.observe(5000.,100.,100.3,site_processing_s=1.)


if __name__=='__main__':unittest.main()
