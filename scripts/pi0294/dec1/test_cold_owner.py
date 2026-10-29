"""Complete cold watch->RPC->journal->native heartbeat CPU call path."""
import contextlib
import errno
import io
import json
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

from scripts.pi0292.durable import read,write
from scripts.pi0294.dec1 import test_owner_transport as fixtures
from scripts.pi0294.dec1 import owner_heartbeat_publish as remote
from scripts.pi0294.dec1 import cold_owner
from scripts.pi0294.dec1.owner_rpc import invoke
from scripts.pi0294.dec1.owner_transport import ReadOnlyFailure,UnknownMutation


class ColdOwnerIntegration(unittest.TestCase):
    make_journal=fixtures.Fixture.make_journal
    # Reuse only fixture setup, not the inherited unit-test collection below.
    def setUp(self):
        fixtures.Fixture.setUp(self)
        self.local=self.base/'owner';self.local.mkdir()
        write(self.local/'SOURCE_REQUEST.private.json',dict(request={'submit_count':1},directory=str(self.directory)))
        write(self.local/'SOURCE_JOB_BOUND.private.json',{'job_id':'fixture-job'})
        write(self.local/'TRAIN_REGISTERED.private.json',{'root':str(self.root)})
        self.status='R';self.receipts={'PBS_READY.json':{}}
        self.calls=[];self.sleeps=[];self.auths=0;self.fail_read=0;self.heartbeat_fault=None
        self.release_fault=False;self.auth_fault=False;self.events=[];self.unresolved_readback=False
        self._fresh_program='FIXTURE_STATE_READ'
        self.owner=types.SimpleNamespace(BASE=self.local,read=read,update=write,native=types.SimpleNamespace(T0=0),
            previous=types.SimpleNamespace(rpc=None),HEARTBEAT='ORIGINAL_NOT_EXECUTED')
        def record(name,value):
            self.events.append((name,value));write(self.local/(name+'.json'),value)
        self.owner.record=record
        def fresh(number,job,light=True):
            rows={site:self.owner.previous.rpc(site,self._fresh_program,{'SITE':site},
                    'source-A40-current-unfiltered-native-'+site) for site in ('hopper','vanda')}
            view=dict(self.view,observed_epoch=self.now,campaign_jobs=[dict(job_id='fixture-job',state=self.status)])
            return dict(rows=rows,view=view)
        self.owner.fresh=fresh
        def release(state,secret,first_update=False):
            # Preserve the mature sequence: fresh view, heartbeat, once-release.
            state=self.owner.fresh(1,'fixture-job',light=False)
            stamp='FIRST_UPDATE' if first_update else 'MODEL_START'
            self.owner.record(stamp+'_RELEASE_INTENT',dict(once=True))
            self.owner.previous.rpc('vanda','original-heartbeat',self.values(state['view']),
                'source-A40-fresh-heartbeat-before-once-'+stamp)
            self.owner.previous.rpc('vanda','FIXTURE_RELEASE',{'FIRST_UPDATE':first_update},
                'source-A40-once-native-'+stamp+'-release')
        self.owner.release=release
        def auth(secret):
            self.auths+=1
            if self.auth_fault:raise subprocess.TimeoutExpired('fixture-auth',40)
        self.owner.auth=auth
        self.rpc=cold_owner.ColdRPC(self.j,argv_by_site={'hopper':['ssh','fixture-hopper'],'vanda':['ssh','fixture-vanda']},
            remote_root=self.root,request_directory=self.directory,code_root=self.root/'capsule/code',
            transport=lambda argv,code,timeout_s:invoke(argv,code,timeout_s=timeout_s,runner=self.runner))
        self.proof=dict(status='PASS_NEW_REGISTERED_COLD_OWNER',identity=self.identity,
                        prior_owner_closed=True,new_recovery_authority_adopted=True)
        cold_owner.bind_cold_owner(self.owner,self.rpc,validate_new_registration=lambda:self.proof)

    def values(self,view):
        return dict(TARGET=str(self.root),DIRECTORY=str(self.directory),JOB='fixture-job',VIEW=view)

    def runner(self,argv,*,input,capture_output,text,timeout):
        site=argv[-1].split('-')[-1]
        if 'FIXTURE_STATE_READ' in input:
            self.calls.append(('READ',site,timeout))
            if self.fail_read:
                self.fail_read-=1;raise subprocess.TimeoutExpired('fixture-read',timeout)
            if site=='hopper':
                result=dict(ledger_summary=dict(unknown_effects=[],pending_updates=[]))
            else:
                result=dict(jobs={'fixture-job':dict(Job_Name='pi0294_dec1_a40_train',project='CFP04-CF-007',job_state=self.status)},receipts=self.receipts)
            return subprocess.CompletedProcess(argv,0,json.dumps(result),'')
        if 'FIXTURE_RELEASE' in input:
            self.calls.append(('RELEASE',site,timeout))
            if self.release_fault is True or self.release_fault=='FIRST_UPDATE' and 'FIRST_UPDATE=True' in input:
                raise subprocess.TimeoutExpired('fixture-release',timeout)
            if "FIRST_UPDATE=True" in input:
                self.receipts['A40_PRETRAIN_RELEASE.json']={}
            else:
                self.receipts['START_NEURAL.json']={};self.receipts['A40_PRETRAIN_READY.json']={}
            return subprocess.CompletedProcess(argv,0,'{"released":true}','')
        action='PUBLISH' if 'import publish' in input else 'READBACK'
        self.calls.append((action,site,timeout))
        if action=='READBACK' and self.unresolved_readback:
            raise subprocess.TimeoutExpired('fixture-readback',timeout)
        if action=='PUBLISH' and self.heartbeat_fault=='before':
            self.heartbeat_fault=None;raise subprocess.TimeoutExpired('fixture-heartbeat',timeout)
        output=io.StringIO()
        real_publish,real_readback=remote.publish,remote.readback
        original_path=list(sys.path)
        with patch.object(remote,'publish',side_effect=lambda **kw:real_publish(**kw,clock=lambda:self.now)), \
             patch.object(remote,'readback',side_effect=lambda **kw:real_readback(**kw,clock=lambda:self.now)), \
             contextlib.redirect_stdout(output):
            try:exec(compile(input,'<native-fixture>','exec'),{})
            finally:sys.path[:]=original_path
        if action=='PUBLISH' and self.heartbeat_fault=='after':
            self.heartbeat_fault=None;raise subprocess.TimeoutExpired('fixture-heartbeat',timeout)
        return subprocess.CompletedProcess(argv,0,output.getvalue(),'')

    def sleep(self,seconds):
        self.sleeps.append(seconds);self.now+=seconds;self.mono+=seconds
        if 'A40_PRETRAIN_RELEASE.json' in self.receipts:self.status='F'

    def run_watch(self):
        return cold_owner.watch(self.owner,'fixture-secret',clock=lambda:self.now,
            monotonic=lambda:self.mono,sleep=self.sleep)

    def test_complete_two_release_chain_and_terminal_without_resubmission(self):
        result=self.run_watch()
        self.assertEqual(result['status'],'NATIVE_TERMINAL_PRESERVED_NEEDS_ACTUAL_EFFECT_SETTLEMENT')
        self.assertEqual([c[0] for c in self.calls].count('RELEASE'),2)
        self.assertEqual([c[0] for c in self.calls].count('PUBLISH'),4)
        self.assertEqual(self.sleeps,[30,30])
        self.assertTrue((self.local/'SOURCE_NATIVE_TERMINAL_UNSETTLED.json').exists())

    def test_new_minimal_session_uses_original_watch_once_without_legacy_release(self):
        from scripts.pi0294.generalization.minimal_execution import VERSION
        self.receipts={'START_NEURAL.json':{}};self.owner.request_template={'platform_training_version':VERSION}
        calls=[]
        def release(state,secret):
            calls.append('ONE_FINITE_SESSION');self.status='F';return True
        self.owner.release_science_session=release
        result=self.run_watch()
        self.assertEqual(result['native']['job_state'],'F')
        self.assertEqual(calls,['ONE_FINITE_SESSION'])
        self.assertEqual([c[0]for c in self.calls].count('RELEASE'),0)

    def test_session_hook_cannot_be_added_to_an_old_scope(self):
        self.receipts={'START_NEURAL.json':{}};self.owner.request_template={'platform_training_version':'OLD_SCOPE'}
        self.owner.release_science_session=lambda *args,**kwargs:self.fail('OLD_SCOPE_RELEASE_CALLED')
        with self.assertRaisesRegex(PermissionError,'ONLY_EXPLICIT_NEW_FIRST512'):
            self.run_watch()
        self.assertFalse(self.sleeps)

    def test_exact_e2_only_owner_survives_real_R_before_READY_and_releases_once(self):
        """1507480 captured seam: R may precede PBS_READY; no new permission."""
        from scripts.pi0294.generalization import minimal_execution_adapter as adapters
        self.receipts={}
        self.owner.request_template=dict(platform_training_version=adapters.e2only.VERSION,
                                         profile=adapters.e2only.PROFILE)
        sessions=[];original_sleep=self.sleep
        def session(state,secret):
            # The original owner phase method itself requires actual tickets.
            sessions.append(tuple(sorted(self.receipts)))
            if 'A40_PRETRAIN_RELEASE.json' in self.receipts:self.status='F';return True
            return False
        def sleep(seconds):
            original_sleep(seconds)
            if not self.receipts:self.receipts={'PBS_READY.json':{}}
        self.owner.release_science_session=session
        self.sleep=sleep
        result=self.run_watch()
        self.assertEqual(result['native']['job_state'],'F')
        self.assertEqual(sessions,[()])
        self.assertEqual([x[0] for x in self.calls].count('RELEASE'),2)
        self.assertEqual(self.sleeps,[30,30,30])

    def test_e2_version_does_not_admit_other_profile(self):
        from scripts.pi0294.generalization import minimal_execution_adapter as adapters
        self.receipts={}
        self.owner.request_template=dict(platform_training_version=adapters.e2only.VERSION,
                                         profile=adapters.saved512.PROFILE)
        self.owner.release_science_session=lambda *args,**kwargs:self.fail('WRONG_PROFILE_RELEASE')
        with self.assertRaises(PermissionError):self.run_watch()
        self.assertFalse(self.sleeps)

    def test_session_mutation_timeout_remains_UNKNOWN_without_watch_retry(self):
        from scripts.pi0294.generalization.minimal_execution import VERSION
        self.receipts={'START_NEURAL.json':{}};self.owner.request_template={'platform_training_version':VERSION}
        calls=[]
        def release(state,secret):
            def unknown(intent,limit):
                calls.append(intent['operation_id']);raise subprocess.TimeoutExpired('CPU_FIXTURE_SESSION_MUTATION',40)
            return self.j.call('FINITE_MINIMAL_SESSION','MUTATION',unknown)
        self.owner.release_science_session=release
        with self.assertRaises(UnknownMutation):self.run_watch()
        self.assertEqual(len(calls),1);self.assertFalse(self.sleeps)
        self.assertEqual(read(self.j.state_path)['status'],'UNKNOWN')

    def test_read_timeout_retries_only_read_then_original_chain(self):
        self.fail_read=1;self.run_watch()
        self.assertEqual(self.calls[:2],[('READ','hopper',40),('READ','hopper',40)])
        self.assertEqual([c[0] for c in self.calls].count('RELEASE'),2)
        self.assertEqual(len([n for n,v in self.events if n.startswith('READ_ONLY_OBSERVATION_INTERRUPTION')]),1)

    def test_three_read_timeouts_stop_without_heartbeat_or_release(self):
        self.fail_read=3
        with self.assertRaises(ReadOnlyFailure):self.run_watch()
        self.assertEqual([x[0] for x in self.calls],['READ']*3)
        self.assertEqual(self.sleeps,[30,30])

    def successor_watch(self):
        from scripts.pi0294.pv1 import successor_cd as cd
        marker=dict(cd.scope(1),grant_sha256=cd.GRANT_SHA,
            before_sha256={'GLOBAL':'a'*64,'LOCAL':'b'*64},
            authorized_storage_caps=dict(cd.NEW_STORAGE),
            old_clock_and_history_preserved=True,scope_only_not_GPU_or_physics_admission=True)
        self.owner.successor_scope=marker
        self.owner.preserve_readonly_until_deadline=True
        return lambda:cold_owner.watch(self.owner,None,clock=lambda:self.now,
            monotonic=lambda:self.mono,sleep=self.sleep,scope_binding=marker)

    def test_local_ESTALE_read_recovers_without_frontend_or_stale_heartbeat(self):
        watch=self.successor_watch();original=self.owner.fresh;attempts=[]
        def fresh(*args,**kwargs):
            attempts.append(self.now)
            if len(attempts)==1:raise OSError(errno.ESTALE,'Stale file handle')
            return original(*args,**kwargs)
        self.owner.fresh=fresh
        self.assertEqual(watch()['native']['job_state'],'F')
        self.assertEqual(self.sleeps,[30,30,30])
        interruption=[v for n,v in self.events if n.startswith('READ_ONLY_OBSERVATION_INTERRUPTION')]
        self.assertEqual(len(interruption),1)
        self.assertFalse(interruption[0]['phase_in_durable_RPC_journal'])
        self.assertEqual(interruption[0]['diagnostic']['errno'],errno.ESTALE)
        self.assertEqual(sum(c[0]=='RELEASE' for c in self.calls),2)

    def test_repeated_ESTALE_never_publishes_until_new_valid_observation(self):
        watch=self.successor_watch();original=self.owner.fresh;remaining=[4]
        def fresh(*args,**kwargs):
            if remaining[0]:
                remaining[0]-=1
                self.assertEqual(self.calls,[])
                raise OSError(errno.ESTALE,'Stale file handle')
            return original(*args,**kwargs)
        self.owner.fresh=fresh
        self.assertEqual(watch()['native']['job_state'],'F')
        self.assertEqual(self.sleeps[:4],[30]*4)
        self.assertEqual(sum(n.startswith('READ_ONLY_SUSPENDED_NO_RELEASE') for n,v in self.events),2)

    def test_legacy_or_nontransient_local_filesystem_error_is_not_retried(self):
        for number in (errno.ESTALE,errno.ENOENT,errno.EACCES):
            with self.subTest(errno=number):
                def fresh(*args,**kwargs):raise OSError(number,'read failure')
                self.owner.fresh=fresh
                with self.assertRaises(OSError):self.run_watch()
                self.assertEqual(self.sleeps,[])
        watch=self.successor_watch()
        with self.assertRaises(PermissionError):watch()
        self.assertEqual(self.sleeps,[])

    def test_ESTALE_inside_mutation_stays_UNKNOWN_and_is_never_retried(self):
        watch=self.successor_watch();original=self.rpc.transport
        def transport(argv,code,**kwargs):
            if 'import publish' in code:raise OSError(errno.ESTALE,'write outcome unknown')
            return original(argv,code,**kwargs)
        self.rpc.transport=transport
        with self.assertRaises(UnknownMutation):watch()
        self.assertEqual(read(self.j.state_path)['status'],'UNKNOWN')
        self.assertEqual(self.sleeps,[])
        self.assertFalse(any(n.startswith('READ_ONLY_OBSERVATION_INTERRUPTION') for n,v in self.events))

    def test_heartbeat_reply_loss_is_read_back_not_republished(self):
        self.heartbeat_fault='after';self.run_watch()
        actions=[c[0] for c in self.calls]
        first=actions.index('PUBLISH')
        self.assertEqual(actions[first:first+2],['PUBLISH','READBACK'])
        self.assertEqual(actions.count('PUBLISH'),4)
        self.assertFalse(any(n.startswith('READ_ONLY_OBSERVATION_INTERRUPTION') for n,v in self.events))

    def test_unknown_heartbeat_stops_all_dependent_release(self):
        self.heartbeat_fault='before'
        with self.assertRaises(UnknownMutation):self.run_watch()
        self.assertEqual([c[0] for c in self.calls],['READ','READ','PUBLISH','READBACK'])
        self.assertFalse(self.sleeps)
        self.assertEqual(read(self.j.state_path)['status'],'UNKNOWN')
        self.assertFalse(any(n.startswith('READ_ONLY_OBSERVATION_INTERRUPTION') for n,v in self.events))

    def test_write_and_readback_timeout_never_retries_cycle(self):
        self.heartbeat_fault='after';self.unresolved_readback=True
        with self.assertRaises(UnknownMutation):self.run_watch()
        self.assertEqual([c[0] for c in self.calls].count('PUBLISH'),1)
        self.assertFalse(self.sleeps)

    def test_release_timeout_preserves_intent_no_repeated_release(self):
        self.release_fault=True
        with self.assertRaises(UnknownMutation):self.run_watch()
        self.assertEqual([c[0] for c in self.calls].count('RELEASE'),1)
        self.assertTrue((self.local/'MODEL_START_RELEASE_INTENT.json').exists())
        self.assertFalse(self.sleeps)

    def test_auth_timeout_is_not_read_retry(self):
        self.receipts={'START_NEURAL.json':{},'A40_PRETRAIN_RELEASE.json':{}}
        self.auth_fault=True;self.mono=301
        with self.assertRaises(subprocess.TimeoutExpired):self.run_watch()
        self.assertEqual(self.auths,1)
        self.assertFalse(self.sleeps)
        self.assertFalse(any(n.startswith('READ_ONLY_OBSERVATION_INTERRUPTION') for n,v in self.events))

    def test_terminal_never_renews_or_releases(self):
        self.status='F';self.run_watch()
        self.assertEqual([c[0] for c in self.calls],['READ','READ'])

    def test_unclassified_rpc_is_effect_bearing(self):
        def timeout(*args,**kwargs):raise subprocess.TimeoutExpired('fixture',40)
        self.rpc.transport=timeout
        with self.assertRaises(UnknownMutation):self.rpc('vanda','nothing',{},'new-unreviewed-label')
        self.assertEqual(read(self.j.state_path)['intent']['effect'],'MUTATION')

    def test_no_false_registration_or_second_binding(self):
        before=self.owner.previous.rpc
        for field,value in [('prior_owner_closed',False),('new_recovery_authority_adopted',False),('identity',{})]:
            proof=dict(self.proof);proof[field]=value
            with self.assertRaises(PermissionError):cold_owner.bind_cold_owner(self.owner,self.rpc,validate_new_registration=lambda:proof)
        with self.assertRaises(PermissionError):cold_owner.bind_cold_owner(self.owner,self.rpc,validate_new_registration=lambda:self.proof)
        self.assertIs(self.owner.previous.rpc,before)

    def test_observation_mismatch_never_publishes(self):
        state=self.owner.fresh(1,'fixture-job')
        values=self.values(dict(state['view'],observed_epoch=999))
        with self.assertRaises(UnknownMutation):self.rpc('vanda','',values,'source-A40-same-owner-existing-job-heartbeat')
        self.assertEqual([c[0] for c in self.calls],['READ','READ'])

    def test_first_update_release_timeout_is_not_reissued(self):
        self.release_fault='FIRST_UPDATE'
        with self.assertRaises(UnknownMutation):self.run_watch()
        self.assertEqual([c[0] for c in self.calls].count('RELEASE'),2)
        self.assertTrue((self.local/'FIRST_UPDATE_RELEASE_INTENT.json').exists())
        self.assertEqual(self.sleeps,[30])
        self.assertFalse(any(n.startswith('READ_ONLY_OBSERVATION_INTERRUPTION') for n,v in self.events))

    def test_running_native_identity_mismatch_blocks_every_write(self):
        original=self.owner.fresh
        def wrong(*args,**kwargs):
            state=original(*args,**kwargs);state['rows']['vanda']['jobs']['fixture-job']['project']='OTHER'
            return state
        self.owner.fresh=wrong
        with self.assertRaises(PermissionError):self.run_watch()
        self.assertEqual([c[0] for c in self.calls],['READ','READ'])

    def test_immutable_request_and_lease_preserved_by_watch(self):
        paths=[self.directory/'REQUEST.json',self.root/'OUTPUT_BINDING.private.json',self.lease]
        before={p:p.read_bytes() for p in paths}
        self.run_watch()
        self.assertEqual(before,{p:p.read_bytes() for p in paths})


# Restrict collection to this class's integration tests: the imported
# fixture is used only for setup and must not silently duplicate unit evidence.
def load_tests(loader, standard_tests, pattern):
    names=[n for n in ColdOwnerIntegration.__dict__ if n.startswith('test_')]
    return unittest.TestSuite(ColdOwnerIntegration(n) for n in names)

if __name__=='__main__':unittest.main()
