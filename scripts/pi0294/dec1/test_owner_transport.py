"""CPU fixtures only; no SSH, PBS, models, shared ledgers or legacy owners."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.pi0292.durable import read, write
from scripts.pi0294.dec1.owner_transport import (
    Journal, JournalFailure, ReadOnlyFailure, UnknownMutation, checked_heartbeat)
from scripts.pi0294.dec1 import owner_heartbeat_publish as remote


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base/'runs'/'fixture-run'
        self.directory = self.root/'pbs'/'alloc-01'
        self.directory.mkdir(parents=True)
        (self.root/'control').mkdir()
        self.journal = self.base/'journal';self.journal.mkdir()
        self.lease = self.base/'lease.json'
        write(self.lease, dict(status='ACTIVE', run_uuid=self.root.name))
        write(self.root/'OUTPUT_BINDING.private.json', {'lease':str(self.lease)})
        write(self.directory/'REQUEST.json', {'source_sha':'a'*40})
        self.identity = dict(run_uuid=self.root.name, job_id='fixture-job', source_sha='a'*40,
                            request_sha256=hashlib.sha256((self.directory/'REQUEST.json').read_bytes()).hexdigest())
        self.now = 1000.;self.mono=0.
        self.view = dict(status='PASS_TWO_PLATFORM_CAMPAIGN_VIEW', observed_epoch=1000.,
                         campaign_jobs=[dict(job_id='fixture-job', state='R')], grant_sha256='b'*64)
        self.ledger = dict(unknown_effects=[],pending_updates=[])
        self.j = self.make_journal();self.writes=0;self.reads=0;self.last=None

    def make_journal(self):
        return Journal(self.journal,self.identity,clock=lambda:self.now,monotonic=lambda:self.mono)

    def publish(self, intent, timeout):
        self.writes+=1;self.last=intent
        self.assertEqual(timeout,30)
        return remote.publish(self.root,self.directory,'fixture-job',self.view,self.ledger,intent,clock=lambda:self.now)

    def readback(self,intent,timeout):
        self.reads+=1;self.assertEqual(timeout,40)
        return remote.readback(self.root,self.directory,'fixture-job',self.view,self.ledger,intent,clock=lambda:self.now)

    def lost_reply(self,intent,timeout):
        self.publish(intent,timeout)
        raise subprocess.TimeoutExpired('fixture',30)

    def test_success_and_next_fresh_intent(self):
        a=self.j.call('HEARTBEAT','HEARTBEAT',self.publish)
        b=self.j.call('HEARTBEAT','HEARTBEAT',self.publish)
        self.assertNotEqual(a['heartbeat']['rpc_intent_id'],b['heartbeat']['rpc_intent_id'])
        self.assertEqual(self.writes,2)
        self.assertEqual(read(self.j.state_path)['status'],'CONFIRMED')

    def test_lost_reply_exact_readback_one_write(self):
        result=self.j.call('HEARTBEAT','HEARTBEAT',self.lost_reply,readback=self.readback)
        self.assertEqual((self.writes,self.reads),(1,1))
        self.assertEqual(result['heartbeat']['rpc_intent_id'],self.last['operation_id'])
        error=read(next(self.journal.glob('*-ERROR.json')))
        self.assertEqual(error['outcome'],'UNKNOWN_WRITE')
        self.assertEqual(error['intent']['phase'],'HEARTBEAT')

    def test_unknown_write_persists_across_process_and_blocks_other_writes(self):
        with self.assertRaises(UnknownMutation):
            self.j.call('HEARTBEAT','HEARTBEAT',self.lost_reply)
        fresh=self.make_journal()
        for kind in ('HEARTBEAT','MUTATION'):
            with self.assertRaises(UnknownMutation):
                fresh.call('NEXT',kind,self.publish)
        self.assertEqual(self.writes,1)
        self.assertEqual(fresh.call('OBSERVE','READ_ONLY',lambda *_:42),42)
        self.assertEqual(read(fresh.state_path)['status'],'UNKNOWN')
        fresh.reconcile_heartbeat(self.readback)
        self.assertEqual((self.writes,self.reads),(1,1))

    def test_missing_remote_write_never_treated_as_zero_or_replayed(self):
        def timeout(intent,limit):
            self.writes+=1
            raise TimeoutError()
        with self.assertRaises(UnknownMutation):
            self.j.call('HEARTBEAT','HEARTBEAT',timeout,readback=self.readback)
        self.assertEqual((self.writes,self.reads),(1,1))
        self.assertEqual(read(self.j.state_path)['status'],'UNKNOWN')

    def test_mismatched_old_heartbeat_does_not_clear_latch(self):
        self.j.call('HEARTBEAT','HEARTBEAT',self.publish)
        def timeout(*_):raise TimeoutError()
        with self.assertRaises(UnknownMutation):
            self.j.call('HEARTBEAT','HEARTBEAT',timeout,readback=self.readback)
        self.assertEqual(self.writes,1)
        self.assertEqual(read(self.j.state_path)['status'],'UNKNOWN')

    def test_readback_timeout_latched_then_explicit_readback(self):
        def timeout(*_):raise TimeoutError()
        with self.assertRaises(UnknownMutation):
            self.j.call('HEARTBEAT','HEARTBEAT',self.lost_reply,readback=timeout)
        self.make_journal().reconcile_heartbeat(self.readback)
        self.assertEqual(self.writes,1)

    def test_stale_readback_and_future_clock_rejected(self):
        with self.assertRaises(UnknownMutation):self.j.call('HEARTBEAT','HEARTBEAT',self.lost_reply)
        for now in (1301.,999.,float('nan')):
            self.now=now;self.view['observed_epoch']=now
            with self.assertRaises(UnknownMutation):self.j.reconcile_heartbeat(self.readback)
        self.assertEqual(self.writes,1)

    def test_read_transport_timeout_only_is_retryable(self):
        def timeout(*_):raise subprocess.TimeoutExpired('fixture',40)
        with self.assertRaises(ReadOnlyFailure):self.j.call('OBSERVE','READ_ONLY',timeout)
        self.assertFalse(self.j.state_path.exists())
        for error in (PermissionError('STOP'),AssertionError('GUARD'),ValueError('BAD_JSON')):
            with self.subTest(error=type(error).__name__):
                def rejected(*_):raise error
                with self.assertRaises(type(error)):
                    self.j.call('OBSERVE','READ_ONLY',rejected)
        self.assertEqual(self.writes,0)

    def test_other_mutation_unknown_never_heartbeat_reconciled(self):
        def timeout(*_):raise TimeoutError()
        with self.assertRaises(UnknownMutation):self.j.call('RELEASE','MUTATION',timeout)
        with self.assertRaises(UnknownMutation):self.j.reconcile_heartbeat(self.readback)
        self.assertEqual(self.reads,0)

    def test_interrupt_persists_without_automatic_readback(self):
        def interrupt(*_):raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.j.call('HEARTBEAT','HEARTBEAT',interrupt,readback=self.readback)
        self.assertEqual(read(self.j.state_path)['status'],'UNKNOWN')
        self.assertEqual(self.reads,0)

    def test_late_success_rejected_and_readonly_reconciled(self):
        def late(intent,limit):
            result=self.publish(intent,limit);self.mono+=31;return result
        self.j.call('HEARTBEAT','HEARTBEAT',late,readback=self.readback)
        self.assertEqual((self.writes,self.reads),(1,1))

    def test_failed_journal_before_dispatch_stops(self):
        with patch('scripts.pi0294.dec1.owner_transport.write',side_effect=OSError('disk')):
            with self.assertRaises(JournalFailure):self.j.call('HEARTBEAT','HEARTBEAT',self.publish)
        self.assertEqual(self.writes,0)

    def test_crash_after_pending_is_conservatively_latched(self):
        with patch.object(self.j,'_event',side_effect=JournalFailure('disk')):
            with self.assertRaises(JournalFailure):self.j.call('HEARTBEAT','HEARTBEAT',self.publish)
        self.assertEqual(self.writes,0)
        with self.assertRaises(UnknownMutation):self.make_journal().call('HEARTBEAT','HEARTBEAT',self.publish)

    def test_foreign_journal_identity_rejected(self):
        self.j.call('HEARTBEAT','HEARTBEAT',self.publish)
        self.identity=dict(self.identity,job_id='other')
        with self.assertRaises(PermissionError):self.make_journal().call('OBSERVE','READ_ONLY',lambda *_:None)

    def test_duplicate_same_remote_intent_is_rejected(self):
        self.j.call('HEARTBEAT','HEARTBEAT',self.publish)
        before=(self.root/'control/OWNER_HEARTBEAT.json').read_bytes()
        with self.assertRaises(PermissionError):self.publish(self.last,30)
        self.assertEqual(before,(self.root/'control/OWNER_HEARTBEAT.json').read_bytes())

    def test_native_guard_rejections_never_write(self):
        self.j.call('HEARTBEAT','HEARTBEAT',self.publish)
        before=(self.root/'control/OWNER_HEARTBEAT.json').read_bytes()
        intent=dict(self.last,operation_id='c'*32)
        cases=[('stale_view',lambda:self.view.update(observed_epoch=880.)),
               ('future_view',lambda:self.view.update(observed_epoch=1001.)),
               ('wrong_job',lambda:self.view.update(campaign_jobs=[])),
               ('unknown',lambda:self.ledger.update(unknown_effects=[{}])),
               ('pending',lambda:self.ledger.update(pending_updates=[{}])),
               ('lease',lambda:write(self.lease,dict(status='CLOSED',run_uuid=self.root.name))),
               ('stop',lambda:write(self.root/'STOP.json',{})),
               ('parent_stop',lambda:write(self.root.parent.parent/'STOP.json',{}))]
        for name,change in cases:
            with self.subTest(name=name):
                self.view.update(observed_epoch=1000.,campaign_jobs=[dict(job_id='fixture-job',state='R')]);self.ledger.update(unknown_effects=[],pending_updates=[])
                write(self.lease,dict(status='ACTIVE',run_uuid=self.root.name))
                for p in (self.root/'STOP.json',self.root.parent.parent/'STOP.json'):p.unlink(missing_ok=True)
                change()
                with self.assertRaises(PermissionError):self.publish(intent,30)
                self.assertEqual(before,(self.root/'control/OWNER_HEARTBEAT.json').read_bytes())

    def test_stop_after_publication_blocks_ack_and_readback(self):
        original=remote.write
        def stop_after(path,value):
            original(path,value);original(self.root/'STOP.json',{})
        with patch.object(remote,'write',side_effect=stop_after):
            with self.assertRaises(UnknownMutation):
                self.j.call('HEARTBEAT','HEARTBEAT',self.publish,readback=self.readback)
        self.assertEqual(read(self.j.state_path)['status'],'UNKNOWN')

    def test_exact_readback_predicate_rejects_wrong_identity_and_unknowns(self):
        value=self.j.call('HEARTBEAT','HEARTBEAT',self.publish)
        for field in ('run_uuid','job_id','source_sha','request_sha256','rpc_intent_id'):
            with self.subTest(field=field):
                bad=json.loads(json.dumps(value));bad['heartbeat'][field]='wrong'
                with self.assertRaises(PermissionError):checked_heartbeat(bad,self.last,now=self.now)
        for field in ('unknown_effects','pending_updates'):
            bad=dict(value);bad[field]=[{}]
            with self.assertRaises(PermissionError):checked_heartbeat(bad,self.last,now=self.now)

    def test_readback_does_not_touch_heartbeat_bytes_or_mtime(self):
        self.j.call('HEARTBEAT','HEARTBEAT',self.publish)
        path=self.root/'control/OWNER_HEARTBEAT.json';before=(path.read_bytes(),path.stat().st_mtime_ns)
        self.readback(self.last,40)
        self.assertEqual(before,(path.read_bytes(),path.stat().st_mtime_ns))

    def test_invalid_effect_and_symlink_journal_rejected(self):
        with self.assertRaises(ValueError):self.j.call('HEARTBEAT','GUESS',self.publish)
        link=self.base/'link';link.symlink_to(self.journal)
        with self.assertRaises(ValueError):Journal(link,self.identity)


    def test_remote_replay_old_token_after_new_publication_rejected(self):
        self.j.call('HEARTBEAT','HEARTBEAT',self.publish);old=self.last
        self.j.call('HEARTBEAT','HEARTBEAT',self.publish)
        before=(self.root/'control/OWNER_HEARTBEAT.json').read_bytes()
        with self.assertRaises(PermissionError):self.publish(old,30)
        self.assertEqual(before,(self.root/'control/OWNER_HEARTBEAT.json').read_bytes())

    def test_late_remote_start_cannot_renew(self):
        def late(intent,limit):
            self.now+=31;return self.publish(intent,limit)
        with self.assertRaises(UnknownMutation):self.j.call('HEARTBEAT','HEARTBEAT',late)
        self.assertFalse((self.root/'control/OWNER_HEARTBEAT.json').exists())


class BoundedAdapterTests(unittest.TestCase):
    def test_timeout_forwarded_and_one_attempt(self):
        from scripts.pi0294.dec1.owner_rpc import invoke
        calls=[]
        def runner(*args,**kwargs):
            calls.append((args,kwargs));raise subprocess.TimeoutExpired('fixture',kwargs['timeout'])
        with self.assertRaises(subprocess.TimeoutExpired):invoke(['ssh','fixture'],'read()',timeout_s=30,runner=runner)
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0][1]['timeout'],30)

    def test_native_error_and_transport_error_distinct(self):
        from scripts.pi0294.dec1.owner_rpc import invoke
        for code,error in [(255,ConnectionError),(1,RuntimeError)]:
            with self.subTest(code=code):
                def runner(*args,**kwargs):return subprocess.CompletedProcess(args,code,'','fixture failure')
                with self.assertRaises(error):invoke(['ssh','fixture'],'read()',timeout_s=40,runner=runner)

    def test_reply_json_is_not_assumed_success(self):
        from scripts.pi0294.dec1.owner_rpc import invoke
        def runner(*args,**kwargs):return subprocess.CompletedProcess(args,0,'invalid','')
        with self.assertRaises(ValueError):invoke(['ssh','fixture'],'read()',timeout_s=30,runner=runner)

    def test_code_uses_explicit_source_and_no_eval(self):
        from scripts.pi0294.dec1.owner_rpc import program
        text=program('scripts.pi0294.dec1.owner_heartbeat_publish','readback',{'root':'fixture'},'/fixture/capsule/code')
        compile(text,'<fixture>','exec')
        self.assertIn('/fixture/capsule/code',text)
        self.assertNotIn('eval(',text)
        with self.assertRaises(ValueError):program('legacy','write',{},'/fixture')


if __name__=='__main__':unittest.main()
