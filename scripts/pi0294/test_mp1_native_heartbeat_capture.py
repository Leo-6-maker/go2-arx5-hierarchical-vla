from copy import deepcopy
import json
from pathlib import Path
import tempfile
import signal
import time
import unittest
from unittest.mock import patch

from .mp1 import native_heartbeat_capture as capture
from .mp1.owner_heartbeat import HeartbeatLost


class NativeHeartbeatCaptureTests(unittest.TestCase):
    def test_600_seconds_are_actually_sampled_after_io_not_filled_from_one_read(self):
        clock=[100.];forced=[];emitted=[];guards=[]
        def reader(root,run,force_revalidate=False):
            forced.append(force_revalidate);clock[0]+=.001
            return dict(server_revalidation=force_revalidate,clock_sampled_after_read=True,
                revalidated_clock_epoch=1000+clock[0],cached_clock_epoch=1000+clock[0],
                heartbeat_written=False,watchdog_s=300,heartbeat=dict(run_uuid=run,observed_epoch=1000+clock[0]-.1))
        def sleep(seconds):clock[0]+=seconds
        rows=capture.collect(Path('/unused'),'source-pair-01',lambda:guards.append(True),emitted.append,
            reader=reader,clock=lambda:clock[0],sleep=sleep)
        self.assertGreaterEqual(rows[-1]['monotonic_s']-rows[0]['monotonic_s'],600)
        self.assertEqual(len(rows),81);self.assertEqual(rows,emitted)
        self.assertEqual(len(guards),2*len(rows));self.assertGreater(sum(forced),1)
        self.assertTrue(all(row['read_duration_s']>0 for row in rows))

    def test_heartbeat_loss_is_terminal_and_partial_samples_are_not_discarded(self):
        emitted=[];calls=[];clock=[0.]
        def reader(root,run,force_revalidate=False):
            calls.append(True)
            if len(calls)>1:raise HeartbeatLost(dict(heartbeat_written=False))
            return dict(server_revalidation=False,clock_sampled_after_read=True,cached_clock_epoch=1000,
                heartbeat_written=False,watchdog_s=300,heartbeat=dict(run_uuid=run,observed_epoch=990))
        def sleep(seconds):clock[0]+=seconds
        with self.assertRaises(HeartbeatLost):capture.collect('/unused','run',lambda:None,emitted.append,
            reader=reader,clock=lambda:clock[0],sleep=sleep)
        self.assertEqual(len(emitted),1);self.assertEqual(len(calls),2)

    def test_no_clock_tolerance_write_or_identity_exception(self):
        proof=dict(server_revalidation=False,clock_sampled_after_read=True,cached_clock_epoch=1000,
            heartbeat_written=False,watchdog_s=300,heartbeat=dict(run_uuid='run',observed_epoch=990))
        for change in ['future','expired','before_read','wrong_run','writer','watchdog']:
            q=deepcopy(proof)
            if change=='future':q['heartbeat']['observed_epoch']=1000.000001
            if change=='expired':q['heartbeat']['observed_epoch']=699.999999
            if change=='before_read':q['clock_sampled_after_read']=False
            if change=='wrong_run':q['heartbeat']['run_uuid']='other'
            if change=='writer':q['heartbeat_written']=True
            if change=='watchdog':q['watchdog_s']=301
            with self.subTest(change=change),self.assertRaises(PermissionError):
                capture.collect('/unused','run',lambda:None,lambda row:None,
                    reader=lambda *a,**kw:q,clock=lambda:1.)

    def test_invalid_capture_schedules_fail_before_io(self):
        for value in [0,-1,float('nan'),float('inf'),True]:
            with self.subTest(value=value),self.assertRaises(ValueError):
                capture.collect('/unused','run',lambda:None,lambda row:None,duration_s=value)
        with self.assertRaises(ValueError):
            capture.collect('/unused','run',lambda:None,lambda row:None,interval_s=8)

    def test_native_entry_requires_n1_before_creating_any_capture_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'RUN_MANIFEST.json').write_text('{}');directory=root/'pbs/alloc-06'
            with patch.object(capture,'resolve',return_value=None),self.assertRaisesRegex(PermissionError,'SEALED_N1'):
                capture.capture(root,directory,{},lambda:None)
            self.assertFalse(directory.exists())

    def test_capture_checks_actual_native_environment_not_a_supplied_dictionary(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);directory=root/'pbs/alloc-06';directory.mkdir(parents=True)
            (root/'RUN_MANIFEST.json').write_text('{}')
            request=dict(submit_count=6,run_uuid=root.name,source_sha='a'*40,
                run_manifest_sha256=capture.sha(root/'RUN_MANIFEST.json'))
            (directory/'REQUEST.json').write_text(json.dumps(request))
            allocation=dict(job_id='123.hopper',hostname='hopper-gpu',request_sha256=capture.sha(directory/'REQUEST.json'))
            (directory/'ALLOCATION_ADMISSION.json').write_text(json.dumps(allocation))
            with patch.dict(capture.os.environ,{'PBS_JOBID':'','USER':'fixture'}),self.assertRaisesRegex(PermissionError,'NO_LOGIN'):
                capture.native_identity(root,directory,allocation,dict(code=dict(vla_commit='a'*40)))

    def test_native_identity_rechecks_owner_project_node_and_no_model_before_capture(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);directory=root/'pbs/alloc-06';directory.mkdir(parents=True)
            (root/'RUN_MANIFEST.json').write_text('{}')
            request=dict(submit_count=6,run_uuid=root.name,source_sha='a'*40,
                run_manifest_sha256=capture.sha(root/'RUN_MANIFEST.json'))
            (directory/'REQUEST.json').write_text(json.dumps(request))
            allocation=dict(job_id='123.hopper',hostname='hopper-gpu',request_sha256=capture.sha(directory/'REQUEST.json'),
                status='PASS_MP1_OWN_SOURCE_PBS',allocation_start_epoch=0,allocation_end_epoch=10**12)
            (directory/'ALLOCATION_ADMISSION.json').write_text(json.dumps(allocation))
            job=dict(job_state='R',project='CFP04-CF-007',Job_Owner='fixture@host',Resource_List=dict(ngpus=1),exec_host='hopper-gpu/0')
            with patch.dict(capture.os.environ,{'PBS_JOBID':'123.hopper','USER':'fixture'}),patch.object(capture.socket,'gethostname',return_value='hopper-gpu'):
                for change in ['good','foreign_project','foreign_user','foreign_node','terminal','model_ready']:
                    jobs={'123.hopper':deepcopy(job)}
                    if change=='foreign_project':jobs['123.hopper']['project']='other'
                    if change=='foreign_user':jobs['123.hopper']['Job_Owner']='other@host'
                    if change=='foreign_node':jobs['123.hopper']['exec_host']='other/0'
                    if change=='terminal':jobs['123.hopper']['job_state']='F'
                    model=root/'pbs/MODEL_STATE.json'
                    if change=='model_ready':model.write_text(json.dumps(dict(job_id='123.hopper')))
                    with self.subTest(change=change),patch('scripts.pi0292.allocation_admission.owned_jobs',return_value=jobs):
                        if change=='good':self.assertEqual(capture.native_identity(root,directory,allocation,dict(code=dict(vla_commit='a'*40)))['job_id'],'123.hopper')
                        else:
                            with self.assertRaises(PermissionError):capture.native_identity(root,directory,allocation,dict(code=dict(vla_commit='a'*40)))

    def test_sampling_stall_cannot_count_as_continuous_600_seconds(self):
        clock=[0.];emitted=[]
        def reader(root,run,force_revalidate=False):
            return dict(server_revalidation=False,clock_sampled_after_read=True,cached_clock_epoch=1000,
                heartbeat_written=False,watchdog_s=300,heartbeat=dict(run_uuid=run,observed_epoch=990))
        def sleep(seconds):clock[0]+=301
        with self.assertRaisesRegex(PermissionError,'SAMPLING_GAP'):
            capture.collect('/unused','run',lambda:None,emitted.append,reader=reader,clock=lambda:clock[0],sleep=sleep)
        self.assertEqual(len(emitted),1)

    def test_native_entry_preserves_failure_journal_and_restores_timer_without_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'RUN_MANIFEST.json').write_text('{}');directory=root/'pbs/alloc-06';directory.mkdir(parents=True)
            now=time.time();allocation=dict(allocation_end_epoch=now+7200)
            identity=dict(job_id='fixture',hostname='fixture',request_sha256='b'*64,
                manifest_sha256=capture.sha(root/'RUN_MANIFEST.json'),source_sha='a'*40)
            calls=[]
            def failed_collect(root,run,guard,emit,**kwargs):
                calls.append(kwargs);guard();emit(dict(sequence=0,fixture=True));raise TimeoutError('FIXTURE_READ_STALLED')
            handler=signal.getsignal(signal.SIGALRM)
            with patch.object(capture,'resolve',return_value=dict(new_experiment_cutoff_epoch=now+7200)),patch.object(capture,'native_identity',return_value=identity),patch.object(capture,'collect',side_effect=failed_collect):
                with self.assertRaisesRegex(TimeoutError,'FIXTURE_READ_STALLED'):
                    capture.capture(root,directory,allocation,lambda:None)
                with self.assertRaises(FileExistsError):capture.capture(root,directory,allocation,lambda:None)
            self.assertEqual(calls,[dict(duration_s=600,interval_s=7.5)])
            self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0],0);self.assertEqual(signal.getsignal(signal.SIGALRM),handler)
            self.assertEqual(json.loads((directory/'heartbeat-capture/READS.jsonl').read_text())['sequence'],0)
            terminal=json.loads((directory/'heartbeat-capture/TERMINAL.json').read_text())
            self.assertEqual(terminal['status'],'FAILED_NATIVE_CAPTURE_PRESERVED_NO_AUTOMATIC_RESTART')
            self.assertFalse((directory/capture.FILENAME).exists())


if __name__=='__main__':unittest.main()
