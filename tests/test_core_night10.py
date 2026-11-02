"""Native-free verification of the thin 10h process DAG; no GPU/simulator."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

SOURCE = Path(__file__).parents[1] / 'scripts/pi0294/night10/supervisor.py'
spec = importlib.util.spec_from_file_location('core_night10', SOURCE)
night = importlib.util.module_from_spec(spec)
spec.loader.exec_module(night)
sys.path.insert(0, str(SOURCE.parents[3]))
from scripts.pi0294.night10 import policy_review


class TestNight10(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cpu = sorted(os.sched_getaffinity(0))[0]

    def command(self, name, code='0', *, after=(), needs=(), admission=None):
        receipt = self.root / (name + '.json')
        script = ('import json, pathlib, sys\n'
                  "pathlib.Path(sys.argv[1]).write_text(json.dumps({'status':'CLOSED','valid':4,'successes':0}))\n"
                  f'sys.exit({code})\n')
        return dict(id=name, lane='cpu', after=list(after), needs=list(needs),
                    command=[sys.executable, '-c', script, str(receipt)], cwd=str(self.root),
                    max_seconds=60, write_gib=0, cpus=[self.cpu],
                    admission=admission or [],
                    receipt=dict(path=str(receipt), equals=dict(status='CLOSED')))

    def plan(self, tasks):
        return dict(schema=night.SCHEMA, duration_s=36000, max_cpu_parallel=2,
                    disk_margin_gib=30, gpu1_training_admitted=False, tasks=tasks)

    def test_cpu_monitor_covers_window_but_preserves_summary_reserve_and_gpu_bound(self):
        monitor=self.command('monitor');monitor['max_seconds']=35900
        night.check_plan(self.plan([monitor]))
        monitor['max_seconds']=35941
        with self.assertRaisesRegex(ValueError,'FINITE_TASK_RUNTIME'):
            night.check_plan(self.plan([monitor]))
        monitor.update(max_seconds=35900,lane='gpu_inference')
        with self.assertRaisesRegex(ValueError,'FINITE_TASK_RUNTIME'):
            night.check_plan(self.plan([monitor]))

    def test_scientific_failure_does_not_stop_independent_branches_or_soft_after(self):
        failed = self.command('failed', '4')
        independent = self.command('independent')
        after_failure = self.command('after_failure', after=['failed'])
        cannot_use_bad_model = self.command('cannot_use_bad_model', needs=['failed'])
        p = self.plan([failed, independent, after_failure, cannot_use_bad_model])
        night.check_plan(p)
        with mock.patch.object(night.shutil, 'disk_usage', return_value=shutil._ntuple_diskusage(100*1024**3,0,100*1024**3)):
            with mock.patch.object(night, 'available_memory_gib', return_value=100):
                runner = night.Supervisor(p, self.root/'night')
                self.assertEqual(runner.run(poll_s=1), 0)
        self.assertEqual(runner.states['failed']['state'], 'FAILED')
        self.assertEqual(runner.states['independent']['state'], 'COMPLETE')
        self.assertEqual(runner.states['after_failure']['state'], 'COMPLETE')
        self.assertEqual(runner.states['cannot_use_bad_model']['state'], 'SKIPPED')
        self.assertTrue((self.root/'night/SUMMARY.json').is_file())

    def test_same_eval_can_return_zero_policy_success_without_killing_chain(self):
        run = self.command('native_eval')
        p = self.plan([run])
        with mock.patch.object(night.shutil, 'disk_usage', return_value=shutil._ntuple_diskusage(100*1024**3,0,100*1024**3)):
            with mock.patch.object(night, 'available_memory_gib', return_value=100):
                runner = night.Supervisor(p, self.root/'night')
                runner.run(poll_s=1)
        self.assertEqual(runner.states['native_eval']['state'], 'COMPLETE')
        self.assertEqual(json.loads((self.root/'native_eval.json').read_text())['successes'], 0)

    def test_gpu1_training_requires_native_positive_admission_not_permission(self):
        t = self.command('gpu1')
        t.update(lane='gpu_train', gpu=1, gpu_uuid='GPU-test', min_free_mib=2000, readiness=[dict(path=str(self.root/'cold.json'), equals=dict(status='CLOSED'))])
        with self.assertRaises(PermissionError):
            night.check_plan(self.plan([t]))
        p = self.plan([t]);p['gpu1_training_admitted'] = True
        with self.assertRaises(PermissionError):
            night.check_plan(p)
        t['admission'] = [dict(path=str(self.root/'native.json'), equals=dict(training_admitted=True))]
        night.check_plan(p)

    def test_physics_requires_real_cold_gate_and_owner(self):
        t = self.command('physics')
        t.update(lane='physics', gpu=0, gpu_uuid='GPU-test', min_free_mib=2048)
        with self.assertRaises(ValueError):
            night.check_plan(self.plan([t]))
        t['readiness'] = [dict(path=str(self.root/'cold.json'), equals=dict(lease='CLOSED'))]
        t['preflight'] = [sys.executable, '-c', 'import sys;sys.exit(0)']
        night.check_plan(self.plan([t]))

    def test_missing_new_dagger_data_skips_without_model_load(self):
        t = self.command('dagger', admission=[dict(path=str(self.root/'qualified.json'),minimum=dict(new_expert_suffixes=1))])
        p = self.plan([t]);runner = night.Supervisor(p, self.root/'night')
        self.assertTrue(runner.tick())
        self.assertEqual(runner.states['dagger']['state'], 'SKIPPED')
        self.assertFalse((self.root/'dagger.json').exists())

    def test_gpu1_physics_needs_an_explicit_native_inference_admission(self):
        t=self.command('physics')
        t.update(lane='physics',gpu=1,gpu_uuid='GPU-test',min_free_mib=13312,
            readiness=[dict(path=str(self.root/'cold.json'),equals=dict(status='CLOSED'))],
            preflight=[sys.executable,'-c','pass'])
        with self.assertRaises(ValueError):night.check_plan(self.plan([t]))
        t['admission']=[dict(path=str(self.root/'capacity.json'),equals=dict(capacity_pass=True,completed_generations=96))]
        night.check_plan(self.plan([t]))

    def test_capacity_waits_for_a_full_stable_interval_after_foreign_work_ends(self):
        t=self.command('inference')
        t.update(lane='gpu_inference',gpu=1,gpu_uuid='GPU-test',min_free_mib=13312,
            gpu_stable_seconds=60,readiness=[dict(path=str(self.root/'cold.json'),equals=dict(status='CLOSED'))])
        clock=[0.];gpu=[('GPU-test',15000,[])]
        runner=night.Supervisor(self.plan([t]),self.root/'night',gpu_probe=lambda _:gpu[0],time_source=lambda:clock[0])
        with mock.patch.object(night.shutil,'disk_usage',return_value=shutil._ntuple_diskusage(100*1024**3,0,100*1024**3)),mock.patch.object(night,'available_memory_gib',return_value=100):
            self.assertFalse(runner.capacity('inference'))
            clock[0]=59.;self.assertFalse(runner.capacity('inference'))
            gpu[0]=('GPU-test',2000,[123]);self.assertFalse(runner.capacity('inference'))
            clock[0]=60.;gpu[0]=('GPU-test',15000,[]);self.assertFalse(runner.capacity('inference'))
            clock[0]=120.;self.assertTrue(runner.capacity('inference'))

    def test_unknown_running_effect_is_not_replayed_on_resume(self):
        t = self.command('train')
        p = self.plan([t]);runner = night.Supervisor(p, self.root/'night')
        runner.change('train','LAUNCHING')
        with self.assertRaises(PermissionError):
            night.Supervisor(p,self.root/'night',resume=True)

    def test_existing_receipt_is_not_reused_to_start_write(self):
        t = self.command('task')
        (self.root/'task.json').write_text('{"status":"CLOSED"}')
        runner = night.Supervisor(self.plan([t]),self.root/'night')
        with mock.patch.object(night.shutil,'disk_usage',return_value=shutil._ntuple_diskusage(100*1024**3,0,100*1024**3)):
            with mock.patch.object(night,'available_memory_gib',return_value=100):
                self.assertTrue(runner.tick())
            self.assertFalse(runner.tick())
        self.assertEqual(runner.states['task']['state'],'UNKNOWN')

    def test_policy_review_requires_real_success_and_zero_hard_stop(self):
        cp = self.root/'cp.json';cp.write_text('{"status":"CLOSED"}')
        a = self.root/'valid.json';a.write_text('{"count":{"valid":4,"success":1,"unrun":6,"safety":1}}')
        b = self.root/'good.json';b.write_text('{"count":{"valid":5,"success":3,"unrun":5,"safety":0}}')
        c = dict(schema=policy_review.SCHEMA, criteria=dict(minimum_valid=4,minimum_complete_successes=2,
                      require_zero_hard_safety=True), candidates=[])
        for tag,path in [('old_phase',a),('dagger',b)]:
            c['candidates'].append(dict(id=tag,panel_conditions=10,
                checkpoint=dict(path=str(cp),equals=dict(status='CLOSED')),
                evaluation=dict(path=str(path),fields=dict(valid='count.valid',successes='count.success',
                          unrun='count.unrun',hard_safety='count.safety'))))
        v=policy_review.review(c)
        self.assertEqual(v['provisional_policy'],'dagger')
        self.assertEqual(v['candidates'][0]['status'],'DEVELOPMENT_GATE_FAIL')
        self.assertTrue(v['unseen_confirmation_not_opened'])
        c['candidates'][1]['evaluation']['path']=str(self.root/'missing.json')
        self.assertIsNone(policy_review.review(c)['provisional_policy'])

    def test_cycle_is_rejected(self):
        a = self.command('task_a', after=['task_b']);b = self.command('task_b',after=['task_a'])
        with self.assertRaises(ValueError):night.check_plan(self.plan([a,b]))

    def test_native_child_failure_without_closed_effect_blocks_other_launches(self):
        t=self.command('native','4');t['closure_receipt']=dict(path=str(self.root/'closed.json'),equals=dict(native_effects_closed=True))
        runner=night.Supervisor(self.plan([t,self.command('later',after=['native'])]),self.root/'night')
        class Exited:
            def poll(self):return 4
        runner.processes['native']=Exited();runner.states['native']=dict(state='RUNNING',start_time=0)
        self.assertTrue(runner.finish('native',runner.processes['native']))
        self.assertEqual(runner.states['native']['state'],'UNKNOWN')
        self.assertFalse(runner.tick())

    def test_timeout_is_not_success_even_with_exit_zero_and_valid_receipt(self):
        t=self.command('native');Path(t['receipt']['path']).write_text('{"status":"CLOSED"}')
        runner=night.Supervisor(self.plan([t]),self.root/'night')
        class Exited:
            def poll(self):return 0
        runner.processes['native']=Exited();runner.states['native']=dict(state='STOPPING',start_time=0)
        runner.finish('native',runner.processes['native'])
        self.assertEqual(runner.states['native']['state'],'FAILED')
        self.assertTrue(runner.states['native']['timeout'])

    def test_gate_must_observe_real_native_close(self):
        gate = dict(id='cold_gate',lane='gate',readiness=[dict(path=str(self.root/'native.json'),equals=dict(status='CLOSED'))])
        post = self.command('post',needs=['cold_gate'])
        runner = night.Supervisor(self.plan([gate,post]),self.root/'night')
        self.assertTrue(runner.tick())
        self.assertEqual(runner.states['cold_gate']['state'],'PENDING')
        (self.root/'native.json').write_text('{"status":"CLOSED"}')
        with mock.patch.object(night.shutil, 'disk_usage', return_value=shutil._ntuple_diskusage(100*1024**3,0,100*1024**3)):
            with mock.patch.object(night, 'available_memory_gib', return_value=100):
                runner.run(poll_s=1)
        self.assertEqual(runner.states['post']['state'],'COMPLETE')


if __name__=='__main__':unittest.main()
