"""CPU observation-window contracts; not GPU timing or new training."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts.pi0294.pv1 import performance as perf
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.test_mp1_transport_fixture import put
from scripts.pi0294.test_pv1_gradient_audit import Backend,parameters


class PerformanceContracts(unittest.TestCase):
    def test_opt_in_exact_presealed_plan_not_mutable_or_scientific_change(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.assertFalse(perf.configured(root))
            put(root/perf.PLAN,perf.plan(),True)
            manifest=dict(execution=dict(sealed_inputs={perf.PLAN:sha(root/perf.PLAN)}))
            put(root/'RUN_MANIFEST.json',manifest)
            self.assertTrue(perf.configured(root))
            for change in (dict(DataLoader_workers=4),dict(microbatch=4),dict(loss_changed=True)):
                put(root/perf.PLAN,dict(perf.plan(),**change),True)
                manifest['execution']['sealed_inputs'][perf.PLAN]=sha(root/perf.PLAN);put(root/'RUN_MANIFEST.json',manifest)
                with self.assertRaisesRegex(PermissionError,'EXACT_PRESEALED'):perf.configured(root)

    def test_window_never_rewrites_authoritative_committed_cursor(self):
        window=perf.Window(1024)
        for step,cursor in ((1024,32768),(1025,32768),(1026,32832)):
            with self.assertRaises(PermissionError):window.commit(step,cursor)
        self.assertEqual(window.completed,0)
        window.commit(1025,1025*32);self.assertEqual(window.completed,1)

    def test_legacy_first100_and_all_three_exact_native_checks_before_switch(self):
        window=perf.Window(0)
        for step in range(1,101):
            self.assertEqual(window.mode(),'DUAL_EQUIVALENCE_RETURN_LEGACY' if step<=3 else 'LEGACY_BASELINE')
            window.commit(step,step*32)
        with self.assertRaisesRegex(PermissionError,'CUDA_GRADIENT_EQUIVALENCE'):window.mode()
        window.qualifications=3;self.assertEqual(window.mode(),'AGGREGATED_SCALAR_TRANSPORT')
        self.assertEqual(window.completed,100)

    def test_dual_path_returns_original_norms_without_advancing_an_update(self):
        with tempfile.TemporaryDirectory() as folder:
            observer=perf.Observer.__new__(perf.Observer)
            observer.window=perf.Window(0);observer.directory=Path(folder)
            observer.phase_wall={};observer.profiler=None;observer.torch=Backend()
            observer.rng_identity=lambda:dict(CPU_fixture='UNCHANGED_NOT_NATIVE_GPU_PROOF')
            observer.synchronize_for_qualification=lambda:None
            for step in range(1,4):
                values=observer.gradients(parameters(observer.torch))
                self.assertGreater(values['projector'],0);self.assertGreater(values['diffusion'],0)
                self.assertEqual(observer.window.completed,step-1)
                observer.window.commit(step,step*32)
            self.assertEqual(observer.window.qualifications,3)
            rows=(observer.directory/'ACTUAL_CUDA_GRADIENT_EQUIVALENCE.jsonl').read_text().splitlines()
            self.assertEqual(len(rows),3)

    def test_instrumented_trainer_preserves_arguments_outputs_and_parent_cursor(self):
        scopes=[]
        class Base:
            def _prepare_inputs(self,value):return value
            def compute_loss(self,value,**kwargs):return value
            def training_step(self,value,**kwargs):return value
        observer=SimpleNamespace(scope=lambda name:(scopes.append(name) or nullcontext()))
        trainer=perf.observed_trainer(Base,observer)();value=object()
        self.assertIs(trainer._prepare_inputs(value),value)
        self.assertIs(trainer.compute_loss(value,return_outputs=True),value)
        self.assertIs(trainer.training_step(value,num_items_in_batch=32),value)
        self.assertEqual(scopes,['HOST_TO_DEVICE','NATIVE_FORWARD_LOSS_AND_CHECKS','FORWARD_BACKWARD_AND_MATURE_TRACE'])

    def test_observer_cannot_start_a_login_node_gpu_profile(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(PermissionError,'LEGAL_ALLOCATION'):
                perf.Observer(folder,folder,dict(status='NOT_AN_ALLOCATION'),None,None,start_step=0)

    def test_rng_change_blocks_scalar_transport_qualification(self):
        with tempfile.TemporaryDirectory() as folder:
            observer=perf.Observer.__new__(perf.Observer);observer.window=perf.Window(0)
            observer.directory=Path(folder);observer.phase_wall={};observer.profiler=None;observer.torch=Backend()
            calls=iter([dict(RNG='BEFORE'),dict(RNG='CHANGED')]);observer.rng_identity=lambda:next(calls)
            observer.synchronize_for_qualification=lambda:None
            with self.assertRaisesRegex(ValueError,'CHANGED_RNG_OR_GRADIENT_STORAGE'):
                observer.gradients(parameters(observer.torch))
            self.assertEqual(observer.window.qualifications,0);self.assertEqual(observer.window.completed,0)

    def test_profiler_stops_after_five_existing_updates_not_a_new_training_loop(self):
        with tempfile.TemporaryDirectory() as folder:
            calls=[];observer=perf.Observer.__new__(perf.Observer);observer.window=perf.Window(0)
            observer.directory=Path(folder);observer.phase_wall={};observer.grad_times={}
            observer.previous_loader_wait=0.;observer.first_wall=time.monotonic()
            observer.torch=SimpleNamespace(cuda=SimpleNamespace(max_memory_allocated=lambda:1234))
            observer.profiler=SimpleNamespace(step=lambda:calls.append('step'),stop=lambda:calls.append('stop'))
            for step in range(1,6):
                trainer=SimpleNamespace(consumed=step*32,measured_loader=SimpleNamespace(wait_s=step*.1))
                observer.after_step(trainer,step,1.)
            self.assertEqual(calls,['step']*5+['stop']);self.assertIsNone(observer.profiler)
            self.assertEqual(observer.window.completed,5)
            self.assertEqual(len((observer.directory/'FIRST100_UPDATES.jsonl').read_text().splitlines()),5)

    def test_gpu_telemetry_targets_only_proven_device_and_no_persistent_process(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'aux_worker_processes').mkdir()
            put(root/'RUN_MANIFEST.json',dict(code=dict(vla_commit='b'*40)))
            observer=perf.Observer.__new__(perf.Observer);observer.root=root;observer.directory=root
            observer.window=perf.Window(0);observer.allocation=dict(allocation_end_epoch=time.time()+10000,job_id='CPU_FIXTURE')
            observer.stopping=SimpleNamespace(is_set=lambda:False,wait=lambda seconds:True)
            process=SimpleNamespace(pid=999999999,returncode=0,poll=lambda:0,communicate=lambda timeout=None:('fixture, 50, 10, 100, 20',''))
            with patch.object(perf.subprocess,'Popen',return_value=process) as popen:
                observer._gpu_samples('GPU-'+'a'*36)
            args=popen.call_args.args[0]
            self.assertEqual(args[:3],['nvidia-smi','-i','GPU-'+'a'*36])
            self.assertEqual(popen.call_count,1)
            self.assertEqual(len((root/'GPU_SAMPLES.jsonl').read_text().splitlines()),1)

    def test_optional_profiler_cannot_swallow_real_device_or_memory_faults(self):
        self.assertTrue(perf.profiler_unavailable(RuntimeError('CUPTI_ERROR_INSUFFICIENT_PRIVILEGES')))
        for message in ('CUDA out of memory','CUDA device-side assert','illegal memory access','unexpected CUDA failure'):
            self.assertFalse(perf.profiler_unavailable(RuntimeError(message)))

    def test_dangling_or_missing_sealed_plan_never_silently_disables_observation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);put(root/'RUN_MANIFEST.json',dict(execution=dict(sealed_inputs={perf.PLAN:'a'*64})))
            with self.assertRaisesRegex(PermissionError,'MISSING_PRESEALED'):perf.configured(root)
            (root/perf.PLAN).parent.mkdir();(root/perf.PLAN).symlink_to(root/'MISSING')
            with self.assertRaisesRegex(PermissionError,'EXACT_PRESEALED'):perf.configured(root)
