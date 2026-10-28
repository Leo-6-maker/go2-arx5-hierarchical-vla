"""Bounded observation of existing updates; no extra optimizer steps or queries."""
from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time

from scripts.pi0292.durable import read, write
from scripts.pi0294.native_dataset import sha
from .gradient_audit import aggregated, equivalent, legacy


PLAN='plans/TRAINING_PERFORMANCE_PLAN.json'


def plan():
    return dict(schema='PI0294_PV1_TRAINING_PERFORMANCE_V1',
        first_local_updates=100,CUDA_equivalence_updates=3,aggregate_only_after_local_update=100,
        profiler_wait_updates=2,profiler_warmup_updates=1,profiler_active_updates=2,
        profiler_record_shapes=False,profiler_with_stack=False,profiler_profile_memory=False,
        maximum_operator_summary_rows=200,GPU_sample_interval_s=5,maximum_GPU_samples=120,
        auxiliary_CPU_precharge_upper_s=120,additional_optimizer_updates=0,
        effective_batch=32,microbatch=2,gradient_accumulation=16,
        DataLoader_workers=0,sample_sequence_changed=False,loss_changed=False,
        full_native_resume_checks_unchanged=True)


def configured(root):
    root=Path(root);path=root/PLAN
    if path.is_symlink():raise PermissionError('PV1_EXACT_PRESEALED_PERFORMANCE_PLAN_REQUIRED')
    if not path.exists():
        if (root/'RUN_MANIFEST.json').exists() and PLAN in read(root/'RUN_MANIFEST.json')['execution']['sealed_inputs']:
            raise PermissionError('PV1_MISSING_PRESEALED_PERFORMANCE_PLAN')
        return False
    m=read(root/'RUN_MANIFEST.json')
    if (path.is_symlink() or path.stat().st_mode&0o222
            or m['execution']['sealed_inputs'].get(PLAN)!=sha(path) or read(path)!=plan()):
        raise PermissionError('PV1_EXACT_PRESEALED_PERFORMANCE_PLAN_REQUIRED')
    return True


def profiler_unavailable(error):
    message=str(error).lower()
    if any(s in message for s in ('out of memory','device-side assert','illegal memory')):return False
    return any(s in message for s in ('cupti_error_insufficient_privileges','cupti is not available',
        'kineto is not available','no profiler activities','cupti_error_not_initialized'))


class Window:
    """Separate local profiling progress from authoritative global sample cursors."""
    def __init__(self,start):
        if type(start) is not int or not 0<=start<4096:raise ValueError('EXACT_COMMITTED_START_STEP')
        self.start=start;self.completed=0;self.qualifications=0
    def mode(self):
        if self.completed<3:return 'DUAL_EQUIVALENCE_RETURN_LEGACY'
        if self.completed<100:return 'LEGACY_BASELINE'
        if self.qualifications!=3:raise PermissionError('PV1_CUDA_GRADIENT_EQUIVALENCE_REQUIRED')
        return 'AGGREGATED_SCALAR_TRANSPORT'
    def commit(self,global_step,cursor):
        if global_step!=self.start+self.completed+1 or cursor!=global_step*32:
            raise PermissionError('PV1_PERFORMANCE_CANNOT_ADVANCE_OR_REWRITE_SAMPLE_CURSOR')
        self.completed+=1


class Observer:
    def __init__(self,root,evidence,allocation,torch,ledger,*,start_step):
        from scripts.pi0294.dec1.a40_allocation import STATUS as A40_STATUS
        if allocation.get('status')==A40_STATUS:
            from scripts.pi0294.dec1.a40_training_context import require_training
            require_training(root,allocation)
        if (not configured(root) or allocation.get('status') not in ('PASS_PV1_OWN_NATIVE_H200_ALLOCATION',A40_STATUS)
                or not os.environ.get('PBS_JOBID') or allocation['job_id']!=os.environ['PBS_JOBID']
                or not time.time()<allocation['allocation_end_epoch']):
            raise PermissionError('PV1_PERFORMANCE_ONLY_IN_EXISTING_LEGAL_ALLOCATION')
        self.root=Path(root);self.directory=Path(evidence)/'PERFORMANCE'
        self.directory.mkdir(mode=0o700,exist_ok=False)
        self.torch=torch;self.allocation=allocation;self.window=Window(start_step)
        self.profiler=None;self.handles=[];self.phase_wall={};self.gpu_status='NOT_STARTED'
        self.stopping=threading.Event();self.thread=None;self.profile_status='NOT_STARTED'
        self.previous_loader_wait=0.;self.first_wall=time.monotonic();self.grad_times=[]
        from scripts.pi0294.native_train import rng_identity
        self.rng_identity=rng_identity
        self.synchronize_for_qualification=torch.cuda.synchronize
        ledger.cpu(plan()['auxiliary_CPU_precharge_upper_s'])
        write(self.directory/'PLAN.json',plan())

    @contextmanager
    def scope(self,name):
        if self.window.completed>=100:yield;return
        began=time.monotonic()
        context=self.torch.profiler.record_function('PV1/'+name) if self.profiler is not None else nullcontext()
        try:
            with context:yield
        finally:self.phase_wall[name]=self.phase_wall.get(name,0.)+time.monotonic()-began

    def start(self,model):
        torch=self.torch;p=plan()
        def summary(profiler):
            rows=[]
            for event in profiler.key_averages():
                rows.append(dict(key=event.key,count=event.count,CPU_total_us=event.cpu_time_total,
                    CPU_self_us=event.self_cpu_time_total,
                    device_total_us=getattr(event,'device_time_total',None),
                    device_self_us=getattr(event,'self_device_time_total',None)))
            rows.sort(key=lambda r:r['CPU_total_us'],reverse=True)
            write(self.directory/'PROFILER_OPERATOR_SUMMARY.json',dict(
                status='ACTUAL_OPERATOR_TIMINGS_NOT_END_TO_END_UTILIZATION_PERCENT',
                rows=rows[:p['maximum_operator_summary_rows']],total_operator_keys=len(rows),
                CUDA_event_data_present=any(r['device_total_us'] is not None and r['device_total_us']>0 for r in rows),
                missing_device_events_are_not_zero_utilization=True,
                summed_nested_or_overlapping_times_are_not_wall_percentages=True))
        try:
            self.profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=2,warmup=1,active=2,repeat=1),on_trace_ready=summary,
                record_shapes=False,profile_memory=False,with_stack=False)
            self.profiler.start();self.profile_status='RUNNING_BOUNDED_FIVE_UPDATE_WINDOW'
        except RuntimeError as exc:
            if not profiler_unavailable(exc):raise
            self.profiler=None;self.profile_status='UNAVAILABLE_'+type(exc).__name__
        except (AttributeError,ValueError) as exc:
            self.profiler=None;self.profile_status='UNAVAILABLE_'+type(exc).__name__
        for name in ('backbone','action_head'):
            module=getattr(model,name,None)
            if module is None:continue
            stack=[]
            def enter(module,args,*,label=name,stack=stack):
                context=self.scope(label.upper());context.__enter__();stack.append(context)
            def leave(module,args,output,*,stack=stack):
                if stack:stack.pop().__exit__(None,None,None)
            self.handles.extend([module.register_forward_pre_hook(enter),module.register_forward_hook(leave,always_call=True)])
        try:device=str(torch.cuda.get_device_properties(0).uuid)
        except (RuntimeError,AttributeError):device=''
        if not re.fullmatch(r'GPU-[0-9a-fA-F-]{36}',device):
            self.gpu_status='UNAVAILABLE_ALLOCATED_DEVICE_UUID_NOT_PROVEN';return
        self.thread=threading.Thread(target=self._gpu_samples,args=(device,),daemon=True)
        self.thread.start()

    def _gpu_samples(self,device):
        self.gpu_status='SAMPLING_ONLY_THIS_ALLOCATED_GPU_UUID'
        path=self.directory/'GPU_SAMPLES.jsonl'
        for index in range(120):
            if self.stopping.is_set() or self.window.completed>=100 or time.time()>=self.allocation['allocation_end_epoch']-2730:break
            began=time.time()
            process=None
            try:
                process=subprocess.Popen(['nvidia-smi','-i',device,
                    '--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw',
                    '--format=csv,noheader,nounits'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                stat=Path('/proc',str(process.pid),'stat')
                if stat.exists():
                    start=stat.read_text().rsplit(')',1)[1].split()[19]
                    record=self.root/'aux_worker_processes'/('performance-smi-'+str(os.getpid())+'-'+str(index)+'.json')
                    manifest=read(self.root/'RUN_MANIFEST.json')
                    with record.open('x') as f:json.dump(dict(pid=process.pid,proc_start=start,
                        operation='PV1_OWN_ALLOCATION_GPU_TELEMETRY_ONLY',job_id=self.allocation['job_id'],
                        owner='EXISTING_LOCAL_PC',run_uuid=self.root.name,profile=manifest['execution'].get('profile','hopper-neural'),
                        source_sha=manifest['code']['vla_commit'],manifest_sha256=sha(self.root/'RUN_MANIFEST.json')),f)
                    record.chmod(0o400)
                try:stdout,stderr=process.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill();process.communicate();raise TimeoutError('OWN_NVIDIA_SMI_TIMEOUT')
                if process.returncode:raise RuntimeError('OWN_NVIDIA_SMI_QUERY_FAILED')
                with path.open('a') as f:f.write(json.dumps(dict(epoch=began,csv=stdout.strip(),local_updates=self.window.completed))+'\n')
            except (OSError,RuntimeError,TimeoutError):
                self.gpu_status='UNAVAILABLE_NATIVE_TELEMETRY_QUERY';break
            finally:
                if process is not None and process.poll() is None:
                    process.kill();process.communicate()
            if self.stopping.wait(5):break
        else:self.gpu_status='BOUNDED_120_SAMPLES_FINISHED'

    def gradients(self,named_parameters):
        parameters=list(named_parameters);mode=self.window.mode();durations={}
        dual=mode=='DUAL_EQUIVALENCE_RETURN_LEGACY'
        if dual and self.window.qualifications!=self.window.completed:
            raise PermissionError('PV1_CUDA_QUALIFICATION_ONCE_PER_EXISTING_UPDATE')
        rng_before=self.rng_identity() if dual else None
        versions=[(id(p.grad),getattr(p.grad,'_version',None)) for _,p in parameters] if dual else None
        with self.scope('GRADIENT_AUDIT'):
            if dual:self.synchronize_for_qualification()
            began=time.monotonic()
            if mode=='AGGREGATED_SCALAR_TRANSPORT':result=aggregated(parameters,self.torch)
            else:result=legacy(parameters,self.torch)
            durations[result['mode']]=time.monotonic()-began
            if dual:
                self.synchronize_for_qualification()
                began=time.monotonic();candidate=aggregated(parameters,self.torch)
                durations[candidate['mode']]=time.monotonic()-began
                if not equivalent(result,candidate):raise ValueError('PV1_CUDA_GRADIENT_AUDIT_EXACT_EQUIVALENCE_FAILED')
                if (rng_before!=self.rng_identity() or versions!=[(id(p.grad),getattr(p.grad,'_version',None)) for _,p in parameters]):
                    raise ValueError('PV1_GRADIENT_AUDIT_CHANGED_RNG_OR_GRADIENT_STORAGE')
                self.window.qualifications+=1
                with (self.directory/'ACTUAL_CUDA_GRADIENT_EQUIVALENCE.jsonl').open('a') as f:
                    f.write(json.dumps(dict(global_step_before=self.window.start+self.window.completed,
                        exact_norms=True,gradient_tensors=result['gradient_tensors_checked'],
                        legacy_host_reads=result['host_reads'],aggregated_host_reads=candidate['host_reads'],
                        gradient_object_and_version_unchanged=True,RNG_fingerprint_before=rng_before,RNG_unchanged=True,
                        timing='ORDERED_SAME_GRADIENT_AUDITS_EACH_STARTS_AFTER_QUALIFICATION_SYNC_NOT_FULL_TRAINING_SPEEDUP',
                        no_gradient_writes_no_RNG_no_extra_update=True,wall_s=durations))+'\n')
        self.grad_times=durations
        return result['norm_squared']

    def after_step(self,trainer,step,step_s):
        self.window.commit(step,trainer.consumed)
        if self.window.completed<=100:
            wait=trainer.measured_loader.wait_s
            row=dict(global_step=step,actual_consumed_cursor=trainer.consumed,local_updates=self.window.completed,
                synchronized_step_s=step_s,parent_loader_wait_s_delta=wait-self.previous_loader_wait,
                phase_host_wall_s=self.phase_wall,gradient_audit_wall_s=self.grad_times,
                peak_GPU_allocated_bytes=self.torch.cuda.max_memory_allocated(),
                nested_phase_times_are_not_additive=True)
            with (self.directory/'FIRST100_UPDATES.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            self.previous_loader_wait=wait;self.phase_wall={}
        if self.profiler is not None:
            try:
                self.profiler.step()
                if self.window.completed==5:
                    self.profiler.stop();self.profiler=None;self.profile_status='COMPLETED_BOUNDED_FIVE_UPDATE_WINDOW'
            except RuntimeError as exc:
                if not profiler_unavailable(exc):raise
                self.profile_status='UNAVAILABLE_NATIVE_PROFILER_RUNTIME'
                try:self.profiler.stop()
                except RuntimeError:pass
                self.profiler=None
        if self.window.completed==100:
            self.stopping.set()
            elapsed=time.monotonic()-self.first_wall
            write(self.directory/'FIRST100_TERMINAL.json',dict(
                status='ACTUAL_FIRST100_FORMAL_UPDATES_NOT_EXTRA_BENCHMARK_TRAINING',
                start_global_step=self.window.start,end_global_step=step,actual_consumed_cursor=trainer.consumed,
                effective_samples=3200,elapsed_including_trainer_setup_profiling_and_audit_s=elapsed,
                effective_samples_per_s=3200/elapsed,formal_updates_per_s=100/elapsed,
                parent_loader_wait_s=trainer.measured_loader.wait_s,CUDA_exact_gradient_qualifications=self.window.qualifications,
                scientific_recipe_unchanged=True,additional_optimizer_updates=0))
            for handle in self.handles:handle.remove()
            self.handles=[]

    def close(self):
        self.stopping.set()
        if self.thread is not None:self.thread.join(timeout=3)
        for handle in self.handles:handle.remove()
        if self.profiler is not None and time.time()<self.allocation['allocation_end_epoch']:
            self.profiler.stop();self.profiler=None
        write(self.directory/'TERMINAL.json',dict(status='PERFORMANCE_OBSERVATION_NOT_TASK_OR_SPEEDUP_SUCCESS',
            actual_local_committed_updates=self.window.completed,CUDA_exact_gradient_qualifications=self.window.qualifications,
            profiler_status=self.profile_status,GPU_sampling_status=self.gpu_status,
            actual_wall_s=time.monotonic()-self.first_wall,additional_optimizer_updates=0,
            scientific_recipe_sampler_RNG_and_full_resume_unchanged=True))


def observed_trainer(base,observer):
    class Observed(base):
        def _prepare_inputs(self,*args,**kwargs):
            with observer.scope('HOST_TO_DEVICE'):return super()._prepare_inputs(*args,**kwargs)
        def compute_loss(self,*args,**kwargs):
            with observer.scope('NATIVE_FORWARD_LOSS_AND_CHECKS'):return super().compute_loss(*args,**kwargs)
        def training_step(self,*args,**kwargs):
            with observer.scope('FORWARD_BACKWARD_AND_MATURE_TRACE'):return super().training_step(*args,**kwargs)
    return Observed
