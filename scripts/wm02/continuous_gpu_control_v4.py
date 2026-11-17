"""Finite32 V4 nonphysical streamed requests; one selected SHIFT7 architecture.

Uses the original real GPU worker, compiler and strict Receiver. No actuator,
MuJoCo, live sensor, new training or student state transition is implemented.
"""
from contextlib import closing
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
from scripts.pi0292.durable import write,OwnerLock
from scripts.pi0294.e1.journal import Journal
from scripts.sla.prepare_r3_stream_v4 import build
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.continuous_r3_v4 import CRITERIA,evaluate
from go2arx5_vla.world_model.storage import digest
from go2arx5_vla.world_model.hybrid_nonphysical_v3 import Receiver,read,wait_receipt
from go2arx5_vla.world_model.hybrid_ipc_storage_v31 import initialize,check_size
from go2arx5_vla.world_model.async_request_archive_v1 import AsyncRequestArchive
from go2arx5_vla.world_model.verified_observation_v1 import VerifiedObservation
from go2arx5_vla.world_model.sla_dense_policy_v5 import compile_proposal
from go2arx5_vla.world_model.mfree_recovery_v2 import register
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import dirs,close


def run(spec):
    grant=read_pin(spec['resource_authorization']);r1=read_pin(spec['R1_result'])
    if (spec['physics'] or spec['new_physical_requests_budget']!=0 or spec['new_optimizer_updates_budget']!=0
            or spec['new_complete_generations_budget']!=32 or spec['unified_requests_budget']!=32
            or spec['selected_combo']!='SHIFT7_SINGLE' or r1['status']!='R1_FULL_PASS'
            or r1['selected_architecture']!='SHIFT7_SINGLE' or not grant['user_granted'] or grant['stage']!='R3'
            or grant['max_requests']!=32 or grant['max_generations']<32 or grant['native_requests']!=0):
        raise PermissionError('EXACT_AUTHORIZED_SINGLE_MODEL32_AFTER_SEALED_R1_REQUIRED')
    root=Path(spec['registered_root']);start=time.monotonic();pin=lambda p:dict(path=str(p),sha256=digest(p))
    write(root/'CRITERIA.private.json',CRITERIA)
    bank_pin=build(spec,root/'stream-bank');bank=read_pin(bank_pin)
    write(root/'BANK_BINDING.private.json',bank_pin)
    if len(bank['records'])!=32:raise ValueError('FIXED32_STREAM_REQUIRED')
    live=initialize(root);(root/'requests').mkdir();child=root/'gpu-worker';dirs(child);clive=live/'gpu-worker';clive.mkdir();bus=clive/'bus';bus.mkdir()
    child_spec=dict(spec,registered_root=str(child),role='MFREE_SHIFT7_R3_GPU_WORKER_V4',live_root=str(clive),ipc_root=str(bus),archive_all_branches=True,
        parent_params=spec['parent_params'],compilation_cache_dir=str(child/'cache'))
    binding=read_pin(spec['site_binding']);binding['paths']['compilation_cache_dir']=str((child/'cache').resolve())
    write(child/'SITE_BINDING.private.json',binding);child_spec['site_binding']=pin(child/'SITE_BINDING.private.json')
    register(spec,[(child,child_spec)]);journal=Journal(root/'control/journal',root.name)
    queue=AsyncRequestArchive(journal,max_pending=4,max_bytes=4*1024**2)
    stopped=lambda:(root/'STOP.json').exists()or time.time()>=spec['deadline_epoch']-60
    rows=[];peak_live=0;process=None
    def wait(path,seconds=90):
        return wait_receipt(path,alive=lambda:process.poll()is None,stopped=stopped,timeout_s=seconds)
    def launch():
        with (child/'logs/worker.log').open('xb')as log:
            p=subprocess.Popen([spec['python'],'-B',str(Path(spec['source_root'])/'scripts/wm02/close_goal_bootstrap_v1.py'),
                '--hybrid-ipc-worker-v3',str(child/'EXECUTION_SPEC.json')],env={**os.environ,**spec['gpu_environment']},stdout=log,stderr=subprocess.STDOUT)
        write(child/'LAUNCH_RECEIPT.private.json',dict(pid=p.pid,proc_start=Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split()[19],epoch=time.time()))
        return p
    try:
        # One-shot process launch is owned by the already registered controller.
        with closing(OwnerLock(root/'control/supervisor-owner.lock')):
            process=launch();ready=wait(child/'READY.private.json',600);cold_release=time.monotonic();warm_origin=None
            for i,entry in enumerate(bank['records']):
                release=cold_release if i==0 else warm_origin+(i-1)*.5
                ticket=queue.reserve('R3-%02d'%i,dict(index=i,input=entry['input'],branch_upper_bound=1),max_bytes=1024**2,timeout_s=30,stopped=stopped)
                while time.monotonic()<release:
                    queue.require_dispatch(ticket)
                    if stopped():raise TimeoutError('R3_STOP_BEFORE_DELIVERY')
                    time.sleep(max(0.,min(.005,release-time.monotonic())))
                queue.require_dispatch(ticket);capture=time.monotonic();lateness=capture-release
                # Read exactly this scheduled recorded frame; no future-frame
                # prefetch or camera-latency claim. File delivery is timed too.
                raw=Path(entry['input']['path']).read_bytes();available=time.monotonic()
                folder=clive/('request-%02d'%i);folder.mkdir();input_pin=dict(path=str(folder/'INPUT.npz'),sha256=entry['input']['sha256'])
                observation=VerifiedObservation(input_pin,raw);(folder/'INPUT.npz').write_bytes(observation.serialized)
                data=observation.values_for(input_pin)
                receiver=Receiver(folder/'receiver',entry['episode']+'-r3-%02d'%i,entry['native_step'],entry['previous_reference'],
                    dict(kind='RECORDED_NATIVE_COMMAND_NOT_SLA_FEEDBACK',bank=bank_pin,index=i),spec['request_bindings'],
                    operator_stop=root/'STOP.json',observation=observation,cache_compiled=True)
                request=receiver.request(input_pin,entry['language'],entry['noise'],clock_mode='WALL_CLOCK_STRICT')
                write(folder/'REQUEST.private.json',request);queue.require_dispatch(ticket)
                sent=time.monotonic();write(bus/'REQUEST.json',dict(request=request,live_path=str(receiver.root/'live.json'),response_root=str(folder),delivery_id=1,branch_barriers=False))
                response=wait(folder/'RESPONSE.json');response_time=time.monotonic();ack=None;error=None;consumption=None;guards=0
                queue.require_dispatch(ticket)
                try:
                    ack=receiver.install(request,response)
                except (ValueError,RuntimeError,TimeoutError)as failure:error=str(failure)
                terminal=time.monotonic()
                if ack:
                    try:consumption=receiver.consume(10);receiver.finish()
                    except (ValueError,RuntimeError,TimeoutError)as failure:
                        error=str(failure);receiver.cancel('STRICT_WALL_GUARD')
                else:receiver.cancel('UNACCEPTED_STRICT_RESPONSE')
                # Raw completed output remains evidence even when cold JIT
                # expired. Offline compilation below never installs that output.
                raw_output=folder/'SHIFT7_OUTPUT.npz'
                full=False;compiler_error=None;transport=False
                if raw_output.exists():
                    with np.load(raw_output,allow_pickle=False)as z:
                        saved={k:z[k].copy()for k in ('normalized','physical_H40')}
                    transport=all(np.isfinite(a).all()for a in saved.values())
                    if response.get('action'):
                        with np.load(response['action']['path'],allow_pickle=False)as z:
                            transport=transport and all(z[k].dtype==a.dtype and z[k].tobytes()==a.tobytes()for k,a in saved.items())
                    from scripts.pi0292.eval_contract import matrix10
                    from go2arx5_vla.world_model.task_learning_student_v1 import StudentProposalRejected
                    try:
                        compile_proposal(raw_output,digest(raw_output),source_from_target=data['policy_from_world'],query_state10=data['state_eef10'],
                            actual_TCP=np.linalg.inv(data['policy_from_world'])@matrix10(data['state_eef10']),consumed_reference_TCP=data['accepted_reference_TCP'],base_hold=np.r_[np.zeros(6),1.]);full=True
                    except StudentProposalRejected as failure:compiler_error=str(failure)
                # No failed/late response may have emitted or installed evidence.
                ttl='TTL' in (error or '') or 'TTL' in response.get('failure','')
                if not ack and list(receiver.root.glob('install-*')):guards+=1
                if not ack and receiver.state['emitted_atoms']:guards+=1
                if not transport:raise ValueError('R3_RAW_OUTPUT_TRANSPORT_CHANGED')
                write(folder/'R3_CONTEXT.private.json',dict(index=i,entry=entry,cached_output=False,clock_mode='WALL_CLOCK_STRICT',new_native_requests=0))
                expected={'INPUT.npz':input_pin['sha256'],'SHIFT7_OUTPUT.npz':digest(raw_output)}
                if response.get('action'):expected['ACTION.npz']=response['action']['sha256']
                if ack:expected['receiver/compiled-0001.npz']=read(receiver.root/'install-0001.json')['bundle']['mechanical_H40_pin']['sha256']
                seal_begin=time.monotonic();queue.submit(ticket,folder,root/'requests'/('request-%02d'%i),expected_pins=expected)
                row=dict(index=i,family=entry['family'],episode=entry['episode'],anchor=entry['anchor'],phase_score_only=entry['phase_score_only'],noise=entry['noise'],
                    cold=i==0,scheduled_monotonic=release,dispatch_monotonic=sent,release_lateness_s=lateness,
                    reservation_wait_capacity_s=ticket.capacity_wait_s,file_delivery_s=available-capture,
                    input_available_to_request_s=request['created_monotonic']-available,input_available_to_terminal_s=terminal-available,
                    capture_to_terminal_s=terminal-capture,IPC_to_response_s=response_time-sent,
                    model_branch_s=response.get('branch_wall_s',[]),ACK=ack,TTL_rejected=ttl,error=error,
                    original_worker_rejection=response.get('failure'),full_H40_pass=full,compiler_error=compiler_error,
                    raw_output_transport_exact=transport,raw_evidence_without_action=not bool(response.get('action')),
                    guard_violation_count=guards,consumption=consumption,emitted_mock_atoms=receiver.state['emitted_atoms'],
                    seal_enqueue_s=time.monotonic()-seal_begin,archive_pending=queue.pending_count,
                    actual_generations=response['new_generations_this_request'],new_updates=0,new_visits=0,new_native_requests=0,new_integrations=0)
                rows.append(row);peak_live=max(peak_live,check_size(live));write(root/'REQUEST_RECORDS.private.json',rows)
                write(root/'goal_state.json',dict(phase='R3_32_NONPHYSICAL',completed_requests=len(rows),actual_generations=sum(r['actual_generations']for r in rows),new_updates=0,new_visits=0,new_native_requests=0,goal_met=False))
                if i==0:warm_origin=time.monotonic()+.5
                if guards:raise RuntimeError('GUARD_VIOLATION_STOP_NEW_DISPATCH')
            write(bus/'FINISH.json',dict(reason='FIXED32_COMPLETE'))
            process.wait(timeout=30)
            if process.returncode:raise RuntimeError('R3_GPU_CHILD_NONZERO_NO_REPLAY')
            archived=queue.close(timeout_s=120);score=evaluate(rows,archived)
            worker=read(child/'RESULT.private.json')
            result=dict(schema='R3_CONTINUOUS_SHIFT7_V4_ACTUAL',exit_code=0,status='R3_REALTIME_PASS'if score['GO']else'BLOCKED_REALTIME',source_sha=spec['source_sha'],
                selected_architecture='SHIFT7_SINGLE',unified_requests=len(rows),actual_generations=worker['attempted_generations'],completed_generations=worker['completed_generations'],
                new_updates=0,new_visits=0,new_native_requests=0,new_integrations=0,new_grasp_or_task_episodes=0,
                score=score,archive=archived,archive_receipts=queue.completed,peak_live_bytes=peak_live,
                input_bank=bank_pin,criteria=pin(root/'CRITERIA.private.json'),worker_result=pin(child/'RESULT.private.json'),worker_ready=ready,
                recorded_RGB_not_live_camera=True,mock_atoms_not_native_or_wallclock_consumption=True,full_physics_control_lifecycle_qualified=False,
                wall_s=time.monotonic()-start,records=pin(root/'REQUEST_RECORDS.private.json'),goal_met=False,native_admission=False)
            write(root/'RESULT.private.json',result);return result
    except BaseException:
        if process is not None and process.poll()is None:
            write(root/'STOP.json',dict(reason='CONTROLLER_FAILURE_STOP_OWNED_NEW_RUN'))
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=30)
        try:queue.close(timeout_s=5)
        except Exception:pass
        raise


if __name__=='__main__':
    spec=read(sys.argv[2]);root=Path(spec['registered_root']);code=1
    try:
        Journal(root/'control/journal',root.name).once('EXECUTE-R3-V4',
            dict(spec_sha256=digest(sys.argv[2])),lambda:run(spec))
        code=0
    except BaseException:
        write(root/'FAILURE.private.json',dict(failure=traceback.format_exc(),automatic_retry=False,new_native_requests=0))
        raise
    finally:close(spec,code)
