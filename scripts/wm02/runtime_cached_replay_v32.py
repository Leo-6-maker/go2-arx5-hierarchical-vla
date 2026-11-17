"""Finite CPU replay of sealed V3.1 outputs; zero neural/native calls.

Two input boundaries are measured separately. FILE_APPLICATION_CACHE_EMPTY
does not flush the kernel page cache; DELIVERED_MEMORY_RECORDED_RGB excludes
the explicitly recorded preparation read and is not a live camera result.
"""
from contextlib import closing
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from scripts.pi0292.durable import OwnerLock, write
from scripts.pi0294.e1.journal import Journal
from go2arx5_vla.world_model.async_request_archive_v1 import AsyncRequestArchive
from go2arx5_vla.world_model.hybrid_ipc_storage_v31 import initialize, check_size
from go2arx5_vla.world_model.hybrid_nonphysical_v3 import Receiver, read
from go2arx5_vla.world_model.storage import digest
from go2arx5_vla.world_model.verified_observation_v1 import VerifiedObservation


def run(spec):
    if (spec['physics'] or spec['new_physical_requests_budget'] != 0
            or spec['new_complete_generations_budget'] != 0
            or spec['new_optimizer_updates_budget'] != 0
            or spec['new_training_visits_budget'] != 0
            or os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            or os.environ.get('JAX_PLATFORMS') != 'cpu'):
        raise PermissionError('CPU_CACHED_ZERO_GENERATION_ZERO_NATIVE_ONLY')
    root = Path(spec['registered_root'])
    old = Path(spec['sealed_v31_root'])
    if digest(old/'RESULT.private.json') != spec['sealed_v31_result_sha256']:
        raise ValueError('SEALED_V31_TERMINAL_CHANGED')
    rows = read(old/'REQUEST_RECORDS.private.json')
    if digest(old/'REQUEST_RECORDS.private.json') != spec['sealed_v31_records_sha256']:
        raise ValueError('SEALED_V31_RECORDS_CHANGED')
    ordinary = [r for r in rows if r['kind'] == 'normal']
    if len(ordinary) != 16:
        raise ValueError('EXACT_SAVED16_ORDINARY_SCOPE_REQUIRED')
    live = initialize(root)
    (root/'requests').mkdir()
    journal = Journal(root/'control/journal', root.name)
    queue = AsyncRequestArchive(journal, max_pending=4, max_bytes=4*1024**2)
    records = []
    peak_live = 0
    started = time.monotonic()
    def stopped():
        return (root/'STOP.json').exists() or time.time() >= spec['deadline_epoch']-30
    try:
        for mode in ('FILE_APPLICATION_CACHE_EMPTY', 'DELIVERED_MEMORY_RECORDED_RGB'):
            for baseline in ordinary:
                index = baseline['index']
                label = ('file' if mode.startswith('FILE') else 'memory') + '-%02d'%index
                queue.require_dispatch()
                # Includes every already active, reserved or queued request.
                ticket = queue.reserve(label, dict(mode=mode, baseline_index=index,
                    cached_output=True, new_generations=0), max_bytes=1024**2,
                    timeout_s=min(30, max(.001, spec['deadline_epoch']-time.time()-30)), stopped=stopped)
                source = old/'requests'/('request-%02d'%index)
                archive_pin = baseline['archive']
                if digest(archive_pin['path']) != archive_pin['sha256']:
                    raise ValueError('SEALED_BASELINE_ARCHIVE_CHANGED')
                sealed = read(archive_pin['path'])
                for item in sealed['files']:
                    if digest(source/item['file']) != item['sha256']:
                        raise ValueError('SEALED_BASELINE_BYTES_CHANGED')
                oq = read(source/'REQUEST.private.json')
                input_pin = dict(path=str(source/'INPUT.npz'), sha256=oq['input']['sha256'])
                output_pin = dict(path=str(source/'ACTION.npz'), sha256=baseline['ACK']['action_sha256'])
                folder = live/label
                folder.mkdir()
                # Cached branch bytes are prepared outside measured input/ACK.
                prep_start = time.monotonic()
                for item in sealed['files']:
                    name = item['file']
                    if name == 'ACTION.npz' or name.endswith('_OUTPUT.npz'):
                        (folder/name).write_bytes((source/name).read_bytes())
                prepared = Path(input_pin['path']).read_bytes() if mode.startswith('DELIVERED') else None
                prepare_s = time.monotonic()-prep_start
                peak_live = max(peak_live, check_size(live))
                queue.require_dispatch(ticket)
                begin = time.monotonic()
                observation = (VerifiedObservation.from_file(input_pin) if prepared is None
                               else VerifiedObservation(input_pin, prepared))
                input_s = time.monotonic()-begin
                bindings = dict(oq['bindings'], source_sha=spec['source_sha'])
                receiver = Receiver(folder/'receiver', oq['episode']+'-cpu-'+label, oq['native_step'],
                    oq['previous_reference'], oq['previous_evidence'], bindings,
                    observation=observation, cache_compiled=True)
                request = receiver.request(input_pin, oq['language'], oq['noise_id'])
                response = dict(request_id=request['request_id'], bindings=request['bindings'],
                    action=dict(path=str(folder/'ACTION.npz'), sha256=output_pin['sha256']))
                queue.require_dispatch(ticket)
                ack = receiver.install(request, response)
                ack_s = time.monotonic()-begin
                install = read(receiver.root/'install-0001.json')
                original = read(source/'receiver/install-0001.json')
                wire = install['bundle']['eef_wire']
                old_wire = original['bundle']['eef_wire']
                if {k:v for k,v in wire.items() if k != '_pi0294'} != {k:v for k,v in old_wire.items() if k != '_pi0294'}:
                    raise AssertionError('ORIGINAL_EEF_WIRE_CHANGED')
                with np.load(receiver.root/'compiled-0001.npz', allow_pickle=False) as new, np.load(source/'receiver/compiled-0001.npz', allow_pickle=False) as prior:
                    for key in ('reference_H40','nominal_H40'):
                        if new[key].dtype != prior[key].dtype or new[key].tobytes() != prior[key].tobytes():
                            raise AssertionError('ORIGINAL_COMPLETE_H40_BITS_CHANGED')
                consumption = receiver.consume(baseline['emitted_atoms'])
                for atom in range(baseline['emitted_atoms']):
                    emitted = read(receiver.root/('emitted-0001-%02d.json'%atom))
                    previous = read(source/'receiver'/('emitted-0001-%02d.json'%atom))
                    for key in ('EEF10_mechanical','source_native_s','end_native_step'):
                        if emitted[key] != previous[key]:
                            raise AssertionError('ORIGINAL_ATOM_OR_CLOCK_CHANGED')
                if consumption['full_P10']:
                    receiver.finish()
                else:
                    receiver.cancel('SAVED_PARTIAL7_CPU_REPLAY')
                # This write is evidence preparation, not camera acquisition.
                (folder/'INPUT.npz').write_bytes(observation.serialized)
                write(folder/'REQUEST.private.json', request)
                write(folder/'RESPONSE.json', dict(response, cached_replay=True, new_generations=0))
                write(folder/'CPU_REPLAY.private.json', dict(mode=mode, baseline_index=index,
                    baseline_request_id=oq['request_id'], cached_output=True, native_integrations=0))
                seal_begin = time.monotonic()
                queue.submit(ticket, folder, root/'requests'/label, expected_pins={
                    'INPUT.npz':input_pin['sha256'], 'ACTION.npz':output_pin['sha256'],
                    'receiver/compiled-0001.npz':install['bundle']['mechanical_H40_pin']['sha256']})
                record = dict(index=len(records), baseline_index=index, input_mode=mode,
                    preparation_excluded_s=prepare_s, verified_input_s=input_s,
                    cached_input_to_ACK_s=ack_s, seal_enqueue_s=time.monotonic()-seal_begin,
                    archive_pending_at_next_boundary=queue.pending_count, H40_bit_exact=True,
                    EEF_wire_exact=True, atoms_exact=baseline['emitted_atoms'],
                    actual_neural_generations=0, native_requests=0, integrations=0)
                records.append(record)
                peak_live = max(peak_live, check_size(live))
                write(root/'PROGRESS.private.json', dict(completed_cached_replays=len(records),
                    peak_live_bytes=peak_live, new_generations=0, native_requests=0))
        archive = queue.close(timeout_s=max(1., min(120., spec['deadline_epoch']-time.time()-5)))
    except BaseException:
        try:
            queue.close(timeout_s=5)
        except Exception:
            pass
        raise
    if any(k in sys.modules for k in ('jax','torch','mujoco','rclpy')):
        raise AssertionError('MODEL_OR_NATIVE_RUNTIME_IMPORTED_IN_CPU_REPLAY')
    # These are cached engineering measurements, never GPU throughput claims.
    result = dict(schema='R2_CPU_SAVED_OUTPUT_REPLAY_V32', source_sha=spec['source_sha'],
        cached_replays=len(records), original_complete_H40_exact=sum(r['H40_bit_exact'] for r in records),
        mock_atoms=sum(r['atoms_exact'] for r in records), archive=archive,
        archived_request_hashes={k:v for k,v in queue.completed.items()}, peak_live_bytes=peak_live,
        model_or_native_modules_loaded=False, new_generations=0, new_updates=0, new_visits=0,
        new_native_requests=0, new_integrations=0, task_success_denominator=0,
        real_time_qualified=False, live_camera_measured=False, wall_s=time.monotonic()-started)
    write(root/'REQUEST_RECORDS.private.json', records)
    write(root/'RESULT.private.json', result)
    return result


def main():
    spec = read(sys.argv[2])
    root = Path(spec['registered_root'])
    code = 1
    try:
        with closing(OwnerLock(root/'control/supervisor-owner.lock')):
            run(spec)
            code = 0
    except BaseException as error:
        write(root/'FAILURE.private.json', dict(reason=repr(error), traceback=traceback.format_exc(),
            automatic_retry=False, new_generations=0, native_requests=0))
        raise
    finally:
        lease = read(root/'control/LEASE.json')
        lease.update(status='CLOSED', worker_exit_code=code, closed_epoch=time.time())
        write(root/'control/LEASE.json', lease)


if __name__ == '__main__':
    main()
