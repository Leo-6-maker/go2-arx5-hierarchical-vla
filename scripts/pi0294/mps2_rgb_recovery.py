"""Recover exact saved native-state RGB after a recorder-context failure.

No physics or controller executes; original RESULT/arrays remain immutable.
"""
import json,struct
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from .mps2_runtime import FOLDER,require_adoption
from .task_capture_runtime import read,sha,validate_entry
from .task_state import parse_state
from .mps2_worker import FrameRecorder

def run(root,episode):
    from scripts.pi0292.durable import OwnerLock,write
    root=Path(root);manifest,parent=validate_entry(root,read_only=True);require_adoption(root,manifest)
    folder=root/FOLDER/'episodes'/episode;target=folder/'RECORDER_RECOVERY.json'
    if target.exists():return read(target)
    result=read(folder/'RESULT.json');samples=[json.loads(s) for s in (folder/'samples.jsonl').read_text().splitlines()]
    if result['status']!='DATA_RECORDER_INCOMPLETE' or result['native_stops'] or not result.get('recorder_error'):raise PermissionError('MPS2_ONLY_RECORDER_FAILURE_RECOVERY')
    wanted={s['native_step']:i for i,s in enumerate(samples)};packets={}
    with (folder/'state.bin.journal').open('rb') as handle:
        while raw:=handle.read(4):
            n,=struct.unpack('<I',raw);packet=parse_state(handle.read(n))
            if packet['step'] in wanted:packets[wanted[packet['step']]]=packet
    if len(packets)!=len(samples):raise ValueError('MPS2_EXACT_SAVED_PACKETS_FOR_RECOVERY')
    previous_count=result['saved_RGB_pairs'];extra=len(samples)-previous_count
    owner=OwnerLock(parent/'R4_RESOURCE_OWNER.lock')
    try:
        ledger=read(parent/'R4_RESOURCE_LEDGER.json');series=read(root/FOLDER/'SERIES_LEDGER.json')
        if series['RGB_pairs']+extra>1764 or ledger['RGB_pairs']+extra>102560:raise PermissionError('MPS2_RECOVERED_RGB_BUDGET')
        reservation=folder/'RGB_RECOVERY_RESERVATION.json'
        if not reservation.exists():
            ledger['RGB_pairs']+=extra;series['RGB_pairs']+=extra
            write(parent/'R4_RESOURCE_LEDGER.json',ledger);write(root/FOLDER/'SERIES_LEDGER.json',series)
            write(reservation,dict(extra_RGB_pairs=extra,original_result_sha256=sha(folder/'RESULT.json'),initial_pairs=previous_count))
    finally:owner.close()
    recorder=FrameRecorder(folder/'task-scene.private.mjb',folder);pool=ThreadPoolExecutor(max_workers=1)
    try:
        for i,sample in enumerate(samples):
            packet=packets[i]
            if packet['step']!=sample['native_step']:raise ValueError('MPS2_RECONSTRUCTION_STEP')
            calibration=folder/'images'/f'{i:06d}-calibration.json'
            images=[folder/'images'/f'{i:06d}-{c}.png' for c in ('body_rgb','wrist_rgb')]
            if calibration.exists() and all(p.exists() for p in images):
                old=read(calibration)
                if any(old[c]['timestamp_s']!=packet['time_s'] for c in ('body_rgb','wrist_rgb')):raise ValueError('MPS2_RECOVERY_CLOCK_CHANGED')
                continue
            if calibration.exists() or any(p.exists() for p in images):raise ValueError('MPS2_PARTIAL_FRAME_PRESERVED_FOR_RECONCILIATION')
            pool.submit(recorder.save,i,packet).result(timeout=30)
        pool.submit(recorder.close).result(timeout=10)
    finally:pool.shutdown(wait=True)
    files=[p for p in (folder/'images').iterdir() if p.is_file()]
    proof=dict(status='PASS_FIXED_ACTUAL_STATE_RGB_RECONSTRUCTION',episode=episode,source_sha=manifest['code']['vla_commit'],
        original_result_sha256=sha(folder/'RESULT.json'),samples_sha256=sha(folder/'samples.jsonl'),journal_sha256=sha(folder/'state.bin.journal'),
        native_model_sha256=sha(folder/'task-scene.private.mjb'),RGB_pairs=len(samples),files={p.name:sha(p) for p in files},
        source='SAME_IMMUTABLE_ACTUAL_NATIVE_PACKETS_AND_ORIGINAL_CAMERA_MODEL',display='HEADLESS_XVFB_GLFW_CONTEXT',
        original_recorder_failure_preserved=True,new_physics=0,new_teacher_queries=0,rerun_task=False)
    write(target,proof);return proof
