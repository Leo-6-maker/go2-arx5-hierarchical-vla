"""One reserved E0 source episode; reuse original episode and protected relay."""
import argparse
import json
import os
from pathlib import Path
import resource
import time

from scripts.pi0292.durable import OwnerLock, read, write
from . import episode, relay_teacher
from .e0_scope import RUN, DEADLINE, LIMITS, validate_entry
from .native_dataset import sha
from .r3_teacher import ProtectedTeacher
from .scene_cards import digest


class E0ProtectedTeacher(ProtectedTeacher):
    def charge(self, count, *, invocation):
        if invocation:
            if self.ledger['shadow_integrations']+200 > LIMITS['shadow_integrations']:
                raise PermissionError('E0_SHADOW_BUDGET')
            self.ledger['shadow_advance_invocations'] += 1
            self.ledger['shadow_integrations'] += 200
            self.ledger['shadow_pending_reservation'] = 200
            write(self.root/'RESOURCE_LEDGER.json', self.ledger)


def classify(result, s1, route_reject=None):
    error = result.get('error') or ''
    if result['status'] == 'SAFETY_STOP_PRESERVED' or s1.get('live_native_failures'):
        return 'ACTUAL_SAFETY_FAIL', 'LIVE_NATIVE_SAFETY', False
    if 'S1_' in error and any(x in error for x in ('MISMATCH', 'STALE', 'SKIPPED', 'REPLAY')):
        return 'ENGINEERING_INCOMPLETE', 'S1_INTEGRITY', False
    if s1.get('pre_execution_rejections', 0):
        return 'VALID_TASK_FAIL', 'S1_PRE_EXECUTION_REJECT', True
    if route_reject and route_reject.get('original_gate', {}).get('decision') == 'REJECT':
        return 'VALID_TASK_FAIL', 'ORIGINAL_REFERENCE_GATE_REJECT', True
    if result['status'] == 'VALID_COMPLETE':
        success = result.get('score', {}).get('persistent_success') is True
        return ('SUCCESS' if success else 'VALID_TASK_FAIL'), ('COMPLETE_SUCCESS' if success else 'OBJECT_TASK_RESULT'), True
    return 'ENGINEERING_INCOMPLETE', 'ENGINEERING_OR_UNKNOWN', False


def run(root, card_path):
    root, card_path = Path(root), Path(card_path)
    m = validate_entry(root, profile='4080-simulation')
    if (root/'E0_LIVE_SAFETY_STOP.json').exists():
        raise PermissionError('E0_REAL_SAFETY_STOP_NO_FURTHER_PHYSICS')
    card = read(card_path)
    panel = read(root/'E0_PANEL_MANIFEST.json')
    original = next((r for r in panel['conditions'] if r['attempt_id'] == card['id']), None)
    if not original or digest({k:v for k,v in card.items() if not k.startswith('relay_request_')}) != original['execution_card_sha256']:
        raise PermissionError('E0_EXACT_PRESEALED_CONDITION')
    ready = read(root/'E0_REMOTE_READY.json')
    if (ready['run_uuid'] != RUN or ready['global_step'] != 2048
            or ready['status'] != 'RESIDENT_WAIT_FRESH_REAL_OBSERVATION'
            or not time.time() < ready['allocation_end_epoch']-1800):
        raise PermissionError('E0_ADMITTED_MODEL_READY_AND_CLOSEOUT_MARGIN_REQUIRED')
    deadline = min(DEADLINE, ready['allocation_end_epoch']-1800)
    lock = OwnerLock(root/'E0_CORE_OWNER.lock')
    ledger_path = root/'RESOURCE_LEDGER.json'
    ledger = read(ledger_path)
    if (ledger['reservations'] or ledger['main_or_diagnostic_starts']+1 > 16
            or ledger['main_or_diagnostic_integrations']+16000 > 256000
            or ledger['RGB_pairs']+641 > 10256
            or ledger['Core_CPU_s']+3600 > 86400):
        lock.close()
        raise PermissionError('E0_MAIN_RESERVATION_CAP')
    folder = root/'episodes'/card['id']
    reservation = dict(schema='PI0294_LIVE_RESERVATION_V1', fixture=False, id=card['id'],
        card=card, card_sha256=digest(card), source_sha=m['code']['vla_commit'],
        root=str(root), output=str(folder), deadline_epoch=deadline,
        cpus=sorted(os.sched_getaffinity(0))[:3], reservation=dict(physics_starts=1,
            physics_steps=16000, rgb_pairs=641, cpu_s=3600,
            memory_bytes=8*1024**3, output_bytes=512*1024**2))
    ledger['reservations'][card['id']] = reservation
    write(ledger_path, ledger)
    start = time.process_time()
    result = None
    teacher_factory, relay_factory = episode.make_teacher, relay_teacher.RelayTeacher
    engines = []

    def guard():
        if time.time() >= deadline or time.process_time()-start >= 3600:
            raise TimeoutError('E0_RESERVED_EPISODE_BOUNDARY')

    def make_teacher(c, initial, engine):
        engines.append(engine)
        return teacher_factory(c, initial, engine)

    def make_relay(t, r, f, c):
        return E0ProtectedTeacher(relay_factory(t,r,f,c), engines[0], root, folder,
            ledger, guard, c['family']+'-'+c['task'])

    episode.make_teacher, relay_teacher.RelayTeacher = make_teacher, make_relay
    try:
        result = episode.run_episode(card, folder, read(root/'INPUT_BINDINGS.private.json')['asset_root'],
            m['code']['vla_commit'], reservation)
        s1 = read(folder/'S1_RESULT.json') if (folder/'S1_RESULT.json').is_file() else {}
        rejected = read(folder/'E0_MODEL_REJECT.json') if (folder/'E0_MODEL_REJECT.json').is_file() else None
        outcome, stage, trustworthy = classify(result, s1, rejected)
        adjudication = dict(status=outcome, failure_stage=stage, pipeline_trustworthy=trustworthy,
            attempt_id=card['id'], scene_group=card['scene_group'], split=card['split'], task=card['task'],
            initial_state_sha256=card['initial_state_sha256'], result_sha256=sha(folder/'RESULT.json'),
            duration_s=result['physics_steps']*.002, main_integrations=result['physics_steps'],
            RGB_pairs=result['rgb_pairs'], S1_rejections=s1.get('pre_execution_rejections',0),
            live_native_failures=s1.get('live_native_failures',0),
            maximum_live_shadow_difference=s1.get('maximum_live_shadow_difference'),
            query_and_video_audit='OWNER_SAVED_EVIDENCE_CLOSEOUT_PENDING',
            selected='NONE', deployment='NONE')
        write(folder/'E0_ADJUDICATION.json', adjudication)
        if outcome == 'ACTUAL_SAFETY_FAIL' or not trustworthy:
            write(root/'E0_LIVE_SAFETY_STOP.json', dict(reason=outcome,stage=stage,attempt_id=card['id']))
        return adjudication
    finally:
        episode.make_teacher, relay_teacher.RelayTeacher = teacher_factory, relay_factory
        ledger['main_or_diagnostic_starts'] += 1
        ledger['main_or_diagnostic_integrations'] += result['physics_steps'] if result else 16000
        ledger['RGB_pairs'] += result['rgb_pairs'] if result else 641
        ledger['Core_CPU_s'] += time.process_time()-start
        ledger['reservations'].pop(card['id'])
        ledger['stages'].append(dict(phase=card['id'],status=result['status'] if result else 'ENGINEERING_FAILURE_PRESERVED'))
        write(ledger_path, ledger)
        lock.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--card', required=True)
    a = p.parse_args()
    print(json.dumps(run(a.root, a.card)), flush=True)
