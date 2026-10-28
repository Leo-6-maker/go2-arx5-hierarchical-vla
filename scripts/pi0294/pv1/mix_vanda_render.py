"""Finite source-render hook in the existing manifest/registry/PBS/once ledger.

One CPU allocation, two original initial scenes, four PNGs, zero integrations.
This is neither a new evaluator nor a task/Teacher/WBC qualification.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import uuid

from . import mix_vanda_cohost as c, mix_accounting as a, successor_cd as s

VERSION = 'V5_CD_VANDA_ORIGINAL_SOURCE_RENDER_ONLY_V1'
PROFILE = 'vanda-cd-source-render-qualification'
JOB_NAME = 'pi0294_cd_render'
ROLE = 'PI0294_V5_CD_VANDA_SOURCE_RENDER_01'
LIMITS = dict(ngpus=0, active_cpus=2, walltime_s=480, CPU_s=960,
              constructors=2, RGB_images=4, physics_ticks=0, episode_starts=0,
              model_loads=0, neural_queries=0, optimizer_updates=0,
              output_bytes=8*1024**2)
STATES = ('QSUB_INTENT_OUTCOME_UNKNOWN_NO_RETRY','SUBMITTED_PENDING_NATIVE','NATIVE_TERMINAL')


def need(ok, why):
    if not ok:
        raise PermissionError('V5_VANDA_RENDER_'+why)


def validate_row(row, global_ledger):
    marker = c.validated(global_ledger)
    need(marker is not None, 'ACTUALLY_ADOPTED_SITE_CONTINUATION')
    expected = dict(platform_qualification_version=VERSION, native_platform='VANDA_A40',
        project='CFP04-CF-007', job_name=JOB_NAME, physical_GPU_count=0,
        scope_grant_sha256=s.GRANT_SHA, authority_release=s.TASK_SOURCE,
        T_CD=marker['T_CD'], deadline_epoch=marker['deadline_epoch'],
        CPU_upper_s=960, reserved_H200_s=0, query_limit=0,
        limits=LIMITS)
    need(all(row.get(k) == v and type(row[k]) is type(v) for k,v in expected.items()), 'EXACT_ZERO_GPU_MODEL_AND_INTEGRATION_SCOPE')
    for k in ('source_sha','request_sha256','run_manifest_sha256'):
        need(type(row.get(k)) is str and re.fullmatch('[0-9a-f]{'+str(40 if k=='source_sha' else 64)+'}',row[k]), 'IMMUTABLE_IDENTITY:'+k)
    for k in ('request_id','submission_nonce'):
        try: valid = str(uuid.UUID(row[k])) == row[k]
        except (KeyError,ValueError,TypeError,AttributeError): valid = False
        need(valid, 'UNIQUE_CANONICAL_REQUEST')
    need(row.get('status') in STATES, 'KNOWN_ONCE_LIFECYCLE')
    need(row.get('job_id') is None and row['status'] == STATES[0]
         or type(row.get('job_id')) is str and re.fullmatch(r'[0-9]+\.stdct-mgmt-02',row['job_id']), 'REAL_PBS_ID_OR_UNRESOLVED_INTENT')
    if row['status'] == STATES[-1]:
        need(type(row.get('native_exit_status')) is int
             and 0 <= a.number(row.get('actual_CPU_s')) <= 960
             and row.get('charged_constructor_upper') == 2
             and type(row.get('terminal_native_sha256')) is str
             and re.fullmatch('[0-9a-f]{64}',row['terminal_native_sha256']), 'ACTUAL_TERMINAL_AND_FULL_FAILURE_CHARGE')
    return row


def validate_global(g):
    rows = [r for r in g['PBS_submissions'] if r.get('platform_qualification_version') == VERSION]
    need(len(rows) <= 1, 'ONE_ALLOCATION_NOT_REPEATED_PROBES')
    for row in rows: validate_row(row,g)
    return rows


def register(g, row):
    a.validate_global(g); need(not validate_global(g), 'NO_REPLAY')
    need(c.validated(g) is not None and all(r['status'] in a.TERMINAL for r in g['PBS_submissions']), 'INACTIVE_REAL_BOUNDARY')
    validate_row(row,g)
    need(row['job_id'] is None and row['status'] == STATES[0], 'NEW_UNSUBMITTED_REQUEST')
    v = deepcopy(g); p = v[a.PACKAGE]
    need(p['remaining_resources']['PBS_submissions'] >= 3
         and p['remaining_resources']['Hopper_CPU_s'] >= 960, 'BOTH_EVAL_PBS_AND_ORIGINAL_ENGINEERING_CPU_RETAINED')
    p['remaining_resources']['PBS_submissions'] -= 1
    v['PBS_submissions'].append(deepcopy(row)); v['used']['PBS_submissions'] += 1
    a.validate_global(v); validate_global(v)
    return v


def bind(g, request_id, job_id):
    v = deepcopy(g); rows = validate_global(v)
    need(len(rows) == 1 and rows[0]['request_id'] == request_id
         and rows[0]['job_id'] is None and rows[0]['status'] == STATES[0], 'ONE_BIND_WITH_REAL_QSUB_REPLY')
    rows[0].update(job_id=job_id,status=STATES[1]); validate_row(rows[0],v)
    return v


def settle(g, request_id, *, native, CPU_s, constructor_intent):
    v = deepcopy(g); rows = validate_global(v)
    need(len(rows) == 1 and rows[0]['request_id'] == request_id
         and rows[0]['status'] == STATES[1], 'NO_SECOND_SETTLEMENT')
    row = rows[0]
    need(native['job_state'] == 'F' and type(native['Exit_status']) is int
         and native['job_id'] == row['job_id'] and native['request_id'] == request_id
         and native['submission_nonce'] == row['submission_nonce'], 'ACTUAL_F_AND_REQUEST_NONCE')
    need(0 <= a.number(CPU_s) <= 960 and type(constructor_intent) is bool, 'BOUNDED_REAL_COST')
    # If the job reached the pre-construction intent, retain its full two-
    # constructor upper even on failure. Neither unknown live physics nor a
    # negative task outcome is inferred from a missing result.
    row.update(status=STATES[-1],native_exit_status=native['Exit_status'],actual_CPU_s=CPU_s,
               charged_constructor_upper=2,terminal_native_sha256=a.sha(a.raw(native)),
               constructor_intent_seen=constructor_intent)
    v['CPU_reserved_s'] -= 960; v['CPU_s'] += CPU_s
    p = v[a.PACKAGE]['remaining_resources']; p['Hopper_CPU_s'] -= 960; p['engine_starts'] -= 2
    v[c.FIELD]['render_constructor_upper_charged'] = 2
    a.validate_global(v); validate_global(v)
    return v


def member(root, name):
    p = Path(name)
    need(not p.is_absolute() and '..' not in p.parts, 'RELATIVE_SEALED_MEMBER')
    p = root/p
    need(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root.resolve())
         and p.stat().st_uid == os.getuid() and not p.stat().st_mode & 0o222, 'OWN_IMMUTABLE_MEMBER:'+name)
    return p


def sha_file(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
    return h.hexdigest()


def native_allocation(job, *, job_id, hostname):
    need(job['job_state'] == 'R' and job['project'] == 'CFP04-CF-007'
         and job['Job_Name'] == JOB_NAME and job['Job_Owner'].split('@')[0] == 'e1192548', 'REAL_OWN_RUNNING_CPU_JOB')
    hosts = {x.split('/')[0].split('.')[0].lower() for x in job['exec_host'].split('+')}
    need(hostname.split('.')[0].lower() in hosts and not hostname.lower().startswith(('stdct-login','vanda')),
         'EXACT_ALLOCATED_COMPUTE_HOST_NOT_LOGIN')
    resources = job['Resource_List']; h,m,sec = map(int,resources['walltime'].split(':'))
    need(int(resources.get('ngpus',0)) == 0 and int(resources['ncpus']) == 2
         and h*3600+m*60+sec == 480 and re.fullmatch(r'[0-9]+\.stdct-mgmt-02',job_id), 'ORIGINAL_BOUNDED_ZERO_GPU_ALLOCATION')
    return job


def validate(root, *, compute=False):
    root = Path(root)
    need(root.is_absolute() and root.is_dir() and not root.is_symlink() and os.getuid() == 22182, 'OWN_REGISTERED_VANDA_ROOT')
    read = lambda p:json.loads(Path(p).read_bytes())
    m = read(member(root,'RUN_MANIFEST.json')); b = read(member(root,'OUTPUT_BINDING.private.json'))
    need(m['scope'] == VERSION and m['execution']['profile'] == PROFILE and m['limits'] == LIMITS
         and m['run_uuid'] == root.name and m['output_root'] == b['run_root'] == str(root)
         and m['owner_uid'] == b['uid'] == os.getuid() and b['role'] == ROLE
         and m['execution']['output_binding_sha256'] == sha_file(root/'OUTPUT_BINDING.private.json'), 'STANDARD_MANIFEST_PROFILE_OUTPUT_BINDING')
    registry = read(b['registry']); entries = [r for r in registry['entries'] if r.get('role') == ROLE]
    need(len(entries) == 1 and entries[0]['path'] == str(root) and entries[0]['run_uuid'] == root.name, 'ONE_EXISTING_REGISTRY_ENTRY')
    environment = read(m['execution']['environment_registry'])
    profiles = [p for p in environment['profiles'] if p.get('name') == PROFILE]
    need(sha_file(m['execution']['environment_registry']) == m['execution']['environment_registry_sha256']
         and len(profiles) == 1 and profiles[0]['source_release'] == m['code']['vla_commit']
         and profiles[0]['manifest'] == str(root/'RUN_MANIFEST.json')
         and profiles[0]['entry_module'] == 'scripts.pi0294.pv1.mix_vanda_render'
         and profiles[0]['PBS_required'] is True and profiles[0]['ngpus'] == 0,
         'ACTUAL_ORIGINAL_ENVIRONMENT_REGISTRY_NEW_TYPED_PROFILE')
    lease = read(b['lease']); g = read(root.parent.parent/'HOPPER_RESOURCE_LEDGER.json'); marker = c.validated(g)
    need(lease['status'] == 'ACTIVE' and lease['run_uuid'] == root.name and lease['authority_release'] == s.TASK_SOURCE
         and lease['deadline_epoch'] == marker['deadline_epoch'] == m['authority']['deadline_epoch']
         and marker['T_CD'] <= time.time() < lease['deadline_epoch'], 'ACTUAL_ADOPTED_CLOCK_AND_ACTIVE_LEASE')
    need(not any(p.exists() for p in (root/'STOP.json',root.parent.parent/'STOP.json')), 'ORIGINAL_STOP_FENCES')
    code = root/m['code']['capsule_relative']; seal = read(member(root,'SOURCE_SEAL.json'))
    need(Path(__file__).resolve() == (code/'scripts/pi0294/pv1/mix_vanda_render.py').resolve()
         and seal['source_sha'] == m['code']['vla_commit'] and sha_file(root/'SOURCE_SEAL.json') == m['code']['source_seal_sha256'], 'EXACT_EXECUTED_IMMUTABLE_CAPSULE')
    for name, pin in seal['files'].items(): need(sha_file(member(code,name)) == pin, 'SOURCE_SHA:'+name)
    for name, pin in m['execution']['sealed_inputs'].items(): need(sha_file(member(root,name)) == pin, 'INPUT_SHA:'+name)
    request = read(member(root,'REQUEST.json')); rows = validate_global(g)
    need(len(rows) == 1 and rows[0]['request_id'] == request['request_id']
         and rows[0]['request_sha256'] == sha_file(root/'REQUEST.json')
         and rows[0]['run_manifest_sha256'] == sha_file(root/'RUN_MANIFEST.json')
         and rows[0]['submission_nonce'] == request['submission_nonce'], 'ACTUAL_ONCE_GLOBAL_ROW_NOT_MANIFEST_ONLY')
    assets = read(root/'plans/ASSETS.private.json')
    need(len(assets['files']) == 26, 'ORIGINAL_COMPLETE_ASSET_SET')
    for name,pin in assets['files'].items(): need(sha_file(Path(assets['root'])/name) == pin, 'ORIGINAL_ASSET_SHA:'+name)
    frozen = read(root/'plans/FROZEN_NUMERICAL_SOURCE.json')
    from .evaluation_core_scope import ORIGINAL_NUMERICAL_TABLE_SHA
    import hashlib
    need(len(frozen) == 15 and hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest() == ORIGINAL_NUMERICAL_TABLE_SHA
         and all(seal['files'].get(n) == pin for n,pin in frozen.items()), 'ALL_FIFTEEN_FROZEN_SCIENCE_FILES')
    if compute:
        jid = os.environ.get('PBS_JOBID')
        need(jid == rows[0]['job_id'] and rows[0]['status'] == STATES[1], 'ACTUAL_BOUND_JOB')
        result = subprocess.run(['/opt/pbs/bin/qstat','-f','-F','json',jid],text=True,capture_output=True,timeout=25,check=True)
        jobs = json.loads(result.stdout)['Jobs']; need(set(jobs) == {jid}, 'NO_FILTERED_NATIVE_JOB')
        native_allocation(jobs[jid], job_id=jid, hostname=socket.gethostname())
        variables = jobs[jid]['Variable_List']
        need(variables['PV1_REQUEST_ID'] == request['request_id'] and variables['PV1_SUBMISSION_NONCE'] == request['submission_nonce'], 'PBS_REQUEST_NONCE')
    return m,assets


def run(root):
    # All entry/lease/cost/source checks precede native-library imports.
    m,assets = validate(root,compute=True); root = Path(root)
    need(os.environ.get('LIBGL_ALWAYS_SOFTWARE') == '1' and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
         and os.environ.get('__EGL_VENDOR_LIBRARY_FILENAMES') == '/usr/share/glvnd/egl_vendor.d/50_mesa.json'
         and os.environ.get('MUJOCO_GL') == 'egl' and 'MUJOCO_EGL_DEVICE_ID' not in os.environ, 'ORIGINAL_MESA_RENDER_ENVIRONMENT')
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
    import resource
    resource.setrlimit(resource.RLIMIT_CPU,(450,450)); resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    output = root/'evidence'; output.mkdir(mode=0o700)
    def put(name,value):
        p = output/name
        with p.open('xb') as f: f.write(a.raw(value)); f.flush(); os.fsync(f.fileno())
        p.chmod(0o400)
    cards = json.loads((root/'plans/INITIAL_SCENES.json').read_bytes())
    need(len(cards) == 2 and {x['task'] for x in cards} == {'T1','T2'}, 'ONLY_TWO_ORIGINAL_INITIAL_SCENES')
    put('CONSTRUCTOR_INTENT.json',dict(constructors_upper=2,physics_ticks=0,epoch=time.time()))
    import mujoco
    import numpy as np
    from PIL import Image
    from scripts.pi0294.scene import make_engine
    from scripts.pi0294.s1_preflight import fingerprint
    need(mujoco.__version__ == '3.2.7' and np.__version__ == '1.26.4', 'ORIGINAL_NATIVE_PACKAGE_VERSIONS')
    records = []; began = time.process_time()
    for i,card in enumerate(cards):
        engine = make_engine(card,assets['root'])
        try:
            state = engine.snapshot(); before = fingerprint(engine); images,geometry = engine.render(state)
            from OpenGL.GL import glGetString, GL_VENDOR, GL_RENDERER
            vendor = glGetString(GL_VENDOR).decode(); renderer = glGetString(GL_RENDERER).decode()
            need('Mesa' in vendor and 'llvmpipe' in renderer.lower(), 'ORIGINAL_SOFTWARE_RENDERER_NOT_HARDWARE_EGL')
            need(set(images) == {'body_rgb','wrist_rgb'} and engine.steps == 0 and engine.data.time == 0
                 and before == fingerprint(engine), 'RENDER_PRESERVES_FULL_NATIVE_STATE_ZERO_INTEGRATIONS')
            saved = {}
            for name,pixels in images.items():
                need(pixels.shape == (256,256,3) and pixels.dtype == np.uint8 and np.isfinite(pixels).all()
                     and float(pixels.std()) > 8, 'ACTUAL_NONBLANK_ORIGINAL_DUAL_RGB')
                p = output/(str(i)+'-'+name+'.png'); need(not p.exists(), 'NO_OVERWRITE')
                Image.fromarray(pixels).save(p); p.chmod(0o400)
                saved[name] = dict(sha256=sha_file(p),pixel_sha256=__import__('hashlib').sha256(pixels.tobytes()).hexdigest(),shape=list(pixels.shape),geometry=geometry[name])
            records.append(dict(card_sha256=a.sha(a.raw(card)),task=card['task'],state_before_sha256=before,
                state_after_sha256=fingerprint(engine),time_s=float(engine.data.time),steps=engine.steps,
                GL_VENDOR=vendor,GL_RENDERER=renderer,images=saved))
        finally: engine.close()
    retained = c.enforce_output(output,LIMITS['output_bytes'])
    put('RESULT.json',dict(status='PASS_NATIVE_VANDA_ORIGINAL_SOURCE_DUAL_RGB_ZERO_INTEGRATION',
        records=records,mujoco=mujoco.__version__,numpy=np.__version__,constructors=2,physics_ticks=0,
        episode_starts=0,model_loads=0,neural_queries=0,optimizer_updates=0,CPU_s=time.process_time()-began,
        retained_output=retained,task_or_Teacher_qualification=False,Eval_conditions_completed=0,epoch=time.time()))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('--root',required=True)
    args = parser.parse_args(); run(args.root)
