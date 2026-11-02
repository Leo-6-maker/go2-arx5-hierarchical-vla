"""Finite native source camera/reset checks, submitted by the existing owner.

The existing Night10 queue supplies resource waiting and a finite user unit.
This command does not schedule training, step physics or admit expert data.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

KIND = 'V7_SOURCE_CAMERA_RESET_ZERO_INTEGRATION_QUALIFICATION'
TASKS = ('T1', 'T2', 'BOOK', 'DOOR')
MODULE = 'scripts.pi0294.multitask.source_reset_qualification'
LIMITS = dict(source_constructors=12, RGB_pairs=8, physics_integrations=0,
              model_loads=0, generation_queries=0, optimizer_updates=0)


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path, value):
    from scripts.pi0292.durable import write
    if Path(path).exists():
        raise PermissionError('QUALIFICATION_OUTPUT_EXISTS_NO_REPLAY:' + str(path))
    write(path, value)


def preflight(root):
    """Read-only custody checks; neither importing MuJoCo nor acquiring a GPU."""
    from scripts.pi0294.task_path_entry import node_local_paths, registered_snapshot
    root = Path(root).resolve(); request = read(root/'REQUEST.json')
    m = read(root/'RUN_MANIFEST.json'); lease = read(root/'control/LEASE.json')
    if (os.getuid() != 2009 or os.uname().nodename != 'corelab-4080s'
            or m.get('kind') != KIND or m.get('limits') != LIMITS
            or m.get('request_sha256') != sha(root/'REQUEST.json')
            or m.get('run_uuid') != root.name or set(request['templates']) != set(TASKS)
            or set(request['selected_conditions']) != set(TASKS)
            or lease.get('status') != 'ACTIVE' or lease.get('run_uuid') != root.name
            or lease.get('manifest_sha256') != sha(root/'RUN_MANIFEST.json')
            or time.time() >= lease.get('deadline_epoch', 0)):
        raise PermissionError('ONE_SEALED_NATIVE_ZERO_INTEGRATION_CHECK')
    for name,pin in m['sealed_inputs'].items():
        if sha(root/name)!=pin:
            raise PermissionError('SEALED_QUALIFICATION_INPUT_CHANGED:'+name)
    if sha(Path(request['dependency_source']).parent/'CAPSULE_SEAL.json')!=request['dependency_seal_sha256']:
        raise PermissionError('UNCHANGED_LOCAL_DEPENDENCY_CAPSULE')
    source = Path(__file__).resolve().parents[3]
    for name, pin in read(root/'SOURCE_CUSTODY.json')['files'].items():
        if sha(source/name) != pin:
            raise PermissionError('QUALIFICATION_SOURCE_CHANGED:' + name)
    for task, template in request['templates'].items():
        folder = Path(template['root'])
        if (folder.is_symlink() or sha(folder/'RUN_MANIFEST.json') != template['manifest_sha256']
                or read(folder/'control/NATIVE_SOURCE_LEASE.json')['status'] != 'CLOSED'):
            raise PermissionError('CLOSED_ORIGINAL_ASSET_TEMPLATE:' + task)
    node_local_paths(read(root/'RUNTIME_PATHS.private.json'))
    registered_snapshot((root/'control/CANONICAL_REGISTRY_READBACK.json').read_bytes(),
        expected_sha256=m['registry_snapshot_sha256'], run_root=str(root), role=m['registry_role'])
    return request


def verify_noise(condition, student):
    """Check every possible condition query against the original fixed keys."""
    task = condition['task']
    namespace = (condition['noise_namespace'] if task in ('T1', 'T2')
                 else 'BOOK5_DEV10' if task == 'BOOK' else 'SHARED4_CUSTOM_PORT08_DEV_V1')
    key = condition['scene_group']+'/'+task if task in ('T1', 'T2') else condition['id']
    expected = [int.from_bytes(hashlib.sha256(f'{namespace}/{key}/{i}'.encode()).digest()[:8],
                              'big') % (2**63) for i in range(80)]
    if student.seed_namespace != namespace or [student.generation_seed(i) for i in range(80)] != expected:
        raise PermissionError('ORIGINAL_MODEL_INDEPENDENT_CONDITION_NOISE')
    return dict(all80_original_keys_match=True, namespace=namespace,
                sequence_sha256=hashlib.sha256(json.dumps(expected).encode()).hexdigest())


def gpu_headroom(request):
    p = subprocess.run(['nvidia-smi', '-i', str(request['GPU_index']),
        '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=5, check=True)
    uuid, free = (x.strip() for x in p.stdout.strip().split(','))
    if uuid != request['GPU_uuid'] or int(free) < 1024:
        raise PermissionError('QUALIFICATION_ACTUAL_ONE_GIB_GPU_HEADROOM')
    return int(free)


def child(root, task, repeat, *, dirty=False):
    """One original native constructor in a new process, always closed once."""
    root = Path(root); request = read(root/'REQUEST.json')
    resource.setrlimit(resource.RLIMIT_CPU, (35, 35))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    import numpy as np
    sys.path.append(request['physics_packages'])
    import mujoco
    from PIL import Image
    from . import shared_evaluation as ev
    from .book_policy import initial_camera_qualification
    folder = root/'outputs'/(task+'_'+str(repeat)+('_dirty' if dirty else ''))
    folder.mkdir(exist_ok=False)
    engine = None; error = None; witness = None; camera = None; noise = None; fault = None
    began = time.process_time(); closed = False; used = []
    try:
        if np.__version__ != '1.26.4' or mujoco.__version__ != '3.2.7' or 'torch' in sys.modules:
            raise PermissionError('ORIGINAL_CORE_PHYSICS_NUMERICS_NO_TORCH')
        condition = request['selected_conditions'][task]; card = ev.card_for_condition(condition)
        put(folder/'CARD.json', card)
        engine = ev.make_engine(card, Path(request['templates'][task]['root']))
        student = ev.SharedStudent(None, condition, None, lambda: None)
        witness = ev.initial_witness(engine, student, card); noise = verify_noise(condition, student)
        if dirty:
            # A deliberate fault in this private disposable actor, never a repair
            # of a real episode. The rejected value must remain unchanged.
            engine.data.qvel[0] = .1
            try:
                ev.initial_witness(engine, student, card)
            except PermissionError as exc:
                fault = dict(rejected=True, error=str(exc), injected_qvel=.1,
                             not_repaired=float(engine.data.qvel[0]) == .1)
            if not fault or not fault['not_repaired']:
                raise PermissionError('DIRTY_NATIVE_INITIAL_STATE_NOT_REJECTED')
        else:
            used.append(gpu_headroom(request))
            initial = engine.snapshot(); images, calibration = engine.render(initial)
            if set(images) != {'body_rgb', 'wrist_rgb'} or set(calibration) != set(images):
                raise PermissionError('EXACT_TWO_ORIGINAL_CAMERAS')
            camera = initial_camera_qualification(initial, images, calibration)
            (folder/'NATIVE_SCENE.private.xml').write_text(engine.xml)
            extra = ev.qualification_extra(engine, folder, images, calibration)
            for name, image in images.items():
                cal = calibration[name]; k = np.asarray(cal['K']); pose = np.asarray(cal['T_world_optical'])
                if (float(cal['timestamp_s']) != 0 or k.shape != (3, 3) or pose.shape != (4, 4)
                        or not np.isfinite(k).all() or not np.isfinite(pose).all()):
                    raise PermissionError('ACTUAL_INITIAL_CAMERA_CALIBRATION_AND_CLOCK')
                Image.fromarray(image).save(folder/(name+'.png'))
            put(folder/'CALIBRATION.json', calibration)
            camera.update(extra=extra, image_sha256={n:sha(folder/(n+'.png')) for n in images},
                          simulation_time_s=float(engine.data.time), native_steps=engine.steps)
            used.append(gpu_headroom(request))
        if engine.steps != 0 or engine.data.time != 0 or student.queries != 0 or 'torch' in sys.modules:
            raise PermissionError('QUALIFICATION_MUST_NOT_STEP_OR_LOAD_A_POLICY')
    except BaseException as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    finally:
        if engine is not None:
            try:
                engine.close(); closed = True
            except BaseException as exc:
                error = error or type(exc).__name__ + ': ' + str(exc)
        result = dict(status='PASS' if error is None and closed else 'FAILED', error=error,
            task=task, repeat=repeat, dirty_fault=dirty, PID=os.getpid(),
            proc_start=Path('/proc/self/stat').read_text().split()[21], witness=witness,
            noise=noise, camera=camera, fault=fault, engine_close_returned=closed,
            physics_integrations=0 if engine is None else engine.steps,
            RGB_pairs=0 if engine is None else getattr(engine, 'render_pairs', 0),
            neural_models=0, optimizer_updates=0, Torch_imported='torch' in sys.modules,
            numpy_version=np.__version__, MuJoCo_version=mujoco.__version__,
            GPU_free_MiB=used, CPU_s=time.process_time()-began,
            maximum_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            epoch=time.time())
        put(folder/'RESULT.json', result)
    return 0 if result['status'] == 'PASS' else 1


def predecessor_faults(root):
    """Use a real finite user unit and a real writer; no mocked OS observer."""
    from .book_reset import verify_predecessor, unit_state
    from scripts.pi0292.durable import write
    root = Path(root); previous = root.with_name(root.name+'-probe'); previous.mkdir(exist_ok=False)
    (previous/'control').mkdir()
    put(previous/'RUN_MANIFEST.json', dict(kind='RESET_QUALIFICATION_FAULT_ACTOR_NOT_A_MODEL',
        run_uuid=previous.name, parent_qualification=root.name, physics_integrations=0))
    put(previous/'RESULT.json', dict(status='DELIBERATE_CLOSED_METADATA_WHILE_ACTOR_LIVE',
        qualification_fault_fixture_not_a_scientific_result=True))
    def lease(status):
        return dict(status=status, run_uuid=previous.name,
            manifest_sha256=sha(previous/'RUN_MANIFEST.json'), result_sha256=sha(previous/'RESULT.json'))
    write(previous/'control/LEASE.json', dict(lease('CLOSED'), deliberate_fault_injection_only=True))
    put(previous/'FAULT_CLOSED_LEASE.json', read(previous/'control/LEASE.json'))
    def predecessor():
        return dict(root=str(previous), unit=previous.name+'.service',
            files={n:sha(previous/n) for n in ('RUN_MANIFEST.json','RESULT.json','control/LEASE.json')})
    def reject(label, expected):
        try:
            verify_predecessor(predecessor(), current_root=root)
        except PermissionError as exc:
            if expected not in str(exc):
                raise
            return dict(rejected=True, error=str(exc), next_condition_started=False)
        raise PermissionError('NATIVE_FAULT_NOT_REJECTED:'+label)
    actor = "import os,sys,time,json;from pathlib import Path;r=Path(sys.argv[1]);(r/'ACTOR_READY.json').write_text(json.dumps({'PID':os.getpid()}));end=time.monotonic()+40\nwhile not(r/'EXIT_REQUESTED').exists()and time.monotonic()<end:time.sleep(.1)\n(r/'ACTOR_EXIT.json').write_text(json.dumps({'normal_exit':True}))"
    evidence = {}; writer = None
    try:
        subprocess.run(['systemd-run','--user','--unit='+previous.name,
            '--property=RemainAfterExit=yes','--property=CPUQuota=25%','--property=MemoryMax=64M',
            '--working-directory='+str(previous),'/usr/bin/env','-i','PATH=/usr/bin:/bin','LANG=C.UTF-8',
            '/usr/bin/python3','-B','-c',actor,str(previous)], capture_output=True, check=True, timeout=8)
        stop = time.monotonic()+12
        while not(previous/'ACTOR_READY.json').exists():
            if time.monotonic() >= stop:raise TimeoutError('NATIVE_PREDECESSOR_ACTOR_READY')
            time.sleep(.1)
        live = unit_state(previous.name+'.service')
        if int(live['MainPID']) != read(previous/'ACTOR_READY.json')['PID']:
            raise PermissionError('REAL_PREDECESSOR_PROCESS_IDENTITY')
        evidence['live_predecessor_rejected'] = dict(reject('live','PROCESS_STILL_LIVE'), actual_unit=live)
        (previous/'EXIT_REQUESTED').touch(exist_ok=False)
        stop = time.monotonic()+12
        while True:
            state = unit_state(previous.name+'.service')
            if state['MainPID']=='0' and state['SubState']=='exited':break
            if time.monotonic() >= stop:raise TimeoutError('NATIVE_PREDECESSOR_NORMAL_EXIT')
            time.sleep(.1)
        if read(previous/'ACTOR_EXIT.json')['normal_exit']is not True:
            raise PermissionError('ACTUAL_PROBE_NORMAL_EXIT')
        write(previous/'RESULT.json',dict(status='COMPLETE_ZERO_PHYSICS_PREDECESSOR_PROBE',
            actual_unit=state, actual_actor=read(previous/'ACTOR_READY.json'), error=None))
        write(previous/'control/LEASE.json',lease('CLOSED'))
        evidence['positive_closed_predecessor'] = verify_predecessor(predecessor(),current_root=root)
        write(previous/'control/LEASE.json',lease('UNKNOWN'))
        evidence['unknown_close_rejected'] = reject('unknown','CLOSE_NOT_ESTABLISHED')
        put(previous/'FAULT_UNKNOWN_LEASE.json',read(previous/'control/LEASE.json'))
        write(previous/'control/LEASE.json',lease('CLOSED'))
        code="import sys;from pathlib import Path;r=Path(sys.argv[1]);f=(r/'probe_writer.bin').open('ab');f.write(b'owned qualification writer');f.flush();(r/'WRITER_READY').touch();sys.stdin.readline();f.close()"
        writer = subprocess.Popen(['/usr/bin/python3','-B','-c',code,str(previous)],stdin=subprocess.PIPE,text=True)
        stop=time.monotonic()+8
        while not(previous/'WRITER_READY').exists():
            if time.monotonic()>=stop:raise TimeoutError('NATIVE_WRITER_READY')
            time.sleep(.1)
        evidence['residual_writer_rejected'] = dict(reject('writer','RESIDUAL_WRITER'),actual_writer_PID=writer.pid)
        writer.communicate('close\n',timeout=8)
        if writer.returncode != 0:raise PermissionError('QUALIFICATION_WRITER_NOT_CLOSED')
        evidence['positive_after_writer_closed'] = verify_predecessor(predecessor(),current_root=root)
        p=subprocess.run(['systemctl','--user','show',previous.name+'.service','-p','CPUUsageNSec'],
            capture_output=True,text=True,check=True,timeout=5)
        evidence['separate_predecessor_actor_CPU_s']=int(p.stdout.strip().split('=')[1])/1e9
        put(root/'PREDECESSOR_FAULTS.json',evidence)
        return evidence
    finally:
        # Only signal our finite probe through its own data/pipe. No kill,
        # service restart or modification of an existing experiment.
        if not(previous/'EXIT_REQUESTED').exists():(previous/'EXIT_REQUESTED').touch()
        if writer is not None and writer.poll()is None:writer.communicate('close\n',timeout=8)
        stop=time.monotonic()+12
        try:
            while unit_state(previous.name+'.service').get('MainPID')!='0':
                if time.monotonic()>=stop:raise RuntimeError('UNKNOWN_PREDECESSOR_ACTOR_CLOSURE')
                time.sleep(.1)
        except subprocess.SubprocessError as exc:
            raise RuntimeError('UNKNOWN_PREDECESSOR_ACTOR_CLOSURE')from exc


def run(root):
    from scripts.pi0292.durable import OwnerLock, write
    root=Path(root).resolve(); request=preflight(root); started=time.monotonic(); parent_cpu=time.process_time()
    lock=None
    resource.setrlimit(resource.RLIMIT_CPU,(180,180))
    rows=[];faults={};error=None;attempted=0;before=resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        lock=OwnerLock(root.parent/'global-native-GPU-locks'/(request['GPU_uuid']+'.lock'))
        gpu_headroom(request)
        for task in TASKS:
            for repeat,dirty in ((0,False),(1,False),(2,True)):
                lease=read(root/'control/LEASE.json')
                if lease['status']!='ACTIVE'or time.time()>=lease['deadline_epoch']:
                    raise PermissionError('FINITE_QUALIFICATION_LEASE_OR_DEADLINE')
                folder=root/'outputs'/(task+'_'+str(repeat)+('_dirty'if dirty else''))
                command=[sys.executable,'-B','-m',MODULE,'child','--root',str(root),
                         '--task',task,'--repeat',str(repeat)]+(['--dirty']if dirty else[])
                put(root/'outputs'/(task+'_'+str(repeat)+'_INTENT.json'),dict(command=command,epoch=time.time()))
                attempted+=1
                p=subprocess.run(command,cwd=root,env=os.environ.copy(),capture_output=True,text=True,timeout=60)
                put(root/'outputs'/(task+'_'+str(repeat)+'_EXIT.json'),dict(returncode=p.returncode,
                    stdout=p.stdout[-4000:],stderr=p.stderr[-4000:]))
                row=read(folder/'RESULT.json'); rows.append(row)
                if p.returncode or row['status']!='PASS':raise RuntimeError('NATIVE_CHILD_FAILED:'+task+':'+str(row['error']))
        faults=predecessor_faults(root)
        from . import shared_evaluation as ev, book_reset
        tasks={}
        for task in TASKS:
            a,b,c=[r for r in rows if r['task']==task]
            if a['PID']==b['PID']or a['witness']!=b['witness']or a['noise']!=b['noise']:
                raise PermissionError('TWO_DISTINCT_NATIVE_PROCESSES_AND_IDENTICAL_RESET:'+task)
            tasks[task]=dict(two_processes=True,identical_initial_witness=True,
                original_dual_camera_qualified=all(r['camera']['original_dual_camera_qualified']for r in(a,b)),
                dirty_initial_rejected=c['fault']['rejected'],children=[r['PID']for r in(a,b,c)])
        if sum(r['physics_integrations']for r in rows)!=0 or sum(r['RGB_pairs']for r in rows)!=8:
            raise PermissionError('EXACT_EIGHT_PAIRS_AND_ZERO_INTEGRATIONS')
        common=dict(passed=True,host=os.uname().nodename,uid=os.getuid(),numpy_version=rows[0]['numpy_version'],
            MuJoCo_version=rows[0]['MuJoCo_version'],qualification_kind=KIND,
            evidence=dict(tasks=tasks,child_results={str((Path('outputs')/(r['task']+'_'+str(r['repeat'])+('_dirty'if r['dirty_fault']else''))/'RESULT.json')):
                sha(root/'outputs'/(r['task']+'_'+str(r['repeat'])+('_dirty'if r['dirty_fault']else''))/'RESULT.json')for r in rows},
                predecessor_faults_sha256=sha(root/'PREDECESSOR_FAULTS.json')))
        put(root/'TASK_NATIVE_QUALIFICATION.json',dict(common,schema='SHARED4_SOURCE_RESET_NATIVE_QUALIFICATION_V1',
            checks=dict.fromkeys(ev.NATIVE_CHECKS,True),source_files=ev.source_files(),task_templates=request['templates']))
        put(root/'RESET_NATIVE_QUALIFICATION.json',dict(common,schema='BOOK_RESET_NATIVE_QUALIFICATION_V1',
            checks=dict.fromkeys(book_reset.NATIVE_CHECKS,True),source_files=book_reset.source_files(),
            template_manifest_sha256=request['templates']['BOOK']['manifest_sha256']))
    except BaseException as exc:
        error=type(exc).__name__+': '+str(exc)
    finally:
        after=resource.getrusage(resource.RUSAGE_CHILDREN)
        result=dict(status='COMPLETE_NATIVE_SOURCE_CAMERA_RESET_QUALIFICATION'if error is None else'FAILED_NATIVE_SOURCE_CAMERA_RESET_QUALIFICATION',
            error=error,children_attempted=attempted,children_completed=len(rows),physics_integrations=0,
            RGB_pairs=sum(r['RGB_pairs']for r in rows)if attempted==len(rows)else None,
            RGB_pairs_upper=8,neural_models=0,optimizer_updates=0,valid_DEV_conditions=0,
            CPU_s=time.process_time()-parent_cpu+after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime
                +faults.get('separate_predecessor_actor_CPU_s',0),
            wall_s=time.monotonic()-started,epoch=time.time(),full_task_dynamics_qualified=False)
        put(root/'RESULT.json',result)
        write(root/'control/LEASE.json',dict(status='UNKNOWN'if error and'UNKNOWN'in error else'CLOSED',run_uuid=root.name,
            manifest_sha256=sha(root/'RUN_MANIFEST.json'),result_sha256=sha(root/'RESULT.json'),closed_epoch=time.time()))
        if lock is not None:lock.close()
    return 0 if error is None else 1


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('preflight','run','child'))
    parser.add_argument('--root',required=True);parser.add_argument('--task',choices=TASKS)
    parser.add_argument('--repeat',type=int);parser.add_argument('--dirty',action='store_true')
    args=parser.parse_args()
    if args.action=='preflight':preflight(args.root);return 0
    if args.action=='child':return child(args.root,args.task,args.repeat,dirty=args.dirty)
    return run(args.root)


if __name__=='__main__':
    raise SystemExit(main())
