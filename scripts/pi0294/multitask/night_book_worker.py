"""One finite, registered source attempt through the existing standard entry.

No scheduler, registration, GLOBAL connection, neural call or retry. The root
owner must first adopt the new shared night allowance and sealed local lease.
"""
import hashlib,json,os,resource,sys,time
from pathlib import Path

from scripts.pi0294.dec1 import source_cpu_once_io as once
from scripts.pi0294.multitask import night_scope as scope
from scripts.pi0294.task_path_entry import node_local_paths,registered_snapshot

raw=lambda x:(json.dumps(x,indent=2,allow_nan=False)+'\n').encode()
sha=lambda b:hashlib.sha256(b).hexdigest()


def put(path,b):
    b=b if type(b)is bytes else raw(b)
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
    with os.fdopen(fd,'wb')as f:f.write(b);f.flush();os.fsync(f.fileno())


def encode(value):
    import numpy as np
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def run(root,side,*,task_kind='BOOK'):
    if task_kind not in('BOOK','ROBOCASA_CABINET'):raise ValueError('FIXED_SOURCE_TASK_KIND')
    import numpy as np
    import mujoco
    import PIL
    from PIL import Image
    from .source_engine import make_engine
    from .libero_book_teacher import LiberoBookTeacher
    from .capture_core import capture
    root=Path(root);start=time.monotonic();cpu=time.process_time()
    from .night_scope import for_root
    scope=for_root(root)
    if os.getuid()!=2009 or os.uname().nodename!='corelab-4080s' or side not in('left','right'):
        raise PermissionError('ONE_REGISTERED_CORE_SOURCE_ATTEMPT')
    resource.setrlimit(resource.RLIMIT_CPU,(2800,2800));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    if len(os.sched_getaffinity(0))>4:raise PermissionError('MAX_FOUR_CPU_CORES')
    m=json.loads((root/'RUN_MANIFEST.json').read_bytes())
    if m['scope']!=scope.SCOPE or Path(__file__).resolve()!=root/m['capsule_relative']/'scripts/pi0294/multitask/night_book_worker.py':
        raise PermissionError('OWN_IMMUTABLE_CAPSULE')
    code=root/m['capsule_relative'];seal=json.loads((root/'SOURCE_SEAL.json').read_bytes())
    for n,pin in seal['files'].items():
        if sha((code/n).read_bytes())!=pin:raise PermissionError('EXACT_SEALED_SOURCE_MEMBER:'+n)
    for n,pin in m['sealed_inputs'].items():
        if sha((root/n).read_bytes())!=pin:raise PermissionError('EXACT_IMMUTABLE_INPUT:'+n)
    scope.authority((root/'NIGHT_HUMAN_APPROVAL.json').read_bytes(),now=time.time())
    binding=json.loads((root/'OUTPUT_BINDING.private.json').read_bytes())
    registered_snapshot((root/'control/CANONICAL_REGISTRY_READBACK.json').read_bytes(),
        expected_sha256=binding['registry_snapshot_sha256'],run_root=str(root),role=binding['role'])
    paths=json.loads((root/'RUNTIME_PATHS.private.json').read_bytes())
    actual=node_local_paths(paths)
    local_origins=[sys.executable,np.__file__,mujoco.__file__,PIL.__file__]
    for name in local_origins:
        if not any(Path(name).resolve().is_relative_to(Path(p['resolved'])) for p in actual['paths']):
            raise PermissionError('ACTUAL_RUNTIME_ORIGIN_OUTSIDE_PREQUALIFIED_LOCAL_PATHS')
    account_path=root/'control/NIGHT_SOURCE_ACCOUNT.json'
    ab=account_path.read_bytes();account=scope.validate_account(json.loads(ab))
    if not account['native_adopted'] or account['qualification_sha256']!=sha((root/'SOURCE_QUALIFICATION.json').read_bytes()):
        raise PermissionError('ACTUAL_ADOPTION_AND_PREDECLARED_ASSET_ACTION_SCORE_QUALIFICATION')
    if (root/'STOP.json').exists():raise PermissionError('SOURCE_STOP_FENCE')
    fixed=json.loads((root/'FIXED_TASK_CARDS.json').read_bytes())
    card=(fixed[task_kind]if task_kind in fixed else fixed)[side]
    if card['task_duration_s']!=32. or card['warmup_s']!=5.:raise PermissionError('FIXED_FULL_TASK_DURATION')
    if task_kind=='ROBOCASA_CABINET':
        from .cabinet_repair02 import checked_card
        checked_card(card,side)
    elif card.get('task')!='BOOK':raise PermissionError('ORIGINAL_BOOK_TASK_CARD')
    operation=('LIBERO_'if task_kind=='BOOK'else'ROBOCASA_')+side.upper()+'_FIRST01'
    if task_kind=='ROBOCASA_CABINET':
        from .cabinet_repair02 import attempt_operation
        operation=attempt_operation(card,side)
    after=scope.reserve(account,operation=operation,kind='ATTEMPT',upper=dict(attempts=1,
        physics_integrations=50000,RGB_pairs=2000,Core_CPU_s=2800,render_device_s=600,
        Core_peak_bytes=512*1024**2),now=time.time(),task='LIBERO_BOOK_'+side.upper()if task_kind=='BOOK'else'ROBOCASA_OPEN_CABINET',
        qualification=account['qualification_sha256'])
    key='NIGHT30-'+operation
    once.commit(account_path,expected_before_sha256=sha(ab),proposed_bytes=raw(after),request_id=key,
        intent_path=account_path.with_name(key+'.INTENT.json'),receipt_path=account_path.with_name(key+'.RECEIPT.json'),
        lock_path=account_path.with_name('NIGHT_SOURCE_ACCOUNT.owner.lock'))
    out=root/'outputs'/operation;out.mkdir(mode=0o700)
    put(out/'CARD.json',card)
    stream=(out/'ACTUAL_RECORDS.jsonl').open('x')
    engine=None;render_begin=None;images_written=0;events={};bytes_written=0
    def guard():
        lease=json.loads((root/'control/NATIVE_SOURCE_LEASE.json').read_bytes())
        if (lease['status']!='ACTIVE' or lease['run_uuid']!=root.name or lease['approval_sha256']!=scope.APPROVAL_SHA
                or time.time()>=min(lease['deadline_epoch'],scope.DEADLINE) or time.monotonic()-start>=580
                or (root/'STOP.json').exists()):raise PermissionError('ORIGINAL_OWNER_LEASE_STOP_OR_FINITE_CLOCK')
        if engine is not None and engine.steps>=50000:raise PermissionError('ALL_INTEGRATIONS_UPPER')
        if engine is not None and getattr(engine,'render_pairs',0)>=2000:raise PermissionError('ALL_RGB_PAIRS_UPPER')
        if bytes_written>=500*1024**2:raise PermissionError('PREPAID_OUTPUT_UPPER')
    def emit(kind,row):
        nonlocal images_written,bytes_written
        guard();events[kind]=events.get(kind,0)+1
        if kind=='observation':
            row=dict(row,policy_obs=dict(row['policy_obs']))
            for name in('body_rgb','wrist_rgb'):
                path=out/(str(row['index']).zfill(4)+'_'+name+'.png')
                Image.fromarray(row['policy_obs'][name]).save(path)
                b=path.read_bytes();row['policy_obs'][name]=dict(file=path.name,sha256=sha(b),bytes=len(b))
                bytes_written+=len(b)
            images_written+=1
        line=json.dumps(dict(kind=kind,value=row),default=encode,separators=(',',':'),allow_nan=False)+'\n'
        stream.write(line);bytes_written+=len(line.encode())
        if kind in('completed_interval','native_tick_failure','qualification','warmup_complete'):
            stream.flush();os.fsync(stream.fileno())
    result=None;error=None
    try:
        guard()
        assets=json.loads((root/'ASSET_BINDINGS.private.json').read_bytes())
        node_local_paths(dict(paths,assets=[pin['path']for pin in assets['files'].values()]))
        for name,pin in assets['files'].items():
            b=Path(pin['path']).read_bytes()
            if sha(b)!=pin['sha256'] or len(b)!=pin['bytes']:raise PermissionError('EXACT_PINNED_NATIVE_ASSET:'+name)
        if task_kind=='BOOK':
            definitions={n:(root/'inputs'/n/(n+'.xml')).read_bytes()for n in('black_book','desk_caddy')}
            engine=make_engine(card,assets['robot_root'],pinned_libero=dict(definitions=definitions,
                asset_paths={n:assets['files'][n]['path']for n in assets['libero_binary_names']}))
        else:
            from .feedback_teacher import SourceSkillTeacher
            definitions={n:Path(assets['files'][key]['path']).read_bytes()for n,key in assets['robocasa_definition_members'].items()}
            textures={n:assets['files'][key]['path']for n,key in assets['robocasa_texture_members'].items()}
            engine=make_engine(card,assets['robot_root'],pinned_robocasa=dict(definitions=definitions,texture_paths=textures))
        put(out/'NATIVE_SCENE.private.xml',engine.xml.encode())
        bytes_written+=(out/'CARD.json').stat().st_size+(out/'NATIVE_SCENE.private.xml').stat().st_size
        initial=engine.snapshot()
        if initial['safety_valid']is not True or initial['native_grasp']:
            raise PermissionError('ACTUAL_NATIVE_INITIAL_SAFETY_OR_UNINTENDED_GRASP')
        render_begin=time.monotonic();rgb,cal=engine.render(initial)
        # Qualification cameras are actual and charged; not dataset frames.
        for n,a in rgb.items():
            if a.shape!=(256,256,3) or a.dtype!=np.uint8:raise PermissionError('QUALIFICATION_DUAL_RGB')
            Image.fromarray(a).save(out/('qualification_'+n+'.png'))
            bytes_written+=(out/('qualification_'+n+'.png')).stat().st_size
            uv=np.asarray(cal[n]['object_points_uv'])
            optical=np.asarray(cal[n]['T_world_optical'])
            depth=(np.asarray(initial['object_points'])-optical[:3,3])@optical[:3,:3]
            if np.sum(np.all((uv>=0)&(uv<256),axis=1)&(depth[:,2]>0))<4:
                raise PermissionError('QUALIFICATION_OBJECT_OUTSIDE_DECLARED_DUAL_CAMERA_VIEW')
        images_written+=1
        extra={}
        if task_kind=='ROBOCASA_CABINET':
            j=engine._id(mujoco.mjtObj.mjOBJ_JOINT,engine.task_binding['passive_joint'])
            extra=dict(source_port=card['source_port'],card_sha256=sha((out/'CARD.json').read_bytes()),
                native_scene_sha256=sha((out/'NATIVE_SCENE.private.xml').read_bytes()),
                passive_joint_name=engine.task_binding['passive_joint'],
                passive_joint_qposadr=int(engine.model.jnt_qposadr[j]),passive_joint_dofadr=int(engine.model.jnt_dofadr[j]),
                passive_joint_range=engine.model.jnt_range[j].tolist())
        emit('qualification',dict(actual_state=initial,calibration=cal,camera_pair=True,**extra,
            new_constructor=1,steps=engine.steps,simulation_time=engine.data.time,
            original_arm_bounds=engine.arm_bounds,original_finger_q_indices=engine.finger_q,
            original_finger_v_indices=engine.finger_v,robot_geom_ids=[i for i in range(engine.model.ngeom)
                if int(engine.model.geom_bodyid[i])in engine.robot_bodies],
            arm_geom_ids=[i for i in range(engine.model.ngeom)
                if (mujoco.mj_id2name(engine.model,mujoco.mjtObj.mjOBJ_BODY,int(engine.model.geom_bodyid[i]))or'').startswith('x5_')],
            obstacle_geom_ids=sorted(engine.obstacles),geom_bindings=engine.geoms))
        home=engine.pose('tool').copy()
        for _ in range(2500):
            guard();engine.step(home,.042/.044);s=engine.snapshot()
            emit('warmup_tick',dict(native_step=engine.steps,actual_state=s))
            if s['safety_valid']is not True:raise PermissionError('ORIGINAL_NATIVE_SOURCE_WARMUP_SAFETY_STOP')
        emit('warmup_complete',dict(steps=engine.steps,actual_state=engine.snapshot()))
        teacher=LiberoBookTeacher(card)if task_kind=='BOOK'else SourceSkillTeacher(card)
        result=capture(engine,teacher,instruction=card['instruction'],guard=guard,emit=emit,duration_s=32.)
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc)
        if engine is not None:
            # Do not let an already expired guard prevent terminal evidence.
            stream.write(json.dumps(dict(kind='terminal_failure',error=error,actual_step=engine.steps,
                actual_state=engine.snapshot()),default=encode,allow_nan=False)+'\n')
        if any(t in error for t in('SAFETY','VELOCITY','PENETRATION','JOINT')):
            put(root/'STOP.json',dict(reason=error,operation=operation,automatic_retry=False))
    finally:
        if engine is not None:engine.close()
        stream.flush();os.fsync(stream.fileno());stream.close()
    terminal=dict(status='RAW_SOURCE_COMPLETE_UNAUDITED' if result else 'SOURCE_ATTEMPT_FAILED',
        operation=operation,error=error,result=result,actual_integrations=engine.steps if engine else 0,
        actual_RGB_pairs=getattr(engine,'render_pairs',0)if engine else 0,
        recorded_RGB_pairs=images_written,events=events,CPU_s=time.process_time()-cpu,
        wall_s=time.monotonic()-start,render_device_s=0 if render_begin is None else time.monotonic()-render_begin,
        task_success=None,training_admitted=False,automatic_retry=False,new_model_queries_updates_PBS=0,
        observed_epoch=time.time())
    put(out/'TERMINAL.json',terminal)
    print(json.dumps(terminal))
    return 0 if result else 1


if __name__=='__main__':sys.exit(run(sys.argv[1],sys.argv[2]))
