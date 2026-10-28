"""Finite registered re-audit of two saved R1 T1s, never Teacher replay.

Original saved-state FK/contact/camera/causality/score functions and original
video auditor are reused. Only the old scalar seed identity read is adapted.
All new audit/video files are written under this role's own fresh output root.
"""
import argparse,hashlib,importlib,json,os,resource,signal,sys,time,traceback
from pathlib import Path
from . import amd_adoption as a,amd_capture as capture,amd_capture_terminal as scope
from .amd_seed_audit import legacy_identity_read_view


def read(path):return json.loads(Path(path).read_bytes())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()


class Counter:
    def __init__(self,root,guard):
        self.root=Path(root);self.guard=guard;self.value=read(self.root/'RESOURCE_LEDGER.json')
        a.need(self.value['status']=='RESERVED'and self.value['engine_starts']==0
            and self.value['unknown_effects']==[],'SAVED_AUDIT_FRESH_COUNTER')
    def progress(self,**fields):
        from scripts.pi0292.durable import write
        self.value.update(fields);write(self.root/'RESOURCE_LEDGER.json',self.value)
    def start(self):
        self.guard();a.need(self.value['engine_starts']<4,'SAVED_AUDIT_CTOR_CAP_BEFORE_CALL')
        self.value['engine_starts']+=1;self.progress(status='RUNNING')


def validate(root):
    root=Path(root);a.need(os.getuid()==2009 and os.uname().nodename=='corelab-amd-server'
        and root.is_absolute()and not root.is_symlink(),'SAVED_AUDIT_HOST_UID')
    m=read(root/'RUN_MANIFEST.json');b=read(root/'OUTPUT_BINDING.private.json')
    a.need(m['scope']==scope.ROLE and m['execution']['profile']==scope.PROFILE
        and m['run_uuid']==root.name=='gen1-amd-saved-r1-reaudit-01'
        and m['output_root']==b['run_root']==str(root)
        and sha(root/'RUN_MANIFEST.json')==b['run_manifest_sha256'],'SAVED_AUDIT_OWN_MANIFEST')
    g=read(root/'plans/ACTUAL_GLOBAL_SAVED_AUDIT_ALLOCATION.private.json');mark=a.validate(g)
    allocation=mark['teacher_pilot']['engineering_recovery']['saved_data_reaudit']
    a.need(allocation['status']=='RESERVED_SAVED_DATA_ONLY'
        and allocation['manifest_sha256']==b['run_manifest_sha256']and allocation['run_uuid']==root.name
        and allocation['engine_starts']==4 and allocation['physics_ticks']==allocation['episode_starts']==0,
        'SAVED_AUDIT_ACTUAL_GLOBAL_TYPED_RESERVATION')
    lease=read(b['lease']);a.need(lease['status']=='ACTIVE'and lease['run_uuid']==root.name
        and lease['manifest_sha256']==b['run_manifest_sha256']
        and mark['T_CD']<=time.time()<min(mark['deadline_epoch'],m['limits']['operation_deadline_epoch']),
        'SAVED_AUDIT_LIVE_OWN_LEASE_AND_CLOCK')
    reg=read(b['registry']);a.need(sum(x['path']==str(root)and x['role']==b['role']for x in reg['entries'])==1,
        'SAVED_AUDIT_REGISTERED_ROLE')
    seal=read(root/'SOURCE_SEAL.json');code=root/m['code']['capsule_relative']
    a.need(sha(root/'SOURCE_SEAL.json')==m['code']['source_seal_sha256']
        and Path(__file__).resolve()==(code/'scripts/pi0294/generalization/amd_saved_audit.py').resolve(),
        'SAVED_AUDIT_ACTUAL_IMMUTABLE_ENTRY')
    for name,h in seal['files'].items():a.need(sha(code/name)==h,'SAVED_AUDIT_SOURCE:'+name)
    for name,h in seal['frozen15'].items():a.need(sha(code/name)==h,'SAVED_AUDIT_ORIGINAL_NUMERICAL:'+name)
    for name,h in m['execution']['sealed_inputs'].items():a.need(sha(root/name)==h,'SAVED_AUDIT_INPUT:'+name)
    cases=read(root/'plans/SAVED_R1_CASES.private.json')
    a.need(a.sha(a.raw(cases))==allocation['input_cases_sha256']and set(cases)=={'0','1'},
        'SAVED_AUDIT_EXACT_TWO_ORIGINAL_INPUTS')
    a.need(m['limits']['engine_starts']==4 and m['limits']['physics_ticks']==0
        and m['limits']['Teacher_episodes']==m['limits']['GR00T_queries']==m['limits']['training_updates']==0
        and m['limits']['CPU_s']==600 and len(m['limits']['cpus'])==1
        and set(m['limits']['cpus'])<=os.sched_getaffinity(0),'SAVED_AUDIT_FINITE_CPU_NO_NEW_TASKS')
    assets=read(root/'plans/AMD_ASSETS.private.json');a.need(len(assets['files'])==26,'SAVED_AUDIT_ORIGINAL_ASSET_SET')
    for name,h in assets['files'].items():a.need(sha(Path(assets['root'])/name)==h,'SAVED_AUDIT_ASSET:'+name)
    a.need(m['execution']['SIF_sha256']==a.SIF_SHA and os.environ.get('CUDA_VISIBLE_DEVICES')==''
        and os.environ.get('MUJOCO_GL')=='egl'and os.environ.get('LIBGL_ALWAYS_SOFTWARE')=='1'
        and os.environ.get('__EGL_VENDOR_LIBRARY_FILENAMES')=='/usr/share/glvnd/egl_vendor.d/50_mesa.json',
        'SAVED_AUDIT_ORIGINAL_CPU_RUNTIME')
    return m,b,cases


def run(root):
    from scripts.pi0292.durable import OwnerLock,write
    from scripts.pi0294.fd1 import put
    root=Path(root);lock=OwnerLock(root/'SAVED_RAW_AUDIT_OWNER.lock');began=time.process_time();children=resource.getrusage(resource.RUSAGE_CHILDREN)
    m,b,cases=validate(root);os.sched_setaffinity(0,set(m['limits']['cpus']));resource.setrlimit(resource.RLIMIT_CPU,(600,600));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    def CPU():
        v=resource.getrusage(resource.RUSAGE_CHILDREN)
        return time.process_time()-began+v.ru_utime+v.ru_stime-children.ru_utime-children.ru_stime
    def guard():
        a.need(time.time()<m['limits']['operation_deadline_epoch']and CPU()<590
            and read(b['lease'])['status']=='ACTIVE'and not(root/'STOP.json').exists(),
            'SAVED_AUDIT_CPU_TIME_LEASE_OR_STOP')
    def expired(*args):raise TimeoutError('GEN1_SAVED_RAW_AUDIT_WALL')
    signal.signal(signal.SIGALRM,expired);signal.alarm(max(1,int(m['limits']['operation_deadline_epoch']-time.time())))
    meter=Counter(root,guard);rows=[];failure=None
    put(root/'SAVED_AUDIT_STARTED.json',dict(pid=os.getpid(),procstart=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],source=m['code']['vla_commit'],epoch=time.time(),Teacher_replay=False))
    try:
        from scripts.pi0294.pv1.mix_vanda_codec import validate as codec_validate
        codec_root,codec_manifest=capture.codec_context(root,m);codec_validate(codec_root,codec_manifest)
        import mujoco
        a.need(mujoco.__version__=='3.2.7','SAVED_AUDIT_ORIGINAL_MUJOCO_ABI')
        import numpy as np
        from scripts.pi0294.pv1.capture import counted_model
        from scripts.pi0294.scene import make_engine
        from scripts.pi0291_arx5_data.config import PARAMETERS
        from scripts.pi0294.e1.audit import video
        native_path=Path(__file__).resolve().parents[2]/'pi0291_arx5_data';sys.path.insert(0,str(native_path))
        base=importlib.import_module('scripts.pi0292.audit_data')
        native_model=mujoco.MjModel;originals=[mujoco.mj_step,mujoco.mj_step1,mujoco.mj_step2]
        def forbidden(*args,**kwargs):raise PermissionError('GEN1_SAVED_AUDITOR_FORBIDS_ALL_PHYSICS_INTEGRATION')
        mujoco.mj_step=mujoco.mj_step1=mujoco.mj_step2=forbidden;mujoco.MjModel=counted_model(native_model,meter)
        try:
            for key,case in cases.items():
                guard();source=Path(case['raw_root']);identifier=case['attempt_id'];folder=source/'episodes'/identifier
                sm=read(source/'RUN_MANIFEST.json');sb=read(source/'OUTPUT_BINDING.private.json')
                a.need(sha(source/'RUN_MANIFEST.json')==case['manifest_sha256']and sm['code']['vla_commit']==scope.SOURCE
                    and read(sb['lease'])['status']=='CLOSED','SAVED_AUDIT_REAL_CLOSED_INPUT_CUSTODY')
                old_seal=read(source/'SOURCE_SEAL.json');new_seal=read(root/'SOURCE_SEAL.json')
                a.need(sha(source/'SOURCE_SEAL.json')==sm['code']['source_seal_sha256'],
                    'SAVED_AUDIT_ORIGINAL_SOURCE_SEAL_UNCHANGED')
                for name in ('scripts/pi0292/audit_data.py','scripts/pi0294/audit.py','scripts/pi0294/scene.py','scripts/pi0294/e1/audit.py'):
                    a.need(old_seal['files'][name]==new_seal['files'][name],'SAVED_AUDIT_UNCHANGED_ORIGINAL_CONSUMER')
                card=read(folder/'CARD.json');ticket=read(source/'reservations'/(identifier+'.json'));worker=read(folder/'WORKER_RECEIPT.json');result=read(folder/'RESULT.json');s1=read(folder/'S1_RESULT.json')
                a.need(card['task']=='T1'and sha(folder/'CARD.json')==case['CARD_sha256']
                    and sha(folder/'FILE_MANIFEST.json')==case['raw_manifest_sha256']
                    and sha(folder/'AUDIT.json')==case['original_audit_sha256']
                    and sha(folder/'WORKER_RECEIPT.json')==case['worker_receipt_sha256']
                    and ticket['card']==card and worker['source_sha']==ticket['source_sha']==scope.SOURCE
                    and Path(ticket['root']).resolve()==source.resolve()
                    and Path(ticket['output']).resolve()==folder.resolve(), 'SAVED_AUDIT_ORIGINAL_RESERVATION_AND_BYTES')
                a.need(result['status']=='VALID_COMPLETE'and result['physics_steps']==16000
                    and s1['live_native_failures']==s1['pre_execution_rejections']==0
                    and s1['actions_modified']is False and s1['maximum_live_shadow_difference']<=1e-10,
                    'SAVED_AUDIT_COMPLETE_ORIGINAL_NATIVE_TRAJECTORY')
                with np.load(folder/'highrate.npz',allow_pickle=False)as z:
                    a.need(z['external_force_max'].shape==(16000,2)and not np.any(z['external_force_max']),
                        'SAVED_AUDIT_ORIGINAL_ALL_BODY_ZERO_FORCE')
                out=root/'evidence'/('case-'+key);out.mkdir();before={name:sha(folder/name)for name in ('CARD.json','FILE_MANIFEST.json','AUDIT.json','WORKER_RECEIPT.json')};attempt_before=sha(source/'ATTEMPT_PLAN.json')
                old_read=base.read
                def guarded_read(path):guard();return old_read(path)
                base.read=guarded_read
                try:
                    with legacy_identity_read_view(base,source,card,(root/'plans/GEN1_V5_PREVIEW_V2.private.json').read_bytes())as view:
                        detail=base.audit_episode(source,identifier,lambda bound,mode:make_engine(bound,read(root/'plans/AMD_ASSETS.private.json')['root']),PARAMETERS)
                finally:
                    base.read=old_read
                    a.need(before=={name:sha(folder/name)for name in before}and attempt_before==sha(source/'ATTEMPT_PLAN.json'),
                        'SAVED_AUDIT_ORIGINAL_FILES_UNCHANGED_AFTER_VIEW')
                admitted=bool(detail['status']=='PASS_FULL_RAW_AUDIT'and detail['complete']
                    and detail['independent_score']['persistent_success']and detail['all_action_supervision_valid']
                    and detail['paired_samples']==641 and detail['physics_steps']==16000
                    and detail['actual_committed_H40_windows']==601)
                movie=video(folder,out/'actual-body-wrist.mp4',641)
                record=dict(status='PASS_INDEPENDENT_RAW_AUDIT',id=identifier,admitted=admitted,
                    source_sha=scope.SOURCE,raw_manifest_sha256=case['raw_manifest_sha256'],
                    original_failed_AUDIT_sha256=case['original_audit_sha256'],original_files_modified=False,
                    legacy_seed_identity_read_view=view,detail=detail,video=movie,
                    video_relative=str((out/'actual-body-wrist.mp4').relative_to(root)),
                    integrations=0,render_calls=0,GR00T_model_loads=0,formal_R0_export='NOT_RUN')
                put(out/'INDEPENDENT_RAW_AUDIT.json',record);rows.append(record);meter.progress(completed=len(rows),CPU_s=CPU())
                a.need(sum(p.stat().st_size for p in root.rglob('*')if p.is_file())<=16*1024**2,
                    'SAVED_AUDIT_FINITE_OWN_LOGICAL_OUTPUT')
        finally:
            mujoco.MjModel=native_model;mujoco.mj_step,mujoco.mj_step1,mujoco.mj_step2=originals
        a.need(meter.value['engine_starts']==4 and len(rows)==2,'SAVED_AUDIT_TWO_COMPLETE_ORIGINAL_READS')
        a.need('torch'not in sys.modules,'SAVED_AUDIT_NO_NEURAL_MODULES')
        result=dict(status='KNOWN_SAVED_RAW_REAUDIT_TERMINAL',source=m['code']['vla_commit'],rows=rows,
            actual_constructor_attempts=meter.value['engine_starts'],integrations=0,Teacher_episodes=0,
            raw_admitted=sum(x['admitted']for x in rows),training_admitted=False,Core_CPU_s=CPU())
        put(root/'SAVED_AUDIT_RESULT.json',result);return result
    except BaseException as exc:
        failure=type(exc).__name__+': '+str(exc)
        meter.value['unknown_effects'].append(dict(error=failure,full_remaining_audit_reservation_retained=True))
        put(root/'SAVED_AUDIT_FAILURE.json',dict(error=failure,traceback=traceback.format_exc(),completed=len(rows),constructor_attempts=meter.value['engine_starts'],integrations=0))
        raise
    finally:
        signal.alarm(0);meter.progress(status='FAILED_PRESERVED'if failure else'KNOWN_TERMINAL',CPU_s=CPU());lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    print(json.dumps(run(parser.parse_args().root)))
