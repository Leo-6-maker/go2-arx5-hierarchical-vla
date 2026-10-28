"""New first512 identity at the original registered A40 entry boundary.

Historical Core capture is an explicit conservative budget input, never a
fresh Core write/read claim. Every model/update/restore release also consumes
the current native owner material and an independent own-job PBS read.
"""
import base64,hashlib,json,os,socket,sys,time
from pathlib import Path
from scripts.pi0292.durable import read
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.dec1.a40_training_context import sealed
from scripts.pi0294.mp1.owner_heartbeat import read_live_owner
from . import minimal_execution as e,minimal_scope as scope,minimal_native_admission as admission

need=e.need
LOCAL='plans/MINIMAL_CORE_LOCAL_CAPTURE.private.json'
ENTRY='scripts.pi0294.dec1.a40_training_runtime'
REQUIRED={LOCAL,'NEURAL_BINDINGS.private.json','plans/A40_ENVIRONMENT_BINDING.private.json',
    'MIX_OWNER_CONFIG.private.json','plans/MINIMAL_ACTUAL_CORE_MEASUREMENT.private.json','plans/PV1_FIXED_TRAIN_PROBES.json',
    'plans/MINIMAL_EFFECTIVE_QUEUE_LIMITS.private.json','plans/MINIMAL_RAW_PARENT_AUDIT_PLAN.private.json',
    'plans/MINIMAL_SIX_PRECONTACT16_TRAIN_CAPTURE.private.json','plans/MINIMAL_LANGUAGE24_PANEL.private.json',
    'plans/MINIMAL_TWELVE_PANEL.json','plans/MINIMAL_ORIGINAL_PROBE_READER.private.json',
    'plans/PRIMARY_LEARNING_RECIPES.json','plans/FROZEN_SOURCE_PAIR_RECIPE.json'}


def captured_LOCAL(root):
    v=sealed(root,LOCAL);body=base64.b64decode(v['LOCAL_base64'],validate=True)
    need(e.sha(body)==v['LOCAL_sha256'] and v.get('source')=='ACTUAL_CANONICAL_CORE_LOCAL_READONLY'
        and v.get('capture_is_fresh_native_authority')is False,'CONTEXT_EXPLICIT_REAL_CAPTURE_NOT_FAKE_CURRENT_LOCAL')
    return body


def registered_identity(root,request_path,m,b,r,g,*,uid,pbs_job,now,require_bound):
    """Pure identity join; outer context reads actual files/environment."""
    root=Path(root);request_path=Path(request_path);index=e.validate(g);mark=e.authority.validate(g);rows=e.rows(g)
    need(type(require_bound)is bool and len(rows)==1 and index is not None,'CONTEXT_ONE_ADOPTED_NATIVE_MAIN')
    row=rows[0]
    need(uid==22182 and not root.is_symlink() and m['scope']==scope.SCOPE
        and m['platform_training_version']==e.VERSION and m['execution']['profile']==e.PROFILE
        and m['execution']['agent_liveness_required']is False
        and m['run_uuid']==r['run_uuid']==root.name and m['output_root']==b['run_root']==str(root)
        and Path(m['registered_runs_root']).is_absolute() and root.is_relative_to(Path(m['registered_runs_root']))
        and b['uid']==uid and b['run_manifest_sha256']==r['run_manifest_sha256']
        and m['code']['vla_commit']==r['source_sha']
        and r['request_id']==row['request_id'] and r['source_sha']==row['source_sha']
        and r['T_CD']==mark['T_CD'] and r['deadline_epoch']==mark['deadline_epoch']
        and r['T_CD']<=now<r['deadline_epoch']
        and request_path.parent.parent==root/'pbs' and request_path.name=='REQUEST.json'
        and request_path.parent.name=='alloc-'+str(r['submit_count']).zfill(2),
        'CONTEXT_EXACT_REGISTERED_SCOPE_ROOT_REQUEST_AND_ORIGINAL_CLOCK')
    need((row['status']==e.STATES[1] and row['job_id']==pbs_job) or
        (not require_bound and row['status']==e.STATES[0] and row['job_id']is None),
        'CONTEXT_ACTUAL_RESERVATION_AND_SAME_PBS_BIND_NOT_OLD_ARM')


def root_context(root,request_path,*,full=False,require_bound=True):
    root=Path(root);request_path=Path(request_path)
    m=sealed(root,'RUN_MANIFEST.json');b=sealed(root,'OUTPUT_BINDING.private.json')
    r=sealed(root,request_path.relative_to(root).as_posix());e.request(r)
    g=read(root.parent.parent/'HOPPER_RESOURCE_LEDGER.json')
    registered_identity(root,request_path,m,b,r,g,uid=os.getuid(),pbs_job=os.environ.get('PBS_JOBID'),
        now=time.time(),require_bound=require_bound)
    need(sha(root/'RUN_MANIFEST.json')==r['run_manifest_sha256'] and sha(request_path)==e.rows(g)[0]['request_sha256'],
        'CONTEXT_IMMUTABLE_REQUEST_AND_MANIFEST_BYTES')
    worker=Path(os.environ.get('PV1_RUN_MANIFEST','/INVALID_UNREGISTERED_ENTRY'))
    need(worker.is_file() and not worker.is_symlink() and worker.stat().st_uid==os.getuid()
        and not worker.stat().st_mode&0o222 and worker.read_bytes()==(root/'RUN_MANIFEST.json').read_bytes(),
        'CONTEXT_STANDARD_ENTRY_ACTUAL_WORKER_MANIFEST')
    lease=read(b['lease']);reg=[x for x in read(b['registry'])['entries']if x.get('role')==b['role']]
    need(lease['status']=='ACTIVE' and lease['run_uuid']==root.name and lease['manifest_sha256']==r['run_manifest_sha256']
        and lease['deadline_epoch']==r['deadline_epoch'] and len(reg)==1 and reg[0]['path']==str(root),
        'CONTEXT_OWN_ACTIVE_LEASE_AND_CANONICAL_REGISTRY')
    profiles=[p for p in read(b['environment'])['profiles']if p['name']==e.PROFILE]
    need(len(profiles)==1 and profiles[0]['manifest']==str(root/'RUN_MANIFEST.json')
        and profiles[0]['entry_module']==ENTRY and profiles[0]['PBS_required']is True,
        'CONTEXT_ORIGINAL_ENTRY_EXPLICIT_NEW_PROFILE_NOT_CD_OR_GEN')
    inputs=m['execution']['sealed_inputs'];need(REQUIRED<=set(inputs),'CONTEXT_COMPLETE_FROZEN_INPUT_ROLES')
    for name,h in inputs.items():need(sha(sealed(root,name,parse=False))==h,'CONTEXT_SEALED_INPUT:'+name)
    seal=sealed(root,'SOURCE_SEAL.json');prefix=Path(m['code']['capsule_relative']);code=root/prefix
    need(not prefix.is_absolute() and '..'not in prefix.parts and sha(root/'SOURCE_SEAL.json')==m['code']['source_seal_sha256']
        and seal['source_sha']==r['source_sha'] and Path(__file__).resolve()==(code/'scripts/pi0294/generalization/minimal_native_context.py').resolve(),
        'CONTEXT_OWN_COLD_EXECUTING_CAPSULE')
    critical={'scripts/pi0294/generalization/'+n for n in ('minimal_native_context.py','minimal_native_runtime.py',
        'minimal_native_admission.py','minimal_worker.py','minimal_session_gate.py','minimal_effects.py')}
    critical.add('scripts/pi0294/dec1/a40_training_runtime.py')
    need(critical<=set(seal['files']),'CONTEXT_FULL_NATIVE_DEPENDENCY_SEAL')
    for name in seal['files']if full else critical:
        need(sha(sealed(root,str(prefix/name),parse=False))==seal['files'][name],'CONTEXT_EXECUTING_SOURCE:'+name)
    from scripts.pi0294.pv1.evaluation_core_scope import ORIGINAL_NUMERICAL_TABLE_SHA
    frozen=m['code']['frozen_numerical_files']
    need(len(frozen)==15 and hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()==ORIGINAL_NUMERICAL_TABLE_SHA
        and all(seal['files'].get(n)==pin and sha(sealed(root,str(prefix/n),parse=False))==pin for n,pin in frozen.items()),
        'CONTEXT_ORIGINAL_FIFTEEN_NUMERICAL_FILES_AND_887E_OVERLAY_BYTES')
    env=sealed(root,'plans/A40_ENVIRONMENT_BINDING.private.json');bindings=sealed(root,'NEURAL_BINDINGS.private.json')
    need(Path(sys.executable).resolve()==Path(env['python']).resolve() and Path(sys.prefix).resolve()==Path(env['venv_prefix']).resolve()
        and sha(env['python'])==env['python_binary_sha256'] and sha(env['environment_registry'])==env['environment_registry_sha256']
        and env['official_source']==bindings['source'] and env['official_commit']=='51d4c89f72fda44cbf77285c6a8114b52676b8a1',
        'CONTEXT_ORIGINAL_NATIVE_ENVIRONMENT_NOT_PACKAGE_INSTALL')
    raw=sealed(root,'plans/MINIMAL_RAW_PARENT_AUDIT_PLAN.private.json')
    from . import minimal_model
    need(raw['files']==minimal_model.FILES and raw['raw_parent_files_identity']==e.training.RAW_PARENT_FILES_ID
        and sha(root/'plans/MINIMAL_RAW_PARENT_AUDIT_PLAN.private.json')==r['raw_parent_audit_plan_sha256'],
        'CONTEXT_RAW_PRETRAINED_PARENT_NOT_CORR_OR_OTHER_ENDPOINT')
    return m,r


class StartupOwnerStillQueued(Exception):pass


class Guard:
    def __init__(self,root,allocation):
        self.root=Path(root);self.allocation=allocation;self.b=sealed(root,'OUTPUT_BINDING.private.json')
        self.local=captured_LOCAL(root);self.cached_global=None;self.forecast=None

    def __call__(self,*,startup=False):
        r=self.root;a=self.allocation;now=time.time();gb=(r.parent.parent/'HOPPER_RESOURCE_LEDGER.json').read_bytes();g=json.loads(gb)
        lease=read(self.b['lease'])
        need(lease['status']=='ACTIVE' and lease['run_uuid']==r.name and lease['manifest_sha256']==a['run_manifest_sha256']
            and not any(p.exists()for p in (r/'STOP.json',r.parent.parent/'STOP.json'))
        and a['allocation_start_epoch']<=now<min(a['allocation_end_epoch'],a['deadline_epoch'])
            and g['unknown_effects']==g['pending_updates']==[],
            'CONTEXT_CURRENT_STOP_LEASE_TIME_AND_UNCLEARED_GLOBAL_FENCES')
        if gb!=self.cached_global:
            from scripts.pi0294.pv1.neural_ledger import check
            check(g);rows=e.rows(g)
            need(len(rows)==1 and rows[0]['status']==e.STATES[1] and rows[0]['job_id']==a['job_id']
                and rows[0]['request_sha256']==a['request_sha256'] and rows[0]['source_sha']==a['source_sha'],
                'CONTEXT_CURRENT_EXACT_GLOBAL_ROW')
            proof=scope.first512_forecast(gb,self.local)
            need(proof['cap_changes_required']=={},'CONTEXT_CURRENT_FULL_FUTURE_RESOURCES')
            self.cached_global=gb;self.forecast=proof
        beat=read_live_owner(r/'control',r.name)['heartbeat']
        need(beat.get('scope')==e.VERSION and beat.get('grant_sha256')==e.authority.GRANT_SHA
            and beat.get('job_id')==a['job_id'] and beat.get('request_sha256')==a['request_sha256']
            and beat.get('source_sha')==a['source_sha'] and beat.get('global_unknown_effects')==[],
            'CONTEXT_REAL_SAME_OWNER_FRESH_HEARTBEAT')
        if startup and beat.get('native_job_state')=='Q':raise StartupOwnerStillQueued()
        need(beat.get('native_job_state')=='R','CONTEXT_NATIVE_R_REQUIRED_FOR_ANY_MODEL_OR_UPDATE')
        return self.forecast


def phase_process(ready,initial,*,job_id,hostname):
    proc_start=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    need(ready['job_id']==initial['job_id']==job_id and ready['hostname']==initial['hostname']==hostname
        and ready['pid']==initial['pid']==os.getpid() and ready['proc_start']==initial['proc_start']==proc_start
        and ready['boot_id']==initial['boot_id']==boot and initial['observed_epoch']<=ready['observed_epoch'],
        'CONTEXT_READY_SAME_ACTUAL_KERNEL_PROCESS_NOT_PID_OR_600S_AGE_ASSUMPTION')


def verify_release(root,request_path,allocation,release,*,ready_bytes=None,effect_bytes=None):
    """Independently join current PBS and raw owner inputs before consumption."""
    from scripts.pi0294.dec1.a40_training_runtime import native_job
    root=Path(root);request_path=Path(request_path);req=read(request_path)
    material_path=sealed(root,release['material_relative'],parse=False)
    need(sha(material_path)==release['material_sha256'],'CONTEXT_ACTUAL_NATIVE_MATERIAL_BYTES')
    v=read(material_path);decode=lambda k:base64.b64decode(v[k],validate=True)
    job,host,raw_job=native_job(req)
    if ready_bytes is not None:
        phase_process(json.loads(ready_bytes),read(request_path.parent/'PBS_READY.json'),job_id=job,hostname=host)
    proof=admission.validate(global_bytes=(root.parent.parent/'HOPPER_RESOURCE_LEDGER.json').read_bytes(),
        local_bytes=decode('LOCAL_base64'),request_bytes=request_path.read_bytes(),manifest_bytes=(root/'RUN_MANIFEST.json').read_bytes(),
        job_id=job,hostname=host,user=os.environ['USER'],native_job_bytes=e.raw(raw_job),queue_proof=v['complete_queue'],
        site_bytes=decode('site_base64'),effective_queue_bytes=sealed(root,str(request_path.parent.relative_to(root)/'EFFECTIVE_QUEUE_LIMITS.json'),parse=False).read_bytes(),
        independent_job_bytes=decode('independent_confirmation_base64')if v.get('independent_confirmation_base64')else None,
        now_epoch=time.time(),ready_bytes=ready_bytes,effect_bytes=effect_bytes)
    need(proof['job_id']==allocation['job_id'] and proof['request_sha256']==allocation['request_sha256']
        and proof['source_sha']==allocation['source_sha'],'CONTEXT_RELEASE_SAME_COMPUTE_ALLOCATION')
    return proof
