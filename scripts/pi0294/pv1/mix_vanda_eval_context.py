"""Registered C/D peers on the actual same Vanda PBS allocation.

The old Core logical wire names remain part of the frozen source worker API.
Host, UID, lease, native job and ledger authority are explicitly Vanda; this
does not alias a Core runtime, a stale LOCAL ledger or the old shared-GPU grant.
"""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from scripts.pi0292.durable import read
from scripts.pi0294.native_dataset import sha
from . import mix_vanda_eval_identity as identity, mix_vanda_eval_accounting as accounting
from . import mix_accounting as a, mix_vanda_cohost as cohost, successor_cd as s
from .evaluation_core_scope import member,ORIGINAL_NUMERICAL_TABLE_SHA,COSMOS_PROCESSOR_PINS

CAPTURE='plans/V5_CD_CLOSED_TRAINING_CAPTURE.private.json'
PEERS='plans/V5_CD_VANDA_REGISTERED_PEERS.private.json'
SCHEMA='V5_CD_VANDA_REGISTERED_EVALUATION_PEER_V1'
LEDGER_SCHEMA='V5_CD_VANDA_POLICY_EFFECT_LEDGER_V1'
ASSET_TABLE_SHA='1b113c2b4bac48bc08a7b6f74880c42210ac419f3f733a3ac0e5d149e29f48ab'
PARENT_BINDING_SHA='c311033e584c232f0409ddfd674d753c882642c3b02eb653cbee1edf57eb8363'


def need(ok,why):
    if not ok:raise PermissionError('V5_VANDA_EVAL_CONTEXT_'+why)


def request_path(root):
    root=Path(root);paths=list((root/'pbs').glob('alloc-*/REQUEST.json'))
    need(len(paths)==1 and paths[0].parent.name=='alloc-01','ONE_LOCAL_REQUEST_NOT_GLOBAL_PBS_ORDINAL')
    member(root,str(paths[0].relative_to(root)))
    return paths[0]


def paths(root,manifest=None):
    root=Path(root);m=read(root/'RUN_MANIFEST.json') if manifest is None else manifest
    need(m['execution']['sealed_inputs'].get(PEERS)==sha(member(root,PEERS)),'SEALED_EXPLICIT_PEER_CUSTODY')
    peers=read(root/PEERS);parent=root.parent.parent
    need(set(peers)=={'core','hopper','owner'},'EXACT_THREE_REGISTERED_ROLES')
    for role,folder in [('core','children'),('hopper','neural-children'),('owner','control')]:
        row=peers[role];p=Path(row['root'])
        need(p.parent==parent/folder and not p.is_symlink()
             and set(row)=={'root','role'} and row['role']=={'core':identity.PHYSICAL_ROLE,'hopper':identity.NEURAL_ROLE,'owner':identity.OWNER_ROLE}[role],
             'REAL_SAME_PARENT_PEER_'+role)
    need(Path(peers['core']['root']).name==Path(peers['hopper']['root']).name==root.name,'SAME_MODEL_AND_PHYSICAL_RUN')
    return peers


def row(root,manifest=None,*,closeout=False,cold=False):
    root=Path(root);m=read(root/'RUN_MANIFEST.json') if manifest is None else manifest
    peers=paths(root,m);neural=Path(peers['hopper']['root']);p=request_path(neural);r=read(p)
    nm=read(neural/'RUN_MANIFEST.json');identity.request_identity(nm,r,sha(neural/'RUN_MANIFEST.json'))
    g=read(root.parent.parent/'HOPPER_RESOURCE_LEDGER.json');accounting.checked(g);record=accounting.item(g,r['request_id'])
    need(record['request_sha256']==sha(p) and record['run_manifest_sha256']==sha(neural/'RUN_MANIFEST.json')
         and record['physical_manifest_sha256']==sha(Path(peers['core']['root'])/'RUN_MANIFEST.json')
         and record['source_sha']==m['code']['vla_commit'] and record['child_run_uuid']==root.name
         and record['submission_nonce']==r['submission_nonce'],'ACTUAL_ORIGINAL_GLOBAL_BOUND_REQUEST')
    need(type(cold) is bool and (not cold or not os.environ.get('PBS_JOBID') and os.uname().nodename.startswith('stdct-login-')),
         'COLD_PREFLIGHT_ONLY_ON_EXISTING_LOGIN_CONTROL')
    states=accounting.STATES[:3] if cold else accounting.STATES[1:3] if not closeout else accounting.STATES[1:]
    need(record['status'] in states,'ACTIVE_OR_EXPLICIT_CLOSEOUT_ONLY')
    if not closeout:need(not g['unknown_effects'] and not g['pending_updates'],'UNCHANGED_UNKNOWN_EFFECT_FENCE')
    identity.bind_closed_endpoint(a.raw(g),r,read(neural/CAPTURE))
    return g,record,r


def deadline(root,manifest,*,closeout=False,cold=False):
    g,record,r=row(root,manifest,closeout=closeout,cold=cold);now=time.time()
    need(type(closeout) is bool and r['T_CD']<=now and (closeout or now<r['deadline_epoch']), 'REAL_ADOPTED_WINDOW_NO_INJECTED_NOW')
    return r['deadline_epoch']


def current_allocation(root,request):
    jid=os.environ.get('PBS_JOBID');need(type(jid) is str,'PBS_REQUIRED_NO_CORE_HANDLE_OR_LOGIN_FALLBACK')
    need(os.getuid()==22182 and os.environ.get('USER')=='e1192548','ACTUAL_VANDA_USER')
    from scripts.pi0294.dec1.owner_transport import ReadOnlyFailure
    command=['/opt/pbs/bin/qstat','-f','-F','json',jid]
    m=read(Path(root)/'RUN_MANIFEST.json')
    if m['scope']==identity.PHYSICAL_ROLE:
        # The sealed CPU-only SIF consumes raw qstat bytes collected on this
        # same compute host by its live owner. It never needs a copied PBS
        # authentication stack or an invented Core handle inside the SIF.
        owner=owner_current(root,request,request_sha256=sha(request_path(Path(paths(root)['hopper']['root']))))
        p=Path(paths(root)['owner']['root'])/'NATIVE_PBS_CURRENT.private.json'
        need(p.is_file() and not p.is_symlink() and p.stat().st_uid==22182 and not p.stat().st_mode&0o222,'OWN_CURRENT_RAW_PBS_SNAPSHOT')
        snapshot=read(p);now=time.time()
        need(snapshot['schema']=='V5_CD_VANDA_SAME_COMPUTE_OWNER_RAW_PBS_V1'
             and snapshot['owner_native_sha256']==a.sha(a.raw(owner))
             and snapshot['request_id']==request['request_id'] and snapshot['job_id']==jid
             and snapshot['source_sha']==request['source_sha'] and 0<=now-snapshot['observed_epoch']<=120,
             'FRESH_REAL_OWNER_RAW_PBS_NOT_A_STALE_PASS_FLAG')
        need(snapshot['raw_PBS_sha256']==a.sha(a.raw(snapshot['raw_PBS'])),'EXACT_RAW_PBS_BYTES')
        jobs={jid:snapshot['raw_PBS']}
    else:
        try:reply=subprocess.run(command,capture_output=True,text=True,timeout=25,check=False)
        except (subprocess.TimeoutExpired,TimeoutError,ConnectionError) as exc:
            error=ReadOnlyFailure('V5_CURRENT_QSTAT_READ_UNKNOWN_NOT_TERMINAL')
            error.diagnostic=dict(argv=command,error_type=type(exc).__name__,stdout=getattr(exc,'stdout',None),stderr=getattr(exc,'stderr',None));raise error from exc
        if reply.returncode or reply.stderr:
            error=ReadOnlyFailure('V5_CURRENT_QSTAT_READ_FAILED_NOT_EMPTY')
            error.diagnostic=dict(argv=command,returncode=reply.returncode,stdout=reply.stdout,stderr=reply.stderr);raise error
        jobs=json.loads(reply.stdout)['Jobs']
    need(set(jobs)=={jid},'EXACT_UNFILTERED_NATIVE_JOB')
    native=identity.native_allocation(jobs[jid],request,job_id=jid,hostname=socket.gethostname(),now_epoch=time.time())
    g,record,_=row(root)
    need(record['job_id']==jid and int(jobs[jid].get('run_count',1))==1,'ACTUAL_BOUND_ONCE_JOB_NO_SCHEDULER_RERUN')
    return dict(native,raw_job=jobs[jid],raw_job_sha256=a.sha(a.raw(jobs[jid])),observed_epoch=time.time())


def owner_current(root,request,*,request_sha256):
    root=Path(root);peers=paths(root);owner=Path(peers['owner']['root'])
    m=read(owner/'RUN_MANIFEST.json');binding=read(owner/'OUTPUT_BINDING.private.json');lease=read(binding['lease'])
    need(m['scope']==identity.OWNER_ROLE and m['run_uuid']==owner.name
         and sha(owner/'RUN_MANIFEST.json')==request['owner_manifest_sha256']
         and m['code']['vla_commit']==request['source_sha']
         and binding['run_root']==str(owner) and binding['uid']==22182
         and lease['status']=='ACTIVE' and lease['run_uuid']==owner.name
         and lease['authority_release']==s.TASK_SOURCE,'REGISTERED_LIVE_OWN_RESIDENT_OWNER')
    entries=[r for r in read(binding['registry'])['entries'] if r.get('role')==binding['role']]
    need(len(entries)==1 and entries[0]['path']==str(owner),'REAL_REGISTERED_OWNER_CUSTODY')
    v=read(member(owner,'NATIVE_OWNER_STARTED.json'))
    need(v['schema']=='V5_CD_VANDA_RESIDENT_NATIVE_OWNER_V1' and v['request_id']==request['request_id']
         and v['request_sha256']==request_sha256 and v['source_sha']==request['source_sha']
         and v['manifest_sha256']==sha(owner/'RUN_MANIFEST.json')
         and v['hostname']==socket.gethostname() and v['uid']==os.getuid()==22182
         and v['job_id']==os.environ.get('PBS_JOBID') and v['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
         'ACTUAL_SAME_JOB_OWNER_NOT_A_CORE_UNIT_OR_SAVED_PID')
    proc=Path('/proc',str(v['pid']));fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
    need(fields[0] not in ('Z','X') and int(fields[19])==v['proc_start']
         and proc.stat().st_uid==22182 and (proc/'cgroup').read_text()==v['cgroup'], 'LIVE_KERNEL_OWNER_PROCSTART_AND_CGROUP')
    return v


def observe_current(root,request,*,bound_actor=False):
    """Real kernel/PBS/GPU facts before Torch, constructors or effect release."""
    root=Path(root);facts=current_allocation(root,request);g,record,_=row(root)
    need(type(bound_actor) is bool,'EXPLICIT_PROCESS_READ')
    pid=record['neural_process']['pid'] if bound_actor else os.getpid()
    proc=Path('/proc',str(pid));fields=(proc/'stat').read_text().rsplit(')',1)[1].split();start=int(fields[19])
    need(fields[0] not in ('Z','X') and proc.stat().st_uid==os.getuid(),'ACTUAL_LIVE_KERNEL_PROCESS')
    if bound_actor:need(start==record['neural_process']['proc_start'],'NO_REUSED_BOUND_PID')
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    epoch=float(next(x.split()[1] for x in Path('/proc/stat').read_text().splitlines() if x.startswith('btime ')))+start/os.sysconf('SC_CLK_TCK')
    visible=os.environ.get('CUDA_VISIBLE_DEVICES');need(type(visible) is str and visible and ',' not in visible,'ONE_PBS_VISIBLE_MODEL_DEVICE')
    reply=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader,nounits'],
        text=True,capture_output=True,timeout=10,check=True)
    devices=[]
    for line in reply.stdout.splitlines():
        index,uid,name,memory=[x.strip() for x in line.split(',')]
        devices.append(dict(index=index,GPU_uuid=uid,name=name,memory_bytes=int(memory)*1024**2))
    selected=[d for d in devices if visible in (d['index'],d['GPU_uuid'])]
    need(len(selected)==1 and 'A40' in selected[0]['name'] and selected[0]['memory_bytes']>=40*1024**3,'ACTUAL_PBS_VISIBLE_A40_INVENTORY')
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],
        text=True,capture_output=True,timeout=10,check=True)
    g,record,_=row(root)
    allowed=record.get('neural_process',{});actual=[]
    for line in apps.stdout.splitlines():
        pp,uid,memory=[x.strip() for x in line.split(',')]
        if uid!=selected[0]['GPU_uuid']:continue
        need(int(pp)==allowed.get('pid'),'NO_UNREGISTERED_GPU_PROCESS_OR_INHERITED_B24_SHARED_EXCEPTION')
        f=Path('/proc',pp,'stat').read_text().rsplit(')',1)[1].split()
        need(f[0] not in ('Z','X') and int(f[19])==allowed.get('proc_start'),'EXACT_CURRENT_MODEL_PROCESS_NOT_PID_REUSE')
        actual.append(dict(pid=int(pp),GPU_uuid=uid,used_memory_MiB=int(memory)))
    return dict(schema='V5_CD_VANDA_NEURAL_PROCESS_STARTED_V1',job_id=facts['job_id'],native_handle=facts['job_id'],
        hostname=socket.gethostname(),uid=os.getuid(),pid=pid,proc_start=start,boot_id=boot,
        cgroup=(proc/'cgroup').read_text(),started_epoch=epoch,observed_epoch=time.time(),
        source_sha=request['source_sha'],request_id=request['request_id'],request_sha256=sha(request_path(root)),
        submission_nonce=request['submission_nonce'],run_manifest_sha256=request['run_manifest_sha256'],
        GPU_uuid=selected[0]['GPU_uuid'],visible=visible,raw_GPU_inventory=devices,raw_GPU_processes=actual,
        actual_PBS=facts['raw_job'],actual_PBS_sha256=facts['raw_job_sha256'],
        allocation_start_epoch=facts['allocation_start_epoch'],allocation_end_epoch=facts['allocation_end_epoch'],
        native_model_permission=False,physics_authorized=False)


def sealed_identity(manifest,request,allocation,*,run_uuid,manifest_sha256,request_sha256):
    identity.request_identity(manifest,request,manifest_sha256)
    expected=dict(status=identity.STATUS,platform_evaluation_version=identity.VERSION,
        run_uuid=run_uuid,source_sha=request['source_sha'],manifest_sha256=manifest_sha256,
        request_sha256=request_sha256,request_id=request['request_id'],submission_nonce=request['submission_nonce'],
        project=request['project'],evaluation_candidate=request['evaluation_candidate'],
        checkpoint_manifest_sha256=request['checkpoint_manifest_sha256'],statistics_sha256=a.STATS)
    need(all(allocation.get(k)==v and type(allocation[k]) is type(v) for k,v in expected.items()), 'FULL_SAME_NATIVE_ALLOCATION_IDENTITY')
    return allocation


def current_identity(root,manifest,request,allocation,**unused):
    need(not unused,'NO_CORE_QUALIFIED_MODEL_READBACK_ALIAS')
    root=Path(root);p=request_path(root)
    sealed_identity(manifest,request,allocation,run_uuid=root.name,manifest_sha256=sha(root/'RUN_MANIFEST.json'),request_sha256=sha(p))
    facts=current_allocation(root,request)
    need(facts['job_id']==allocation['job_id'] and facts['hostname']==allocation['hostname']
         and facts['allocation_start_epoch']==allocation['allocation_start_epoch']
         and facts['allocation_end_epoch']==allocation['allocation_end_epoch'],'FRESH_SAME_JOB_AND_UNCHANGED_WINDOW')
    owner_current(root,request,request_sha256=sha(p))
    return facts


def validate(root,*,closeout=False,compute=True):
    need(type(closeout) is bool and type(compute) is bool,'EXPLICIT_ENTRY_PHASE')
    root=Path(root)
    need(root.is_absolute() and root.is_dir() and not root.is_symlink() and root.stat().st_uid==os.getuid()==22182,
         'REAL_REGISTERED_VANDA_ROOT_AND_UID')
    for name in ('RUN_MANIFEST.json','OUTPUT_BINDING.private.json'):
        p=member(root,name);need(p.is_file() and p.stat().st_uid==22182 and not p.stat().st_mode&0o222,'OWN_IMMUTABLE_'+name)
    m=read(root/'RUN_MANIFEST.json');b=read(root/'OUTPUT_BINDING.private.json');role=m['scope']
    need(m['schema']==SCHEMA and role in (identity.NEURAL_ROLE,identity.PHYSICAL_ROLE)
         and m['run_uuid']==root.name and m['output_root']==b['run_root']==str(root)
         and b['uid']==22182 and b['role']==role+'__'+root.name
         and b['registered_parent']==str(root.parent)
         and m['authority']['protocol_ref']==s.TASK_SOURCE and m['authority']['scope_grant_sha256']==s.GRANT_SHA,
         'EXPLICIT_REGISTERED_SITE_ROLE_NO_OLD_CORE_ALIAS')
    neural=role==identity.NEURAL_ROLE
    need(root.parent.name==('neural-children' if neural else 'children')
         and m['execution']['profile']==(identity.NEURAL_PROFILE if neural else identity.PHYSICAL_PROFILE), 'EXACT_PROFILE_AND_ROLE')
    need(m['execution']['output_binding_sha256']==sha(root/'OUTPUT_BINDING.private.json'),'EXACT_OUTPUT_CUSTODY')
    parent=root.parent.parent
    need(sha(member(parent,'OUTPUT_BINDING.private.json'))==PARENT_BINDING_SHA
         and Path(read(parent/'OUTPUT_BINDING.private.json')['registry']).resolve()==Path(b['registry']).resolve(),
         'EXACT_ORIGINAL_VANDA_PARENT_AND_CANONICAL_REGISTRY_NOT_A_COPIED_LEDGER')
    entries=[r for r in read(b['registry'])['entries'] if r.get('role')==b['role']]
    need(len(entries)==1 and entries[0]['path']==str(root) and entries[0]['run_uuid']==root.name,'ONE_STANDARD_REGISTRY_ROLE')
    lease=read(b['lease']);end=deadline(root,m,closeout=closeout,cold=not compute)
    need(lease['status']=='ACTIVE' and lease['run_uuid']==root.name and lease['authority_release']==s.TASK_SOURCE
         and lease['deadline_epoch']==end==m['authority']['deadline_epoch'], 'OWN_CURRENT_LEASE_AND_UNRESET_CLOCK')
    code=member(root,m['code']['capsule_relative']);seal=member(root,m['code']['source_seal_relative'])
    need(Path(__file__).resolve()==(code/'scripts/pi0294/pv1/mix_vanda_eval_context.py').resolve()
         and sha(seal)==m['code']['source_seal_sha256'],'EXECUTED_OWN_IMMUTABLE_CAPSULE')
    worker=code.parent/'WORKER_RUN_MANIFEST.json'
    need(os.environ.get('PV1_RUN_MANIFEST')==str(worker) and member(root,str(worker.relative_to(root))).read_bytes()==(root/'RUN_MANIFEST.json').read_bytes(),
         'STANDARD_MANIFEST_ENTRY_NOT_PATH_OVERRIDE')
    source=read(seal);need(source['source_sha']==m['code']['vla_commit'],'ONE_ENGINEERING_SOURCE')
    for n,h in source['files'].items():need(sha(member(code,n))==h,'CODE_'+n)
    frozen=m['code']['frozen_numerical_files']
    need(len(frozen)==15 and hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()==ORIGINAL_NUMERICAL_TABLE_SHA
         and all(source['files'].get(n)==h for n,h in frozen.items()),'ALL_FIFTEEN_ORIGINAL_SCIENCE_FILES')
    for n,h in m['execution']['sealed_inputs'].items():
        p=member(root,n);need(p.is_file() and p.stat().st_uid==22182 and not p.stat().st_mode&0o222 and sha(p)==h,'INPUT_'+n)
    environment=read(m['execution']['environment_registry'])
    profiles=[p for p in environment['profiles'] if p['name']==m['execution']['profile']]
    need(len(profiles)==1 and a.sha(a.raw(profiles[0]))==m['execution']['environment_profile_sha256']
         and profiles[0]['source_release']==m['code']['vla_commit'] and profiles[0]['PBS_required'] is True,
         'ACTUAL_EXISTING_ENVIRONMENT_REGISTRY_EXPLICIT_SITE_PROFILE')
    peers=paths(root,m);g,record,r=row(root,m,closeout=closeout,cold=not compute)
    need(r['T_CD']==m['authority']['T_CD'] and r['deadline_epoch']==m['authority']['deadline_epoch'],'SAME_REAL_ADOPTED_WINDOW')
    from .dataset import POPULATION
    from .mix_evaluation import candidate_plan,plan,stage
    pop=member(root,'plans/PV1_DATA_PLAN_AND_SPLIT_SEAL.json');groups=read(pop)['groups']
    need(sha(pop)==POPULATION and read(root/'plans/SOURCE_EVALUATION_PLAN.json')==candidate_plan(groups,r['evaluation_candidate'])
         and read(root/'plans/V5_CD_PAIRED_EVALUATION_PLAN.json')==plan(groups),
         'ORIGINAL_FROZEN_24_CONDITIONS_AND_NOISE_NO_SUBSET')
    need(m['candidate']==r['evaluation_candidate'],'EXPLICIT_CD_IDENTITY')
    need(read(root/'PV1_STAGE_BINDING.json')==stage(r['evaluation_candidate'],colocated=True)
         and m['execution']['sealed_inputs'].get('plans/SOURCE_EVALUATION_PLAN.json')==r['planned_panel_sha256'],
         'ORIGINAL_FROZEN_STAGE_AND_QUERY_PLAN')
    if neural:
        bindings=read(root/'NEURAL_BINDINGS.private.json');inputs=read(root/'plans/PV1_TRAIN_INPUTS.private.json')
        need(m['code']['official_gr00t_commit']=='51d4c89f72fda44cbf77285c6a8114b52676b8a1'
             and Path(bindings['source']).is_dir() and Path(bindings['python']).is_file(),'EXISTING_OFFICIAL_MODEL_RUNTIME')
        need(all(sha(Path(bindings['cosmos'])/n)==h for n,h in COSMOS_PROCESSOR_PINS.items()),'ORIGINAL_PROCESSOR_AND_TOKENIZER')
        from .mix_vanda_eval_output import cosmos_view
        need(cosmos_view(root)==root/'model_inputs','ORIGINAL_OFFICIAL_BACKBONE_ALIAS_REQUIRED_BEFORE_ENTRY')
        from scripts.pi0294.e1.scope import DATA,STATS
        from scripts.pi0294.dec1.authority import BANK
        need(sha(Path(bindings['dataset'])/'DATASET_MANIFEST.json')==DATA
             and sha(Path(bindings['dataset'])/'statistics.json')==STATS
             and sha(Path(inputs['new_dataset'])/'DATASET_MANIFEST.json')==BANK
             and sha(Path(inputs['new_dataset'])/'statistics.json')==STATS,'ORIGINAL_TRAIN_METADATA_AND_R0')
        meta=read(root/'plans/EVALUATION_CHECKPOINT.private.json');view=root/'model_views'/r['evaluation_candidate']
        plan=read(root/'plans/VANDA_READONLY_INFERENCE_REFERENCE.private.json')
        need(meta['checkpoint']==str(view) and meta['full_resume_files_present'] is False and meta['global_step']==1024
             and meta['checkpoint_manifest_sha256']==r['checkpoint_manifest_sha256']==sha(view/'CHECKPOINT_COMPLETE.json')
             and plan['checkpoint_manifest_sha256']==r['checkpoint_manifest_sha256'],'ONLY_ORIGINAL_CLOSED_INFERENCE_REFERENCE')
        from .mix_vanda_reference import metadata
        for n,x in plan['files'].items():
            link=view/n
            need(link.is_symlink() and link.resolve()==Path(x['path']) and metadata(x['path'])['bytes']==x['bytes'],
                 'ONLY_NINE_EXACT_READONLY_MODEL_REFERENCES')
    else:
        from .mix_vanda_codec import validate as validate_codec
        validate_codec(root,m)
        assets=read(root/'plans/ASSETS.private.json');expected=read(root/'plans/ORIGINAL_ASSET_SHA_TABLE.json')
        need(len(assets['files'])==len(expected)==26 and assets['files']==expected
             and hashlib.sha256(json.dumps(expected,sort_keys=True,separators=(',',':')).encode()).hexdigest()==ASSET_TABLE_SHA
             and all(sha(Path(assets['root'])/n)==h for n,h in expected.items()), 'ORIGINAL_ALL_NATIVE_ASSETS_RELOCATED_EXPLICITLY')
        need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('LIBGL_ALWAYS_SOFTWARE')=='1'
             and os.environ.get('MUJOCO_GL')=='egl' and 'MUJOCO_EGL_DEVICE_ID' not in os.environ
             and os.environ.get('__EGL_VENDOR_LIBRARY_FILENAMES')=='/usr/share/glvnd/egl_vendor.d/50_mesa.json', 'ORIGINAL_MESA_CPU_PHYSICS_ENVIRONMENT')
    if compute and not closeout:current_allocation(root,r)
    if not closeout:need(not any(p.exists() for p in (root/'STOP.json',root.parent.parent/'STOP.json',Path(peers['core']['root'])/'SOURCE_PHYSICS_STOP.json')),'ORIGINAL_STOP_FENCES')
    return m,(bindings if neural else assets),(inputs if neural else {}),r,g
