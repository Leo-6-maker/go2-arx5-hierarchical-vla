"""Registered raw512 peers on the existing node-local Core runtime."""
import hashlib,json,os,time,sys,tempfile
from pathlib import Path
from scripts.pi0292.durable import read
from scripts.pi0294.native_dataset import sha
from . import minimal_core_accounting as c, minimal_core_identity as identity
from scripts.pi0294.pv1 import node_local_custody as custody
from scripts.pi0294.pv1.mix_core_context import LOCAL_BINDING
from scripts.pi0294.pv1.evaluation_core_scope import member,PARENT_BINDING_SHA,ORIGINAL_NUMERICAL_TABLE_SHA,COSMOS_PROCESSOR_PINS
from scripts.pi0294.pv1.successor_core_context import ASSET_IDENTITY_SHA

need=c.need


def ledger_binding(root,*,closeout=False):
    root=Path(root);m=read(root/'RUN_MANIFEST.json')
    need(c.typed(m) and m['execution']['sealed_inputs'].get(LOCAL_BINDING)==sha(root/LOCAL_BINDING),
         'EXPLICIT_TYPED_CANONICAL_LOCAL_BINDING')
    path=Path(read(root/LOCAL_BINDING)['canonical_LOCAL_ledger']).resolve()
    need(path.name=='LOCAL_RESOURCE_LEDGER.private.json' and path.parent==root.parent.parent/'control',
         'ORIGINAL_CANONICAL_LOCAL_NOT_A_COPIED_LEDGER')
    local=read(path);need(local['unknown_effects']==local['pending_updates']==local['pending_queries']==[],
         'LOCAL_UNKNOWN_AND_PENDING_FENCES')
    rid=m['authority']['request_id'];row=local.get(c.FIELD,{}).get(rid)
    need(type(row) is dict,'ACTUAL_ONCE_RESERVED_RAW512_ALLOCATION')
    c.validate_row(row);physical=root.parent.name=='children'
    need(row['child_run_uuid']==root.name and row['source_sha']==m['code']['vla_commit']
        and row['status']!=c.STATES[3] and row['scope_grant_sha256']==c.GRANT_SHA
        and row['physical_manifest_sha256' if physical else 'run_manifest_sha256']==sha(root/'RUN_MANIFEST.json'),
        'EXACT_LIVE_RESERVED_PEER_AND_IMMUTABLE_MANIFEST')
    now=time.time();need(row['T_CD']<=now and (closeout or now<row['deadline_epoch']), 'ACTUAL_DECLARED_WINDOW')
    return dict(T_CD=row['T_CD'],deadline_epoch=row['deadline_epoch'],scope_grant_sha256=c.GRANT_SHA),path


def deadline(root,manifest,*,closeout=False):
    marker,_=ledger_binding(root,closeout=closeout)
    need(manifest['limits']['operation_deadline_epoch']==marker['deadline_epoch'],'SAME_FINITE_WINDOW')
    return marker['deadline_epoch']


def validate(root,*,closeout=False):
    need(type(closeout) is bool,'EXPLICIT_PHASE')
    root=Path(root)
    need(root.is_absolute() and root.is_dir() and not root.is_symlink() and root.stat().st_uid==os.getuid()==2009
        and os.uname().nodename=='corelab-4080s','ACTUAL_OWN_CORE_ROOT')
    for p in (root,Path(sys.executable),Path(sys.prefix),Path.cwd(),Path(tempfile.gettempdir())):
        custody.local_path(p.resolve())
    for name in ('HF_HOME','HUGGINGFACE_HUB_CACHE','XDG_CACHE_HOME','TORCH_HOME','CUDA_CACHE_PATH'):
        need(os.environ.get(name) is not None,'EXPLICIT_LOCAL_CACHE_'+name)
        custody.local_path(Path(os.environ[name]).resolve())
    for name,module in list(sys.modules.items()):
        if name.split('.')[0] in ('numpy','PIL','mujoco','torch','gr00t','transformers') and getattr(module,'__file__',None):
            custody.local_path(Path(module.__file__).resolve())
    for name in ('RUN_MANIFEST.json','OUTPUT_BINDING.private.json'):
        p=member(root,name);need(p.is_file() and p.stat().st_uid==2009 and not p.stat().st_mode&0o222,'IMMUTABLE_'+name)
    m=read(root/'RUN_MANIFEST.json');b=read(root/'OUTPUT_BINDING.private.json');physical=root.parent.name=='children'
    expected_role=c.PHYSICAL_ROLE if physical else c.ROLE
    expected_profile='4080-simulation' if physical else c.PROFILE
    need(c.typed(m) and m.get('scope')==expected_role and m.get('candidate')=='E1'
        and m['execution']['profile']==expected_profile
        and m['authority']['protocol_ref']==c.campaign.TASK_SOURCE and m['authority']['scope_grant_sha256']==c.GRANT_SHA
        and m['run_uuid']==root.name and m['output_root']==str(root)
        and b['run_root']==str(root) and b['run_uuid']==root.name and b['uid']==2009
        and b['role']==expected_role+'__'+root.name and m['execution']['private_path_role']==b['role']
        and Path(b['registered_parent'])==root.parent and root.parent.name in ('children','neural-children'),
        'EXPLICIT_RAW512_ROLE_AND_CANONICAL_REGISTERED_OUTPUT')
    if physical:need(b['run_manifest_sha256']==sha(root/'RUN_MANIFEST.json'),'PHYSICAL_BINDING_MANIFEST')
    else:need(m['execution']['output_binding_sha256']==sha(root/'OUTPUT_BINDING.private.json'),'NEURAL_BINDING_MANIFEST')
    parent=root.parent.parent
    need(sha(parent/'OUTPUT_BINDING.private.json')==PARENT_BINDING_SHA,'ORIGINAL_CAMPAIGN_PARENT')
    custody.registration(root,m,b,legacy_read=read)
    code=member(root,m['code']['capsule_relative']);seal=member(root,m['code']['source_seal_relative'])
    need(Path(__file__).resolve()==code/'scripts/pi0294/generalization/minimal_core_context.py'
        and sha(seal)==m['code']['source_seal_sha256'],'ACTUAL_OWN_SEALED_CAPSULE')
    worker=code.parent/'WORKER_RUN_MANIFEST.json'
    need(os.environ.get('PV1_RUN_MANIFEST')==str(worker) and not worker.stat().st_mode&0o222
        and worker.read_bytes()==(root/'RUN_MANIFEST.json').read_bytes(),'STANDARD_IMMUTABLE_WORKER_ENTRY')
    source=read(seal);need(source['source_sha']==m['code']['vla_commit'],'EXACT_SOURCE')
    for name,pin in source['files'].items():need(sha(member(code,name))==pin,'SOURCE_'+name)
    frozen=m['code']['frozen_numerical_files']
    need(len(frozen)==15 and hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()==ORIGINAL_NUMERICAL_TABLE_SHA
        and all(source['files'].get(n)==pin and sha(member(code,n))==pin for n,pin in frozen.items()),
        'ALL_ORIGINAL_CONTROLLER_PHYSICS_S1_SCORE_NUMERICS')
    for name,pin in m['execution']['sealed_inputs'].items():
        p=member(root,name);need(p.is_file() and p.stat().st_uid==2009 and not p.stat().st_mode&0o222 and sha(p)==pin,'SEALED_'+name)
    def sealed(name):
        need(name in m['execution']['sealed_inputs'],'REQUIRED_INPUT_'+name)
        return read(member(root,name))
    need(sealed(c.GRANT_FILE)==c.grant() and sha(root/c.GRANT_FILE)==c.GRANT_SHA,'EXACT_DIRECT_SCIENCE_SCOPE')
    marker,path=ledger_binding(root,closeout=closeout);local=read(path);row=local[c.FIELD][m['authority']['request_id']]
    request=row['request']
    if not physical:
        identity.request_identity(m,request,sha(root/'RUN_MANIFEST.json'))
        rp=root/'pbs/alloc-01/REQUEST.json'
        need(sha(rp)==row['request_sha256'] and read(rp)==request,'EXACT_IMMUTABLE_NATIVE_REQUEST')
    spec=c.plan.session_spec(root,'E1-DEV-N294101')
    need(m['execution']['sealed_inputs'][c.plan.PLAN]==request['planned_panel_sha256'] and len(spec['cards'])==24,
         'ORIGINAL_24_NOT_A_SELECTED_PANEL')
    sealed('plans/PRIOR_EVALUATION_OUTCOMES.json')
    custody.native_assets(root,m,expected_original_sha256=ASSET_IDENTITY_SHA)
    bindings=sealed('NEURAL_BINDINGS.private.json');inputs=sealed('plans/PV1_TRAIN_INPUTS.private.json')
    for name in ('source','python','cosmos','dataset'):
        p=Path(bindings[name]);need(p.exists(),'ACTUAL_INPUT_'+name);custody.local_path(p.resolve())
    need(m['code']['official_gr00t_commit']=='51d4c89f72fda44cbf77285c6a8114b52676b8a1'
        and all(sha(Path(bindings['cosmos'])/n)==pin for n,pin in COSMOS_PROCESSOR_PINS.items()),
        'ORIGINAL_OFFICIAL_RUNTIME_AND_TOKENIZER')
    from scripts.pi0294.e1.scope import DATA,STATS
    need(sha(Path(bindings['dataset'])/'DATASET_MANIFEST.json')==DATA
        and sha(Path(bindings['dataset'])/'statistics.json')==STATS,'ORIGINAL_DATA_AND_R0')
    checkpoint=sealed('plans/EVALUATION_CHECKPOINT.private.json');model_path=Path(checkpoint['checkpoint'])
    custody.local_path(model_path.resolve())
    need(checkpoint['candidate']=='E1' and checkpoint['global_step']==512
        and checkpoint['checkpoint_manifest_sha256']==c.CHECKPOINT and checkpoint['full_resume_files_present'] is False
        and sha(model_path/'CHECKPOINT_COMPLETE.json')==c.CHECKPOINT,'FIXED_ALREADY_DELIVERED_INFERENCE_VIEW')
    lease=read(b['lease'])
    need(lease['status']=='ACTIVE' and lease['run_uuid']==root.name and lease['authority_release']==c.campaign.TASK_SOURCE
        and lease['deadline_epoch']==marker['deadline_epoch'],'ACTUAL_OWN_ACTIVE_LEASE')
    infra=custody.entry_directory(root,m,b)
    need(sha(infra/'enter.sh')==m['execution']['entry_sha256']
        and sha(infra/'doctor.py')==m['execution']['doctor_sha256'],'EXISTING_STANDARD_ENTRY_AND_DOCTOR')
    if physical:
        need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and 'PBS_JOBID' not in os.environ,'CPU_ONLY_PHYSICAL_ROLE')
        from scripts.pi0294.pv1.successor_core_context import validate_cpu_affinity
        validate_cpu_affinity(m['limits'],os.sched_getaffinity(0))
    if not closeout:
        need(not any(p.exists()for p in (root/'STOP.json',parent/'STOP.json',root/'SOURCE_PHYSICS_STOP.json',root/'SOURCE_SAFETY_STOP_FROM_CORE.json')),
             'ORIGINAL_STOP_FENCES')
    return m,bindings,inputs,sealed('PV1_STAGE_BINDING.json'),marker
