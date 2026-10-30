"""Finite cold metadata preparation, invoked by the existing CPU-only launcher.

No canonical GLOBAL/LOCAL adoption, qsub, model load, physics or automatic retry.
The native owner still performs actual queue, quota and registration admission.
"""
import base64,json,os,resource,time
from copy import deepcopy
from pathlib import Path
from . import minimal_e2only_cold2 as n,minimal_e2only_execution as x

raw=n.e.raw;sha=n.e.sha;need=n.need
ROLE='PI0294_ARX5_E2_ONLY_RAW512_02_COLD02'
ENV='plans/A40_ENVIRONMENT_BINDING.private.json'
LOCAL='plans/MINIMAL_CORE_LOCAL_CAPTURE.private.json'
PACKET='plans/MINIMAL_FROZEN_INPUT_PACKET.private.json'


def metadata(registry,environment,binding,manifest_sha256):
    """Replace exactly the old closed profile; retain all other registrations."""
    reg,env=deepcopy(registry),deepcopy(environment)
    old=Path(binding['run_root']);root=old.with_name(n.RUN)
    need(old.name==n.OLD_RUN and old.is_absolute(),'E2_COLD02_EXACT_OLD_REGISTERED_ROOT')
    profiles=[p for p in env['profiles']if p['name']==x.PROFILE]
    need(len(profiles)==1 and profiles[0]['manifest']==str(old/'RUN_MANIFEST.json')
         and profiles[0]['role']==binding['role']
         and not any(p.get('role')==ROLE or p.get('path')==str(root)for p in reg['entries']),
         'E2_COLD02_ONE_CLOSED_PROFILE_NO_EXISTING_NEW_ROLE')
    env.setdefault('closed_profile_history',[]).append(deepcopy(profiles[0]))
    profile=deepcopy(profiles[0]);profile.update(role=ROLE,manifest=str(root/'RUN_MANIFEST.json'))
    env['profiles']=[profile if p['name']==x.PROFILE else p for p in env['profiles']]
    reg['entries'].append(dict(role=ROLE,scope=x.VERSION,path=str(root),run_uuid=n.RUN,
        executor_node=os.uname().nodename,owner='EXISTING_SOLE_MUTATION_OWNER',lifecycle='PRE_NATIVE_COLD_NO_GPU'))
    b=dict(binding,run_root=str(root),role=ROLE,run_manifest_sha256=manifest_sha256,
        lease=str(Path(binding['lease']).with_name('ARX5-E2-ONLY-RAW512-02-COLD02.json')))
    return reg,env,b


def lease(old,binding):
    need(old['status']=='CLOSED'and old['run_uuid']==n.OLD_RUN,'E2_COLD02_CLOSED_LEASE_STAYS_CLOSED')
    # Explicit allowlist: no old stop/closure fields survive in the new lease.
    return dict(authority_release=old['authority_release'],deadline_epoch=old['deadline_epoch'],
        owner=old['owner'],status='ACTIVE',run_uuid=n.RUN,role=ROLE,
        manifest_sha256=binding['run_manifest_sha256'],CPU_upper_s=2600,
        original_paid_owner_bank_s=old['original_paid_owner_bank_s'],new_GPU_permission=False)


def prepare(P):
    """One invocation in an original finite child; partial effects are retained."""
    began=time.process_time();resource.setrlimit(resource.RLIMIT_CPU,(178,179))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
    need(os.getuid()==22182 and os.uname().nodename.startswith('stdct-login-')
         and not os.environ.get('PBS_JOBID')and os.environ.get('CUDA_VISIBLE_DEVICES')==''
         and time.time()<1793404800,'E2_COLD02_APPROVED_FINITE_CPU_ONLY_WINDOW')
    from scripts.pi0294.generalization import minimal_remaining_physical as inventory
    from scripts.pi0294.dec1.source_cpu_once_io import commit
    old=Path(P['old_root']);root=old.with_name(n.RUN);source=Path(P['source_root'])
    need(Path(__file__).resolve()==source/'code/scripts/pi0294/generalization/minimal_e2only_cold2_prepare.py'
         and not root.exists()and not any(p.is_symlink()for p in(root,*root.parents)),
         'E2_COLD02_ONLY_NEW_ROOT_OWN_PUBLISHED_EXECUTING_SOURCE')
    mb=(old/'RUN_MANIFEST.json').read_bytes();m=json.loads(mb)
    need(sha(mb)==n.OLD_MANIFEST and str(old)==m['output_root'],'E2_COLD02_ACTUAL_OLD_MANIFEST')
    oldb=json.loads((old/'OUTPUT_BINDING.private.json').read_bytes());lb=Path(oldb['lease']).read_bytes()
    need(sha(lb)==n.LEASE,'E2_COLD02_REAL_CLOSED_LEASE')
    snapshot=P['snapshot'];current=inventory.prefix_inventory(old);closed=n.closed_role(P['closed_capture']['record'])
    need(sha(raw(snapshot))==closed['snapshot_sha256']
         and inventory.identity(snapshot)==inventory.identity(current)==closed['snapshot_identity_sha256'],
         'E2_COLD02_FROZEN_OLD_ROOT_FULL_IDENTITY')
    small={name:(old/name).read_bytes()for name in m['execution']['sealed_inputs']}
    need(all(sha(body)==m['execution']['sealed_inputs'][name]for name,body in small.items()),
         'E2_COLD02_ALL_OLD_SEALED_INPUT_BYTES')
    cfg=json.loads(small['MIX_OWNER_CONFIG.private.json']);prior=deepcopy(P['old_request'])
    need(all(cfg['request_template'].get(k)==v for k,v in prior.items()
         if k not in {'created_epoch','cold_input_audit_sha256','effective_queue_proof_sha256','submit_count','run_manifest_sha256'}),
         'E2_COLD02_ARCHIVED_ACTUAL_REQUEST_TEMPLATE')
    gp=old.parent.parent/'HOPPER_RESOURCE_LEDGER.json';gb=gp.read_bytes()
    local=base64.b64decode(P['LOCAL_base64'],validate=True);sb=(source/'SOURCE_SEAL.json').read_bytes();seal=json.loads(sb)
    for name,pin in seal['files'].items():
        need(not Path(name).is_absolute()and '..'not in Path(name).parts,
             'E2_COLD02_SOURCE_RELATIVE_MEMBER')
        f=source/'code'/name
        need(not f.is_symlink()and f.stat().st_uid==os.getuid()and f.stat().st_mode&0o777==0o400
             and sha(f.read_bytes())==pin,'E2_COLD02_PUBLISHED_SOURCE_FULL_BYTES:'+name)
    scope=n.prepare_scope(gb,local,authority_bytes=base64.b64decode(P['authority_base64'],validate=True),
        closed_capture=P['closed_capture'],old_request=prior,old_manifest=m,
        old_physical_custody=json.loads(small[x.PHYSICAL]),source_seal_bytes=sb,
        frozen=m['code']['frozen_numerical_files'],source_readback=P['source_readback'],now_epoch=time.time())
    regpath,envpath=Path(oldb['registry']),Path(oldb['environment']);regb,envb=regpath.read_bytes(),envpath.read_bytes()
    need(sha(regb)==P['registry_sha256']and sha(envb)==P['environment_sha256'],
         'E2_COLD02_FRESH_EXACT_REGISTRY_ENV_BEFORE_BYTES')
    reg,env,binding=metadata(json.loads(regb),json.loads(envb),oldb,'0'*64)
    oldenv=small[ENV];envbinding=json.loads(oldenv);envbinding['environment_registry_sha256']=sha(raw(env));small[ENV]=raw(envbinding)
    p=json.loads(small[x.PHYSICAL]);p['run_uuid']=n.RUN
    p['retained_initial_bytes']+=max(current['allocated_bytes'],snapshot['allocated_bytes'])
    p['closed_pre_native_roots'].append(dict(root=str(old),closure=closed,snapshot_file=n.SNAPSHOT))
    small[x.PHYSICAL]=raw(p);small[n.BINDING]=raw(scope);small[n.SNAPSHOT]=raw(snapshot)
    capture=json.loads(small[LOCAL]);capture.update(LOCAL_base64=base64.b64encode(local).decode(),LOCAL_sha256=sha(local),
        source='ACTUAL_CANONICAL_CORE_LOCAL_READONLY',capture_is_fresh_native_authority=False)
    small[LOCAL]=raw(capture)
    for name in ['SOURCE_SEAL.json','MIX_OWNER_CONFIG.private.json',PACKET]:small.pop(name,None)
    small[PACKET]=raw({name:sha(body)for name,body in small.items()})
    request=deepcopy(prior);request.update(x.fixed_request_fields(2,2),source_sha=seal['source_sha'],
        pre_native_owner_CPU_upper_s=2600,closed_pre_native_role=closed,**{n.TAG:n.digest(scope)},
        physical_custody_sha256=sha(small[x.PHYSICAL]),frozen_input_packet_sha256=sha(small[PACKET]))
    need(not any(str(old)in body for body in cfg['programs'].values()),
         'E2_COLD02_ORIGINAL_AUTH_QUOTA_SUBMIT_PROGRAMS_NO_LITERAL_OLD_ROOT')
    # The final manifest cannot be included in its own sealed configuration.
    template=deepcopy(request);template.pop('run_manifest_sha256',None)
    cfg.update(source_sha=seal['source_sha'],request_template=template)
    small['MIX_OWNER_CONFIG.private.json']=raw(cfg);small['SOURCE_SEAL.json']=sb
    manifest=deepcopy(m);manifest.update(run_uuid=n.RUN,output_root=str(root))
    manifest['code'].update(vla_commit=seal['source_sha'],capsule_relative='capsules/'+seal['source_sha']+'/code',source_seal_sha256=sha(sb))
    manifest['execution']['sealed_inputs']={name:sha(body)for name,body in small.items()}
    binding['run_manifest_sha256']=sha(raw(manifest));request['run_manifest_sha256']=binding['run_manifest_sha256']
    need(not Path(binding['lease']).exists(),'E2_COLD02_PRESERVE_EXISTING_LEASE_NO_REPLAY')
    def put(path,body):
        path=Path(path);path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
        with os.fdopen(fd,'wb')as f:f.write(body);f.flush();os.fsync(f.fileno())
    root.mkdir(mode=0o700)
    for name in ['control','pbs','evidence','reservations','training','datasets']:(root/name).mkdir(mode=0o700)
    code=root/manifest['code']['capsule_relative']
    for name in seal['files']:put(code/name,(source/'code'/name).read_bytes())
    for name,body in small.items():put(root/name,body)
    from scripts.pi0294.pv1.model_inputs import prepare_alias
    prepare_alias(root,json.loads(small['NEURAL_BINDINGS.private.json']))
    put(root/'RUN_MANIFEST.json',raw(manifest));put(root/'OUTPUT_BINDING.private.json',raw(binding))
    from .minimal_saved512_inputs import Inputs
    Inputs(root,manifest,guard=lambda:None).verify_current()
    auditb=Path(P['original_audit_path']).read_bytes();audit=json.loads(auditb)
    need(sha(auditb)==prior['cold_input_audit_sha256']and audit['actual_twelve_reader_rows']==7212,
         'E2_COLD02_EXACT_ORIGINAL_READER_QUALIFICATION_NOT_NEW_TRAINING')
    audit=dict(status='PASS_ACTUAL_MINIMAL_FIRST512_COLD_INPUTS_NO_MODEL',source_sha=seal['source_sha'],
        run_manifest_sha256=sha(raw(manifest)),source_files=len(seal['files']),sealed_inputs=len(small),actual_twelve_reader_rows=7212,
        original_reader_qualification_reused=dict(audit_sha256=sha(auditb),own_input_bytes_reverified=True),
        new_GPU_PBS_model_query_update_physics=0,observed_epoch=time.time(),CPU_actual_s=time.process_time()-began)
    request.update(cold_input_audit_sha256=sha(raw(audit)),created_epoch=time.time())
    x.request(request)
    from .minimal_e2only_physical import Guard,CONTROL_GROWTH_RESERVE
    physical=Guard(root,request,manifest,lambda **_:None).inspect(next_output_upper_bytes=CONTROL_GROWTH_RESERVE)
    proposal=n.propose(gb,local,scope=scope,manifest_bytes=raw(manifest),physical_bytes=small[x.PHYSICAL],
        request=request,environment_binding_bytes=(oldenv,small[ENV]))
    need(gp.read_bytes()==gb and regpath.read_bytes()==regb and envpath.read_bytes()==envb,
         'E2_COLD02_NO_CONCURRENT_CANONICAL_MUTATION')
    put(root/'control/COLD_INPUT_AUDIT.private.json',raw(audit))
    put(root/'control/COLD_FIXED_REQUEST_PREVIEW.private.json',raw(request))
    put(root/'control/COLD_FULL_FUTURE_PREVIEW.private.json',raw(proposal['full_future_proof']))
    put(root/'control/PROPOSED_GLOBAL_NOT_CANONICAL.private.json',raw(proposal['proposed_global']))
    put(binding['lease'],raw(lease(json.loads(lb),binding)))
    for path,before,after,key in [(regpath,regb,raw(reg),'E2-COLD02-REGISTER-ROLE'),(envpath,envb,raw(env),'E2-COLD02-ENV-PROFILE')]:
        commit(path,expected_before_sha256=sha(before),proposed_bytes=after,request_id=key,
            intent_path=path.with_name(key+'.INTENT.json'),receipt_path=path.with_name(key+'.RECEIPT.json'),lock_path=path.with_suffix('.owner.lock'))
        need(path.read_bytes()==after,'E2_COLD02_SAME_PARENT_ONCE_IO_FULL_READBACK')
    need(gp.read_bytes()==gb and all(sha((root/name).read_bytes())==pin for name,pin in manifest['execution']['sealed_inputs'].items())
         and all(sha((code/name).read_bytes())==pin for name,pin in seal['files'].items()),
         'E2_COLD02_ALL_FINAL_BYTES_NO_CANONICAL_ADOPTION')
    return dict(status='ACTUAL_E2_COLD02_FINITE_PREPARATION_FULL14_NO_NATIVE',root=str(root),
        source_sha=seal['source_sha'],manifest_sha256=sha(raw(manifest)),scope_sha256=n.digest(scope),
        proposed_GLOBAL_sha256=sha(raw(proposal['proposed_global'])),physical_proof=physical,full14_shortfalls={},
        CPU_s=time.process_time()-began,observed_epoch=time.time(),new_model_GPU_PBS_physics=0,
        native_permission=False,canonical_adoption=False,automatic_retry=False)
