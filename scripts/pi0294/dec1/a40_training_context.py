"""Typed source-training context; never grants the historical H200 qualification."""
from copy import deepcopy
import os
from pathlib import Path
import socket
import sys
import time

from scripts.pi0292.durable import read, write
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.mp1.owner_heartbeat import read_live_owner
from . import a40_allocation as native
from .alternate_accounting import digest
from .dual_platform import grant, GRANT_SHA
from .long_training import admitted, probe_receipt

PROFILE='vanda-neural-training'
CONSUMERS='plans/A40_CLOSED_SOURCE_DATA_CONSUMERS.private.json'
QUALIFICATION='A40_PREMODEL_QUALIFICATION.json'
COMPATIBILITY='A40_COMPATIBILITY/RESULT.json'
DEFERRED='FIXED_MANIFEST_AWAIT_CLOSED_READER_V1'


def consumer_records(root,*,require_closed=True):
    """Permit queueing a fixed CPU consumer, never using its unfinished data."""
    records=sealed(root,CONSUMERS)
    if set(records)!={'new139','original126_C1'}:fail('BOTH_FIXED_TRAIN_CONSUMERS')
    resolved={}
    for kind,original in records.items():
        row=dict(original)
        if row.get('binding_mode')==DEFERRED:
            if kind!='original126_C1' or set(row)!={'root','manifest_sha256','binding_mode'}:
                fail('ONLY_PREBOUND_ORIGINAL_CPU_CONSUMER_MAY_FINISH_IN_QUEUE')
            child=Path(row['root'])
            manifest=sealed(child,'RUN_MANIFEST.json')
            binding=sealed(child,'OUTPUT_BINDING.private.json')
            if (sha(child/'RUN_MANIFEST.json')!=row['manifest_sha256']
                    or binding.get('run_manifest_sha256')!=row['manifest_sha256']
                    or manifest.get('source_data_profile')!='ORIGINAL126_C1'
                    or manifest.get('authority_release')!=native.AUTHORITY
                    or manifest.get('owner')!='EXISTING_LOCAL_PC'
                    or manifest.get('owner_uid')!=os.getuid()):
                fail('ORIGINAL_QUEUED_CPU_MANIFEST_IDENTITY')
            if (child/'RESULT.private.json').exists():
                result=sealed(child,'RESULT.private.json')
                if result.get('status')!='PASS_VANDA_CPU_ORIGINAL126_AND_C1_FIXED_MEMBERS_AND_OFFICIAL_READER':
                    fail('CPU_READER_TERMINAL_FAILURE_BEFORE_QUEUE_OR_MODEL')
            if require_closed:
                result=sealed(child,'RESULT.private.json')
                lease_path=Path(binding['lease'])
                lease=sealed(lease_path.parent,lease_path.name)
                if lease.get('status')!='CLOSED':fail('CPU_READER_NOT_CLOSED_BEFORE_MODEL')
                row.update(result_sha256=sha(child/'RESULT.private.json'),lease_sha256=sha(lease_path))
        elif 'binding_mode' in row:fail('UNKNOWN_CPU_CONSUMER_BINDING_MODE')
        resolved[kind]=row
    return resolved


def fail(message):raise PermissionError('VAP1_A40_CONTEXT_'+message)


def sealed(root,name,*,parse=True):
    root=Path(root);relative=Path(name);p=root/relative
    if (relative.is_absolute() or '..' in relative.parts or p.is_symlink()
            or not p.resolve().is_relative_to(root.resolve()) or not p.is_file()
            or p.stat().st_uid!=os.getuid() or p.stat().st_mode&0o777!=0o400):
        fail('OWN_IMMUTABLE_BOUND_FILE:'+str(name))
    return read(p) if parse else p


def validate_root(root,request_path,*,verify_all=False):
    root=Path(root);request_path=Path(request_path)
    m=sealed(root,'RUN_MANIFEST.json');b=sealed(root,'OUTPUT_BINDING.private.json')
    if (root.is_symlink() or root.stat().st_uid!=os.getuid()
            or m.get('schema')!='PI0294_PV1_NEURAL_CHILD_RUN_MANIFEST_V1'
            or m.get('platform_training_version')!=native.VERSION
            or m.get('owner')!='EXISTING_LOCAL_PC' or m.get('owner_uid')!=os.getuid()
            or m.get('owner_user')!='e1192548' or m.get('run_uuid')!=root.name
            or m.get('output_root')!=str(root) or b.get('run_root')!=str(root)
            or b.get('uid')!=os.getuid() or b.get('run_manifest_sha256')!=sha(root/'RUN_MANIFEST.json')
            or m['execution'].get('profile')!=PROFILE
            or not root.resolve().is_relative_to(Path(m['registered_runs_root']).resolve())
            or request_path.is_symlink() or request_path.name!='REQUEST.json'
            or request_path.parent.parent.resolve()!=(root/'pbs').resolve()):
        fail('REGISTERED_OWN_TYPED_TRAINING_ROOT_AND_REQUEST')
    r=sealed(root,request_path.relative_to(root).as_posix());native.request_scope(r,now=time.time())
    if (r['run_uuid']!=root.name or r['source_sha']!=m['code']['vla_commit']
            or r['run_manifest_sha256']!=sha(root/'RUN_MANIFEST.json')
            or request_path.parent.name!='alloc-'+str(r['submit_count']).zfill(2)
            or m['authority'].get('protocol_ref')!=native.PARENT_AUTHORITY
            or m['authority'].get('amendments')!=[native.AUTHORITY]
            or m['authority'].get('training_admission_version')!=native.SCIENCE_VERSION):
        fail('EXACT_PUBLISHED_SCIENCE_SOURCE_AND_NATIVE_REQUEST')
    lease=read(b['lease']);rows=[x for x in read(b['registry'])['entries'] if x.get('role')==b['role']]
    if (lease.get('status')!='ACTIVE' or lease.get('run_uuid')!=root.name
            or lease.get('authority_release')!=native.PARENT_AUTHORITY
            or lease.get('original_T0_epoch')!=native.T0 or lease.get('deadline_epoch')!=native.HARD_CLOSE
            or len(rows)!=1 or Path(rows[0]['path']).resolve()!=root.resolve()):
        fail('ONE_ACTIVE_EXISTING_OWNER_LEASE_AND_REGISTRY')
    inputs=m['execution']['sealed_inputs']
    from .segmented_training import enabled,admitted as segment_admitted
    if enabled(r):segment_admitted(root,m,r)
    from .storage_expansion import admitted as storage_admitted
    storage_admitted(root,m,r)
    from .heartbeat_recovery import admitted as heartbeat_admitted
    heartbeat_admitted(root,m,r)
    from .zero_update_recovery import admitted as zero_admitted
    zero_admitted(root,m,r)
    from .b_heartbeat_recovery import admitted as b_admitted
    b_admitted(root,m,r)
    from .b_vanda_recovery2 import admitted as vanda2_admitted
    vanda2_admitted(root,m,r)
    if r.get('recovery_authorization_release') is not None:
        from .lost_save_recovery import PLAN,RELEASE,grant as recovery_grant
        if (m['authority'].get('recovery_authorization_release')!=RELEASE
                or PLAN not in inputs or sha(sealed(root,PLAN,parse=False))!=inputs[PLAN]
                or (root/PLAN).read_bytes()!=(root.parent.parent/'authority/DEC1'/Path(PLAN).name).read_bytes()):
            fail('EXACT_NATIVE_AND_MANIFEST_SEALED_RECOVERY_AUTHORIZATION')
        recovery_grant((root/PLAN).read_bytes())
    required={CONSUMERS,'SOURCE_SEAL.json','NEURAL_BINDINGS.private.json','plans/PV1_TRAIN_INPUTS.private.json',
        'plans/VAP1_DUAL_PLATFORM_CONCURRENCY_ADOPTION.json','plans/PUBLICATION_AND_CI.json',
        'plans/PRIMARY_LEARNING_RECIPES.json','plans/FROZEN_SOURCE_PAIR_RECIPE.json',
        'plans/INITIAL_CORR_PARAMETER_INVENTORY.json','plans/PV1_FIXED_TRAIN_PROBES.json',
        'plans/PROBE_OBSERVATION.private.json','POLICY_IDENTITY.json','ASSET_CONSUMPTION.json',
        'plans/A40_ENVIRONMENT_BINDING.private.json','plans/VAP1_ADOPTION_AND_DELTA_BUDGET.json'}
    if not required<=set(inputs) or not set(inputs)<=set(r['sealed_inputs']):
        fail('EXPLICIT_SCIENCE_DATA_CODE_AND_PLATFORM_INPUTS')
    for name in (inputs if verify_all else required):
        sealed(root,name,parse=False)
        if inputs[name]!=sha(root/name) or r['sealed_inputs'].get(name)!=inputs[name]:
            fail('ORIGINAL_MANIFEST_AND_REQUEST_SEALED_BYTES:'+name)
    proof=sealed(root,'plans/PUBLICATION_AND_CI.json')
    if (proof.get('source_sha')!=r['source_sha'] or proof.get('CI_head_sha')!=r['source_sha']
            or proof.get('CI_conclusion')!='success' or proof.get('CI_status')!='completed'
            or proof.get('files_exact_readback') is not True):fail('EXACT_SOURCE_CI_AND_READBACK')
    source=sealed(root,'SOURCE_SEAL.json');members=source.get('files',{})
    prefix=Path(m['code']['capsule_relative'])
    critical={str(prefix/'scripts/pi0294/dec1'/name) for name in (
        'a40_training_context.py','a40_training_runtime.py','a40_native_ledger.py','a40_allocation.py')}
    if (source.get('source_sha')!=r['source_sha'] or not critical<=set(members)
            or m['code'].get('source_seal_sha256')!=sha(root/'SOURCE_SEAL.json')
            or any(inputs.get(name)!=digest or not Path(name).is_relative_to(prefix)
                for name,digest in members.items())):
        fail('FULL_ORIGINAL_CAPSULE_SEAL_NOT_JUST_A_COMMIT_LABEL')
    for name in members if verify_all else critical:
        sealed(root,name,parse=False)
        if sha(root/name)!=members[name]:fail('EXECUTING_SOURCE_MEMBER_CHANGED:'+name)
    grant_path=root/'plans/VAP1_DUAL_PLATFORM_CONCURRENCY_ADOPTION.json'
    grant(grant_path.read_bytes())
    native_grant=root.parent.parent/'authority/VAP1/VAP1_DUAL_PLATFORM_CONCURRENCY_ADOPTION.json'
    if native_grant.read_bytes()!=grant_path.read_bytes():fail('ACTUAL_NATIVE_GRANT_NOT_JUST_A_PLAN')
    env=sealed(root,'plans/A40_ENVIRONMENT_BINDING.private.json');bindings=sealed(root,'NEURAL_BINDINGS.private.json')
    if (env.get('schema')!='VAP1_A40_REUSED_ENVIRONMENT_BINDING_V1'
            or env.get('profile')!=PROFILE or env.get('old_probe_profile_scope_changed') is not False
            or env.get('python')!=r.get('python') or Path(sys.executable).resolve()!=Path(env['python']).resolve()
            or Path(sys.prefix).resolve()!=Path(env['venv_prefix']).resolve()
            or sha(env['python'])!=env.get('python_binary_sha256')
            or sha(env['environment_registry'])!=env.get('environment_registry_sha256')
            or env.get('official_source')!=bindings['source']
            or env.get('official_commit')!='51d4c89f72fda44cbf77285c6a8114b52676b8a1'):
        fail('REUSED_ACTUAL_PYTHON_ENVIRONMENT_NOT_A_HOT_PROFILE_CHANGE')
    if not admitted(root,m,r):fail('EXPLICIT_SAME_JOB_LONG_ALLOCATION_POLICY')
    code=root/m['code']['capsule_relative']
    if Path(__file__).resolve()!=(code/'scripts/pi0294/dec1/a40_training_context.py').resolve():
        fail('EXECUTE_OWN_IMMUTABLE_CAPSULE_NOT_MUTABLE_SOURCE')
    return m,r


def current_directory(root,allocation):
    root=Path(root)
    if (allocation.get('status')!=native.STATUS or os.environ.get('PBS_JOBID')!=allocation.get('job_id')
            or socket.gethostname()!=allocation.get('hostname')
            or not socket.gethostname().split('.')[0].lower().startswith('gn-a40-')
            or not allocation['allocation_start_epoch']<=time.time()<allocation['allocation_end_epoch']-2700):
        fail('OWN_CURRENT_A40_ALLOCATION_WITH_FULL_SAVE_RESERVE')
    matches=[p for p in (root/'pbs').glob('alloc-[0-9][0-9]')
        if (p/'ALLOCATION_ADMISSION.json').is_file() and read(p/'ALLOCATION_ADMISSION.json')==allocation]
    if len(matches)!=1:fail('ONE_EXACT_NATIVE_ADMISSION_NOT_OLD_OR_TERMINAL')
    directory=matches[0];relative=directory.relative_to(root)
    sealed(root,str(relative/'ALLOCATION_ADMISSION.json'))
    if (sha(directory/'REQUEST.json')!=allocation['request_sha256']
            or (directory/'PBS_PROCESS_FINAL.json').exists()):fail('IMMUTABLE_CURRENT_REQUEST_NO_RESTART')
    return directory


def owner_guard(root,allocation):
    root=Path(root);directory=current_directory(root,allocation)
    m=sealed(root,'RUN_MANIFEST.json');b=sealed(root,'OUTPUT_BINDING.private.json');lease=read(b['lease'])
    if (lease.get('status')!='ACTIVE' or lease.get('run_uuid')!=root.name
            or (root/'STOP.json').exists() or (root.parent.parent/'STOP.json').exists()
            or m['code']['vla_commit']!=allocation['source_sha']):fail('LIVE_ROOT_SCOPE_LEASE_AND_ORIGINAL_SOURCE')
    beat=read_live_owner(root/'control',root.name)['heartbeat']
    from . import vanda_campaign as single
    resident=m.get('execution',{}).get('control_execution_platform')=='VANDA_A40'
    if resident!=(beat.get('campaign_view_schema')==single.SCHEMA):
        fail('EXACT_PRESEALED_CONTROL_PLATFORM_NOT_DROPPED_OR_INJECTED')
    if resident:
        single.validate_owner_heartbeat(beat,ledger=read(root.parent.parent/'HOPPER_RESOURCE_LEDGER.json'),now_epoch=time.time())
    if (beat.get('owner')!='EXISTING_LOCAL_PC' or beat.get('job_id')!=allocation['job_id']
            or beat.get('request_sha256')!=allocation['request_sha256']
            or beat.get('source_sha')!=allocation['source_sha'] or beat.get('global_unknown_effects')!=[]
            or beat.get('original_parent_STOP') is not False or beat.get('native_job_state')!='R'
            or beat.get('source_pair_platform')!=native.PLATFORM or beat.get('grant_sha256')!=GRANT_SHA):
        fail('FRESH_SAME_OWNER_GLOBAL_FENCE_WITHOUT_HIDING_OTHER_PROJECTS')
    return directory


def verify_consumers(root):
    from scripts.pi0294.e1.scope import DATA
    from .source_data_original import C1_MANIFEST
    root=Path(root);records=consumer_records(root)
    b=sealed(root,'NEURAL_BINDINGS.private.json');inputs=sealed(root,'plans/PV1_TRAIN_INPUTS.private.json')
    if set(records)!= {'new139','original126_C1'}:fail('BOTH_CLOSED_FIXED_TRAIN_CONSUMERS')
    actual={}
    for kind,row in records.items():
        r=Path(row['root']);binding=read(r/'OUTPUT_BINDING.private.json');manifest=read(r/'RUN_MANIFEST.json')
        for name,key in [('RUN_MANIFEST.json','manifest_sha256'),('RESULT.private.json','result_sha256')]:
            if sha(r/name)!=row[key]:fail('ACTUAL_FROZEN_CPU_CONSUMER_BYTES')
        lease=read(binding['lease'])
        registered=[v for v in read(binding['registry'])['entries'] if v.get('role')==binding['role']]
        if (r.is_symlink() or r.stat().st_uid!=os.getuid() or binding['uid']!=os.getuid()
                or not r.resolve().is_relative_to(Path(read(root/'RUN_MANIFEST.json')['registered_runs_root']).resolve())
                or binding.get('run_root')!=str(r) or binding.get('run_manifest_sha256')!=row['manifest_sha256']
                or sha(binding['lease'])!=row['lease_sha256'] or lease.get('status')!='CLOSED'
                or lease.get('run_uuid')!=r.name or lease.get('owner')!='EXISTING_LOCAL_PC'
                or lease.get('authority_release')!=native.AUTHORITY
                or lease.get('result_sha256')!=row['result_sha256']
                or len(registered)!=1 or Path(registered[0]['path']).resolve()!=r.resolve()
                or registered[0].get('result_sha256')!=row['result_sha256']
                or manifest.get('run_uuid')!=r.name or manifest.get('owner')!='EXISTING_LOCAL_PC'
                or manifest.get('owner_uid')!=os.getuid()
                or manifest.get('source_data_profile','PV1_139')!=('PV1_139' if kind=='new139' else 'ORIGINAL126_C1')
                or manifest['authority_release']!=native.AUTHORITY
                or manifest['original_T0_epoch']!=native.T0):fail('OWN_REGISTERED_CLOSED_CPU_CONSUMER')
        result=read(r/'RESULT.private.json')
        if (result.get('failure') is not None or result.get('full_member_hashes_verified') is not True
                or result.get('statistics_sha256')!=native.R0
                or any(result.get(k)!=0 for k in ('GPU','PBS','models','new_optimizer_updates','new_physics'))):
            fail('REAL_FULL_CPU_READERS_NOT_DIRECTORY_COUNTS_OR_MODEL_QUALIFICATION')
        if kind=='new139':
            valid=(result.get('status')=='PASS_VANDA_CPU_FIXED_PV1_139_TRAIN_FULL_MEMBER_AND_READER'
                and result.get('dataset_manifest_sha256')==native.BANK
                and Path(inputs['new_dataset'])==r/'datasets/FROZEN_TRAIN_BANK'
                and Path(inputs['new_raw_bindings'])==r/'RAW_BINDINGS.private.json'
                and sha(Path(inputs['new_dataset'])/'DATASET_MANIFEST.json')==native.BANK)
        else:
            valid=(result.get('status')=='PASS_VANDA_CPU_ORIGINAL126_AND_C1_FIXED_MEMBERS_AND_OFFICIAL_READER'
                and result.get('original_manifest_sha256')==DATA and result.get('C1_manifest_sha256')==C1_MANIFEST
                and result.get('original_TRAIN_parents')==126 and result.get('C1_student_parents')==4
                and result.get('DEV_raw_fetched')==0 and Path(b['dataset'])==r/'datasets/original'
                and Path(b['raw_bindings'])==r/'ORIGINAL_RAW_BINDINGS.private.json'
                and Path(inputs['C1_dataset'])==r/'datasets/C1'
                and Path(inputs['C1_raw_bindings'])==r/'C1_RAW_BINDINGS.private.json'
                and sha(Path(b['dataset'])/'DATASET_MANIFEST.json')==DATA
                and sha(Path(inputs['C1_dataset'])/'DATASET_MANIFEST.json')==C1_MANIFEST)
        if not valid:fail('FIXED_R0_DATASET_AND_ACTUAL_RAW_BINDING_PATHS')
        actual[kind]=row['result_sha256']
    return actual


def wait_for_consumers(root,guard,*,maximum_s=300,sleep=time.sleep,clock=time.monotonic):
    """Bounded allocation time, before torch/CUDA, for the exact queued reader."""
    if not 0<maximum_s<=300:fail('BOUNDED_DATA_WAIT_AT_MOST_300_SECONDS')
    records=consumer_records(root,require_closed=False);started=clock()
    pending=[row for row in records.values() if row.get('binding_mode')==DEFERRED]
    while True:
        guard()
        ready=True
        for row in pending:
            child=Path(row['root']);binding=sealed(child,'OUTPUT_BINDING.private.json')
            lease=read(binding['lease'])
            if lease.get('status') not in ('ACTIVE','CLOSED'):fail('CPU_READER_INVALID_LEASE_STATE')
            if (child/'RESULT.private.json').exists():
                result=sealed(child,'RESULT.private.json')
                if result.get('status')!='PASS_VANDA_CPU_ORIGINAL126_AND_C1_FIXED_MEMBERS_AND_OFFICIAL_READER':
                    fail('CPU_READER_TERMINAL_FAILURE_NO_RETRY')
            ready=ready and lease.get('status')=='CLOSED'
        if ready:return verify_consumers(root)
        if clock()-started>=maximum_s:raise TimeoutError('A40_FIXED_CPU_READER_NOT_READY_NO_MODEL_LOADED')
        sleep(min(15,maximum_s-(clock()-started)))


def require_premodel(root,allocation):
    root=Path(root);directory=owner_guard(root,allocation);rel=directory.relative_to(root)
    validate_root(root,directory/'REQUEST.json')
    q=sealed(root,str(rel/QUALIFICATION))
    if (q.get('status')!='PASS_A40_NATIVE_SOURCE_PREMODEL_NO_CORE_TRANSPORT_USED'
            or q.get('job_id')!=allocation['job_id'] or q.get('request_sha256')!=allocation['request_sha256']
            or q.get('source_sha')!=allocation['source_sha'] or q.get('native_GPU')!='NVIDIA_A40'
            or q.get('visible_GPU_count')!=1 or q.get('models_loaded')!=0
            or q.get('queries')!=0 or q.get('physics')!=0 or q.get('closed_consumers')!=verify_consumers(root)):
        fail('SAME_JOB_REAL_NATIVE_HARDWARE_AND_CLOSED_INPUTS_BEFORE_MODEL')
    return directory


def require_training(root,allocation):
    root=Path(root);directory=require_premodel(root,allocation)
    v=sealed(root,COMPATIBILITY)
    if (v.get('status')!='PASS_ACTUAL_SAME_JOB_17_SAVED_TRAIN_DRAWS_NOT_TASK_SUCCESS'
            or v.get('job_id')!=allocation['job_id'] or v.get('request_sha256')!=allocation['request_sha256']
            or v.get('parent_checkpoint_manifest_sha256')!=native.PARENT
            or v.get('statistics_sha256')!=native.R0 or v.get('new_physics')!=0
            or [r.get('index') for r in v.get('records',[])]!=list(range(17))):
        fail('ACTUAL_SAME_JOB_COMPATIBILITY_BEFORE_ANY_OPTIMIZER_UPDATE')
    for i,row in enumerate(v['records']):
        if (row.get('seed')!=29400+i or row.get('finite_shapes_verified') is not True
                or sha(root/'A40_COMPATIBILITY'/('query-'+str(i).zfill(2)+'.npz'))!=row['arrays_sha256']):
            fail('EXACT_SEVENTEEN_REAL_QUERY_ARRAYS_AND_SEEDS')
    return directory


def authorize_phase(root,directory,allocation,guard,long_phase,*,sleep=time.sleep):
    root,directory=Path(root),Path(directory)
    if require_training(root,allocation)!=directory:fail('EXACT_CURRENT_PHASE_DIRECTORY')
    guard()
    r=read(directory/'REQUEST.json');ready=directory/'A40_PRETRAIN_READY.json';release=directory/'A40_PRETRAIN_RELEASE.json'
    stop=r['training_block']['stop_after_step']
    if long_phase=='AFTER_PROBES':
        if not release.is_file():fail('ORIGINAL_WHOLE_JOB_FIRST_UPDATE_RELEASE_REQUIRED')
        proof=sealed(root,str(release.relative_to(root)))
        probes=probe_receipt(root,r['training_block']['arm'])
        folder=root/'evidence'/(r['training_block']['arm']+'-1024')/'TRAIN_PROBES'
        sealed(root,str((folder/'RESULT.json').relative_to(root)))
        for row in probes['records']:
            path=folder/(row['id']+'.npz')
            sealed(root,str(path.relative_to(root)),parse=False)
            if sha(path)!=row['arrays_sha256']:fail('FIXED_PROBE_ARRAY_CHANGED_BEFORE_UPDATE_1025')
        native=sealed(root,str((directory/'A40_FIRST_UPDATE_ADMISSION.json').relative_to(root)))
        if (proof.get('whole_block_stop_step')!=stop or native.get('release_sha256')!=sha(release)
                or native.get('request_sha256')!=allocation['request_sha256']):
            fail('NO_1024_ONLY_RESERVATION_PROMOTED_TO_4096')
    elif long_phase in ('BEFORE_PROBES','SEGMENT_REENTRY','B_RECOVERY_REENTRY'):
        if long_phase=='B_RECOVERY_REENTRY':
            from .b_recovery_intake import verify_copy
            verify_copy(root)
        if long_phase=='SEGMENT_REENTRY':
            from .segmented_training import enabled,limits
            if not enabled(r) or limits(r)!=(2048,4096):fail('EXPLICIT_SEGMENT_FULL_STATE2048_REENTRY')
            probes=probe_receipt(root,r['training_block']['arm'])
            folder=root/'evidence'/(r['training_block']['arm']+'-1024')/'TRAIN_PROBES'
            sealed(root,str((folder/'RESULT.json').relative_to(root)))
            for row in probes['records']:
                path=folder/(row['id']+'.npz')
                sealed(root,str(path.relative_to(root)),parse=False)
                if sha(path)!=row['arrays_sha256']:fail('IMPORTED_FIXED_PROBE_ARRAY_CHANGED_BEFORE_REENTRY')
        if ready.exists():raise FileExistsError('A40_PRETRAIN_READY_PRESERVED_NO_RESTART')
        now=time.time();write(ready,dict(job_id=allocation['job_id'],request_sha256=allocation['request_sha256'],
            submission_nonce=r['submission_nonce'],observed_epoch=now,expected_start_step=r['expected_native_start_step'],
            no_update_yet=True));ready.chmod(0o400)
        while not release.exists():
            guard()
            if time.time()-now>=180:raise TimeoutError('A40_FRESH_FIRST_UPDATE_RELEASE_NOT_RECEIVED')
            sleep(.2)
        proof=sealed(root,str(release.relative_to(root)))
        if (not now<=proof['observed_epoch']<=time.time() or time.time()-now>=180
                or proof.get('ready_sha256')!=sha(ready)):
            fail('FIRST_UPDATE_RELEASE_AFTER_ACTUAL_READY_WITHIN_ORIGINAL_WINDOW')
        material=sealed(root,proof['material_relative'])
        if sha(root/proof['material_relative'])!=proof['material_sha256']:fail('FRESH_NATIVE_BUDGET_MATERIAL_BYTES')
        from .a40_future import recompute
        # Native ledger hashes identify their original serialization, not a new JSON rendering.
        import base64
        values={k:base64.b64decode(material[k+'_base64'],validate=True)
            for k in ('ledger','DEC1_ledger','snapshot','progress')}
        expected=proof['recomputed_budget']
        if not 0<=time.time()-expected['observed_epoch']<=120:fail('FRESH_REAL_FIRST_UPDATE_RECOMPUTATION')
        sealed(root,proof['plan_relative'])
        m=read(root/'RUN_MANIFEST.json')
        if m['execution']['sealed_inputs'].get(proof['plan_relative'])!=sha(root/proof['plan_relative']):
            fail('ORIGINAL_PRESEALED_FULL_WORK_PLAN')
        result=recompute(request_bytes=(directory/'REQUEST.json').read_bytes(),
            ledger_bytes=values['ledger'],DEC1_ledger_bytes=values['DEC1_ledger'],snapshot_bytes=values['snapshot'],
            progress_bytes=values['progress'],plan_bytes=(root/proof['plan_relative']).read_bytes(),
            amendment_bytes=(root/'plans/VAP1_ADOPTION_AND_DELTA_BUDGET.json').read_bytes(),
            recovery_bytes=((root/'plans/DEC1_LOST_SAVE_RECOVERY_AUTHORIZATION.json').read_bytes()
                if r.get('recovery_authorization_release') is not None else None),
            segment_bytes=((root/'plans/DEC1_SEGMENTED_RECOVERY_AUTHORIZATION.json').read_bytes()
                if r.get('segmented_authorization_release') is not None else None),
            storage_bytes=((root/'plans/VAP1_STORAGE_ONLY_AUTHORIZATION.json').read_bytes()
                if r.get('storage_authorization_release') is not None else None),
            heartbeat_bytes=((root/'plans/A2_HEARTBEAT_RECOVERY_AUTHORIZATION.json').read_bytes()
                if r.get('heartbeat_recovery_release') is not None else None),
            zero_update_bytes=((root/'plans/ZERO_UPDATE_RESUME_AUTHORIZATION.json').read_bytes()
                if r.get('zero_update_resume_release') is not None else None),
            b_recovery_bytes=((root/'plans/B1_HEARTBEAT_RECOVERY_AUTHORIZATION.json').read_bytes()
                if r.get('B_heartbeat_recovery_release') is not None else None),
            retry_bytes=((root/'plans/B2_SYSTEM_STARTUP_RETRY_PBS_AUTHORIZATION.json').read_bytes()
                if r.get('B_system_startup_retry_release') is not None else None),
            vanda2_bytes=((root/'plans/B2970_RECOVERY2_FIVE_CAP_AUTHORIZATION.json').read_bytes()
                if r.get('B_Vanda_recovery2_release') is not None else None),
            now_epoch=time.time(),job_id=allocation['job_id'])
        def stable(value):
            v=deepcopy(value);v.pop('observed_epoch',None);v['forecast'].pop('projected_finish_epoch',None);return v
        if stable(expected)!=stable(result):fail('ACTUAL_RECOMPUTED_NATIVE_SOURCE_BUDGET_NOT_A_LABEL')
        from scripts.pi0294.pv1.pbs_owner import publish_release_record
        publish_release_record(directory/'A40_FIRST_UPDATE_ADMISSION.json',dict(
            status='PASS_NATIVE_A40_FIRST_UPDATE_FULL_BLOCK_RECOMPUTATION',
            request_sha256=allocation['request_sha256'],release_sha256=sha(release),recomputed_budget=result))
    else:fail('EXPLICIT_ORIGINAL_LONG_TRAIN_PHASE_ONLY')
    if (proof.get('job_id')!=allocation['job_id'] or proof.get('request_sha256')!=allocation['request_sha256']
            or proof.get('submission_nonce')!=r['submission_nonce'] or proof.get('whole_block_stop_step')!=stop):
        fail('SAME_ALLOCATION_WHOLE_BLOCK_RELEASE')
    return dict(status='PASS_A40_SAME_JOB_WHOLE_BLOCK_AUTHORIZATION',new_allocation_or_budget=False)


def first32(root,directory,allocation,times,*,native_loss,peak_bytes):
    """Observe real A updates only; a cost miss preserves step32 for review."""
    import math
    root,directory=Path(root),Path(directory);require_training(root,allocation)
    if len(times)!=32 or not all(math.isfinite(t) and t>0 for t in times) or not math.isfinite(native_loss):
        fail('ACTUAL_FINITE_FIRST32_NATIVE_UPDATE_TIMINGS_AND_LOSS')
    release=sealed(root,str((directory/'A40_PRETRAIN_RELEASE.json').relative_to(root)))
    plan=sealed(root,release['plan_relative']);ceiling=plan['seconds_per_update_upper']
    manifest=sealed(root,'RUN_MANIFEST.json')
    initial=sealed(root,str((directory/'A40_FIRST_UPDATE_ADMISSION.json').relative_to(root)))
    if (manifest['execution']['sealed_inputs'].get(release['plan_relative'])!=sha(root/release['plan_relative'])
            or initial.get('release_sha256')!=sha(directory/'A40_PRETRAIN_RELEASE.json')):
        fail('FIRST32_ORIGINAL_NATIVE_ADMITTED_PLAN_AND_RELEASE')
    observed=max(times)*1.1
    passes=observed<=ceiling
    from scripts.pi0294.pv1.pbs_owner import publish_release_record
    result=dict(status='PASS_FIRST32_INSIDE_PRESEALED_A40_COST_CEILING' if passes else 'STOP_FIRST32_RUNTIME_COST_REFORECAST_REQUIRED',
        job_id=allocation['job_id'],request_sha256=allocation['request_sha256'],actual_A_CONTROL_updates=32,
        additional_updates=0,synchronized_step_s=times,observed_maximum_with_1p1_headroom_s=observed,
        presealed_unmeasured_ceiling_s=ceiling,native_loss=native_loss,peak_GPU_allocated_bytes=peak_bytes,
        gradients_and_original_loss_qualified_by_mature_Trainer=True,continue_same_state=passes,
        failed_cost_gate_does_not_discard_or_refund_updates=True)
    publish_release_record(directory/'A40_FIRST32_ACTUAL.json',result)
    return result
