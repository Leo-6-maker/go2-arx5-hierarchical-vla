"""Closed2048 full-state intake for a later cold, funded continuation.

The1024 state and original twelve probe arrays travel with2048. No new draw,
optimizer step, source mutation, model load or lease is created by an audit.
"""
import json
import os
from pathlib import Path
import re
import shutil

from scripts.pi0294.native_dataset import sha
from scripts.pi0294.pv1.training import checked_resume
from .alternate_accounting import digest
from .authority import R0
from .lost_save_recovery import LINEAGE,RECIPE
from .segmented_training import RELEASE,limits
from .recovery32_intake import owned
from .long_training import probe_receipt
from . import b_saved_state_reuse as saved

BINDING='plans/SEGMENT2048_SOURCE_BINDING.private.json'
PROOF='plans/SEGMENT2048_FULL_STATE_IMPORT.private.json'


def fail(reason):raise PermissionError('DEC1_SEGMENT2048_INTAKE_'+reason)


def read(path):return json.loads(owned(path).read_bytes())


def b_seed_identity():
    from . import b_heartbeat_recovery as b
    return dict(authority_release=b.RELEASE,grant_sha256=b.GRANT_SHA,job_id=b.FAILED_JOB,
        source_sha=b.FAILED_SOURCE,checkpoint1024_sha256=b.SEED_SHA,probe_result_sha256=b.PROBE_SHA)


def recovery_seed(value,arm):
    if 'B_heartbeat_recovery_seed' not in value:return False
    if arm!='B_COVERAGE' or value['B_heartbeat_recovery_seed']!=b_seed_identity():
        fail('EXACT_ORIGINAL_B1024_SEED_IDENTITY')
    return True


def valid_binding(value,arm):
    required={'root','arm','job_id','request_id','request_sha256','source_sha','run_manifest_sha256',
        'terminal_sha256','lease_sha256','checkpoint1024_sha256','checkpoint2048_sha256','probe_result_sha256'}
    new_b=recovery_seed(value,arm)
    if new_b:required.add('B_heartbeat_recovery_seed')
    if saved.binding(value):
        if not new_b:fail('APPROVED_FAILED_SAVE_REQUIRES_ORIGINAL_B_RECOVERY_SEED')
        required.add(saved.KEY)
    if (set(value)!=required or value['arm']!=arm or arm not in ('A_CONTROL','B_COVERAGE')
            or re.fullmatch(r'[0-9]+\.stdct-mgmt-02',value['job_id']) is None
            or not Path(value['root']).is_absolute()
            or any(re.fullmatch('[0-9a-f]{'+str(40 if k=='source_sha' else 64)+'}',value[k]) is None
                for k in required if k.endswith('sha256') or k=='source_sha')):
        fail('EXACT_COLD_OWNER_SOURCE_IDENTITY_AND_HASH_SET')
    if new_b:
        from . import b_heartbeat_recovery as b
        if (value['job_id']==b.FAILED_JOB or value['checkpoint1024_sha256']!=b.SEED_SHA
                or value['probe_result_sha256']!=b.PROBE_SHA):fail('NEW_B2048_JOB_PRESERVES_OLD1024_AND_PROBES')
    return value


def full_state(root,arm,step,recipe,dataset,expected_sha):
    root=Path(root);p=root/'training'/arm/('checkpoint-'+str(step))
    complete_path=owned(p/'CHECKPOINT_COMPLETE.json',root)
    if sha(complete_path)!=expected_sha:fail('COMPLETE_MANIFEST_CHANGED')
    complete=read(complete_path)
    for name in complete.get('files',{}):
        n=Path(name)
        if n.is_absolute() or '..' in n.parts or n.as_posix()!=name or name in ('','.','CHECKPOINT_COMPLETE.json'):
            fail('CANONICAL_RELATIVE_CHECKPOINT_MEMBER')
        owned(p/n,p)
    if checked_resume(root,arm,p,recipe,dataset)!=step:fail('MATURE_FULL_OPTIMIZER_RNG_SAMPLER_RESUME')
    if sha(complete_path)!=expected_sha:fail('MANIFEST_CHANGED_DURING_READ')
    return complete


def probes(root,arm):
    result=probe_receipt(root,arm)
    folder=Path(root)/'evidence'/(arm+'-1024')/'TRAIN_PROBES'
    files={folder.relative_to(root).as_posix()+'/RESULT.json':sha(owned(folder/'RESULT.json',root))}
    for row in result['records']:
        if not isinstance(row['id'],str) or Path(row['id']).name!=row['id'] or row['id'] in ('.','..'):
            fail('CANONICAL_ORIGINAL_PROBE_IDENTIFIER')
        path=owned(folder/(row['id']+'.npz'),root)
        if sha(path)!=row['arrays_sha256']:fail('ORIGINAL_PROBE_ARRAY_CHANGED')
        files[path.relative_to(root).as_posix()]=row['arrays_sha256']
    return result,files


def audit_closed_source(binding,*,recipe,dataset,reuse_grant_bytes=None):
    arm=binding['arm'];valid_binding(binding,arm);root=Path(binding['root'])
    failed_saved=saved.binding(binding)
    if failed_saved:saved.grant(reuse_grant_bytes)
    if root.is_symlink() or root.stat().st_uid!=os.getuid():fail('OWN_CLOSED_REGISTERED_SOURCE_ROOT')
    b=read(root/'OUTPUT_BINDING.private.json');lease_path=owned(b['lease'])
    paths={'run_manifest_sha256':root/'RUN_MANIFEST.json','terminal_sha256':root/'TERMINAL_RECONCILIATION.json',
        'lease_sha256':lease_path}
    before={k:sha(owned(p,root if k!='lease_sha256' else None)) for k,p in paths.items()}
    if any(binding[k]!=v for k,v in before.items()):fail('ACTUAL_IMMUTABLE_MANIFEST_CLOSURE_AND_LEASE')
    m=read(paths['run_manifest_sha256']);terminal=read(paths['terminal_sha256']);lease=read(lease_path)
    if failed_saved:saved.terminal(terminal)
    if (b.get('run_root')!=str(root) or b.get('uid')!=os.getuid() or b.get('run_manifest_sha256')!=binding['run_manifest_sha256']
            or m['code'].get('vla_commit')!=binding['source_sha']
            or m.get('authority',{}).get('segmented_authorization_release')!=RELEASE
            or lease.get('status')!='CLOSED' or lease.get('run_uuid')!=root.name
            or lease.get('terminal_receipt_sha256')!=binding['terminal_sha256']
            or not failed_saved and terminal.get('status')!='CLOSED_SEGMENT_FULL_STATE_PRESERVED'
            or terminal.get('job_id')!=binding['job_id'] or terminal.get('recoverable_full_state') is not True
            or not failed_saved and terminal.get('global_step')!=2048
            or terminal.get('settlement',{}).get('unknown_effects')!=[]):
        fail('CLOSED2048_KNOWN_EFFECTS_NOT_A_RUNNING_OR_FAILED_PARTIAL_SOURCE')
    new_b=recovery_seed(binding,arm)
    if new_b:
        from . import b_heartbeat_recovery as b
        if any(m.get('authority',{}).get(k)!=v for k,v in zip(b.SCOPE_KEYS,(b.RELEASE,b.GRANT_SHA))):
            fail('EXACT_NEW_B_AUTHORITY_ON_COMPLETED2048_MANIFEST')
    e=terminal['effects']
    if (e.get('job_id')!=binding['job_id'] or e.get('request_id')!=binding['request_id']
            or e.get('request_sha256')!=binding['request_sha256'] or e.get('source_sha')!=binding['source_sha']
            or e.get('arm')!=arm or e.get('last_native_step')!=2048 or e.get('recipe_sha256')!=digest(recipe)
            or e.get('pending_updates')!=[] or e.get('unknown_query_intents')!=[]
            or e.get('query_attempts')!=(17 if new_b else 29) or e.get('effect_journal_complete') is not True
            or e.get('start_native_step') not in ((1024,) if new_b else (0,32))
            or e.get('optimizer_updates_committed')!=2048-e['start_native_step']):
        fail('SAME_ARM_ACTUAL2048_SETTLED_EFFECT_JOURNAL')
    if recipe.get('arm')!=arm:fail('SAME_ARM_RECIPE_NOT_OTHER_SHARED_POLICY')
    if arm=='A_CONTROL' and digest(recipe)!=RECIPE:fail('ORIGINAL_A_RECIPE_PRESERVED')
    complete={}
    for step in (1024,2048):
        c=full_state(root,arm,step,recipe,dataset,binding[f'checkpoint{step}_sha256'])
        origin=binding['B_heartbeat_recovery_seed'] if new_b and step==1024 else dict(job_id=binding['job_id'],source_sha=binding['source_sha'])
        if c.get('origin_job_id')!=origin['job_id'] or c.get('origin_source_sha')!=origin['source_sha']:
            fail('BOTH_SCIENTIFIC1024_AND_ENGINEERING2048_FROM_SAME_SETTLED_JOB')
        complete[str(step)]=c['files']
    result,files=probes(root,arm)
    if sha(root/'evidence'/(arm+'-1024')/'TRAIN_PROBES/RESULT.json')!=binding['probe_result_sha256']:
        fail('EXACT_TWELVE_COMPLETED_ORIGINAL_TRAIN_PROBES')
    if before!={k:sha(p) for k,p in paths.items()}:fail('SOURCE_METADATA_CHANGED_DURING_AUDIT')
    return dict(schema='DEC1_SEGMENT2048_CLOSED_FULL_STATE_IMPORT_PROOF_V1',
        status=('PASS_CLOSED_FAILED_SAVED2048_FULL_STATES_NOT_TRAINING_ADMISSION' if failed_saved else 'PASS_CLOSED2048_FULL_STATES_AND_ORIGINAL1024_PROBES_NOT_TRAINING_ADMISSION'),
        authority_release=RELEASE,arm=arm,training_lineage_key=(b.LINEAGE if new_b else (LINEAGE if arm=='A_CONTROL' else arm)),
        source=dict(binding),global_step=2048,actual_consumed_cursor=65536,recipe_sha256=digest(recipe),
        statistics_sha256=R0,checkpoint_members=complete,probe_files=files,new_GPU_PBS_models_queries_updates_physics=0)


def verify_copy(root,*,recipe,dataset):
    from .a40_training_context import sealed
    root=Path(root);binding=sealed(root,BINDING);proof=sealed(root,PROOF);arm=binding['arm']
    valid_binding(binding,arm)
    failed_saved=saved.binding(binding)
    if failed_saved:saved.grant(owned(root/saved.PLAN,root).read_bytes())
    new_b=recovery_seed(binding,arm)
    if new_b:
        from . import b_heartbeat_recovery as b
    expected=dict(schema='DEC1_SEGMENT2048_CLOSED_FULL_STATE_IMPORT_PROOF_V1',
        status=('PASS_CLOSED_FAILED_SAVED2048_FULL_STATES_NOT_TRAINING_ADMISSION' if failed_saved else 'PASS_CLOSED2048_FULL_STATES_AND_ORIGINAL1024_PROBES_NOT_TRAINING_ADMISSION'),
        authority_release=RELEASE,arm=arm,training_lineage_key=(b.LINEAGE if new_b else (LINEAGE if arm=='A_CONTROL' else arm)),
        source=binding,global_step=2048,actual_consumed_cursor=65536,recipe_sha256=digest(recipe),statistics_sha256=R0,
        new_GPU_PBS_models_queries_updates_physics=0)
    if set(proof)!=set(expected)|{'checkpoint_members','probe_files'} or any(proof.get(k)!=v for k,v in expected.items()):
        fail('EXACT_SEALED_SOURCE2048_PROOF_AND_SAME_ARM')
    m=read(root/'RUN_MANIFEST.json')
    if (m.get('authority',{}).get('segmented_authorization_release')!=RELEASE
            or any(m['execution']['sealed_inputs'].get(n)!=sha(root/n) for n in (BINDING,PROOF))):
        fail('NATIVE_NEW_MANIFEST_MUST_SEAL_IMPORT_IDENTITIES')
    for step in (1024,2048):
        complete=full_state(root,arm,step,recipe,dataset,binding[f'checkpoint{step}_sha256'])
        if complete['files']!=proof['checkpoint_members'].get(str(step)):
            fail('COPIED_FULL_STATE_MEMBER_SET_EQUALS_CLOSED_SOURCE')
    result,files=probes(root,arm)
    if files!=proof['probe_files'] or sealed(root,'plans/COMPLETED_1024_TRAIN_PROBES.json')!=result:
        fail('EXACT_ORIGINAL_PROBE_RESULT_AND_ARRAYS_NOT_NEW_INFERENCE')
    return dict(status='PASS_OWN2048_FULL_RESUME_AND_PRESERVED1024_PROBES_NO_NEW_DRAW',
        arm=arm,global_step=2048,actual_consumed_cursor=65536,checkpoint_complete_sha256=binding['checkpoint2048_sha256'],
        new_GPU_PBS_models_queries_updates_physics=0)


def stage(root,*,recipe,dataset,guard,allocation):
    from .a40_training_context import current_directory,sealed
    from .checkpoint_storage import POLICY,_parent_files
    from scripts.pi0294.pv1.pbs_owner import publish_release_record as once
    root=Path(root);guard();directory=current_directory(root,allocation);request=read(directory/'REQUEST.json')
    if not os.environ.get('PBS_JOBID') or limits(request)!=(2048,4096) or request['additional_output_upper_bytes']!=70000000000:
        fail('EXACT_FUNDED_NATIVE2048_TO4096_COPY_SCOPE')
    binding=sealed(root,BINDING);arm=request['training_block']['arm']
    if binding['arm']!=arm:fail('NO_OTHER_ARM_IMPORTED')
    reuse_grant=None
    if saved.binding(binding):
        if not saved.request(request):fail('FAILED_SAVE_REQUIRES_EXPLICIT_REQUEST_SCOPE')
        reuse_grant=owned(root/saved.PLAN,root).read_bytes();saved.grant(reuse_grant)
        manifest=read(root/'RUN_MANIFEST.json')
        if (any(manifest.get('authority',{}).get(k)!=v for k,v in saved.scope().items())
                or manifest['execution']['sealed_inputs'].get(saved.PLAN)!=sha(root/saved.PLAN)):
            fail('FAILED_SAVE_GRANT_SEALED_IN_CURRENT_MANIFEST')
    proof=audit_closed_source(binding,recipe=recipe,dataset=dataset,reuse_grant_bytes=reuse_grant)
    if proof!=sealed(root,PROOF):fail('ACTUAL_SOURCE_EQUALS_PRESEALED_CLOSED_PROOF')
    source=Path(binding['root']);policy=sealed(root,POLICY)
    parent=Path(sealed(root,'plans/PV1_TRAIN_INPUTS.private.json')['CORR_parent_checkpoint'])
    frozen=policy['members'];parent_files=_parent_files(root,parent)
    if len(frozen)!=1:fail('ONLY_ONE_KNOWN_FROZEN_PARENT_SHARD_LINK')
    name,expected=next(iter(frozen.items()));frozen_path=owned(parent/name,parent)
    if parent_files.get(name)!=expected or sha(frozen_path)!=expected:fail('EXACT_PARENT_FROZEN_BYTES')
    for members in proof['checkpoint_members'].values():
        if members.get(name)!=expected:fail('NEVER_LINK_TRAINED_SHARD_AS_PARENT')
    if any((root/'training'/arm/f'checkpoint-{step}').exists() for step in (1024,2048)):
        fail('NO_PARTIAL_IMPORT_OVERWRITE_OR_REPLAY')
    before=frozen_path.stat()
    once(root/'control/SEGMENT2048_IMPORT_INTENT.json',dict(job_id=os.environ['PBS_JOBID'],source_proof_sha256=sha(root/PROOF),new_updates=0))
    for step in (1024,2048):
        cp=root/'training'/arm/f'checkpoint-{step}';cp.mkdir(parents=True,mode=0o700)
        for n in proof['checkpoint_members'][str(step)]:
            guard();dst=cp/n;dst.parent.mkdir(parents=True,exist_ok=True)
            if n==name:
                if before.st_dev!=cp.stat().st_dev:fail('SAME_FILESYSTEM_FROZEN_LINK')
                os.link(frozen_path,dst)
            else:
                with owned(source/'training'/arm/f'checkpoint-{step}'/n).open('rb') as src,dst.open('xb') as out:
                    shutil.copyfileobj(src,out,1024*1024);out.flush();os.fsync(out.fileno())
                dst.chmod(0o400)
        src=source/'training'/arm/f'checkpoint-{step}'/'CHECKPOINT_COMPLETE.json'
        with owned(src).open('rb') as inp,(cp/'CHECKPOINT_COMPLETE.json').open('xb') as out:
            shutil.copyfileobj(inp,out);out.flush();os.fsync(out.fileno())
        (cp/'CHECKPOINT_COMPLETE.json').chmod(0o400)
    for n in proof['probe_files']:
        guard();dst=root/n;dst.parent.mkdir(parents=True,exist_ok=True)
        with owned(source/n,source).open('rb') as src,dst.open('xb') as out:
            shutil.copyfileobj(src,out);out.flush();os.fsync(out.fileno())
        dst.chmod(0o400)
    after=frozen_path.stat()
    if (after.st_ino,after.st_mode,after.st_mtime_ns)!=(before.st_ino,before.st_mode,before.st_mtime_ns):
        fail('PROTECTED_PARENT_NOT_MUTATED')
    result=verify_copy(root,recipe=recipe,dataset=dataset);guard()
    once(root/'control/SEGMENT2048_IMPORT_RESULT.json',dict(result,source_states_and_probe_arrays_preserved=True,
        frozen_parent_link_is_not_independent_backup=True,job_id=os.environ['PBS_JOBID']))
    return result
