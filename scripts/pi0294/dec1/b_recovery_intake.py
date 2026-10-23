"""Funded immutable B1024 intake from the exact closed failed B1.

Reuses mature full-state/probe validation and storage linking. No new probe,
training runner, optimizer implementation or source-state mutation.
"""
import json
import os
from pathlib import Path
import shutil
from scripts.pi0294.native_dataset import sha
from . import b_heartbeat_recovery as b
from .segment_intake import owned,read,full_state,probes
from .alternate_accounting import digest

BINDING='plans/B_RECOVERY1024_SOURCE_BINDING.private.json'
PROOF='plans/B_RECOVERY1024_FULL_STATE_IMPORT.private.json'
SOURCE_KEYS={'root','run_manifest_sha256','lease_sha256','control_terminal_sha256','terminal_sha256',
    'job_id','request_id','request_sha256','source_sha','checkpoint1024_sha256','probe_result_sha256'}


def fail(reason):raise PermissionError('DEC1_B1024_INTAKE_'+reason)


def valid_binding(v):
    if (type(v) is not dict or set(v)!=SOURCE_KEYS or not Path(v.get('root','')).is_absolute()
            or v.get('job_id')!=b.FAILED_JOB or v.get('request_id')!=b.FAILED_REQUEST
            or v.get('request_sha256')!=b.REQUEST_SHA or v.get('source_sha')!=b.FAILED_SOURCE
            or v.get('checkpoint1024_sha256')!=b.SEED_SHA or v.get('probe_result_sha256')!=b.PROBE_SHA
            or v.get('terminal_sha256')!=b.CLOSURE_SHA or v.get('control_terminal_sha256')!=b.CONTROL_CLOSURE_SHA
            or v.get('lease_sha256')!='28a7252b7b05a9d7cdf7fdd7c1ff168f3ee303d5e97392e361b9562c2564e524'
            or v.get('run_manifest_sha256')!='8c549bbca0fed64e05112444e9e460ec66fa4bc487a0983cd8d1ddf20772669d'):
        fail('EXACT_FAILED_B1_SOURCE_AND_AUDITED1024_NOT_A_OR1829')
    return v


def audit_closed_source(binding,*,recipe,dataset):
    valid_binding(binding);root=Path(binding['root'])
    if root.is_symlink() or root.stat().st_uid!=os.getuid() or digest(recipe)!=b.RECIPE:
        fail('OWN_ORIGINAL_B_RECIPE_AND_REGISTERED_SOURCE')
    output=read(root/'OUTPUT_BINDING.private.json');lp=owned(output['lease'])
    paths=dict(run_manifest_sha256=root/'RUN_MANIFEST.json',terminal_sha256=root/'TERMINAL_RECONCILIATION.json',lease_sha256=lp)
    before={k:sha(owned(p,root if k!='lease_sha256' else None)) for k,p in paths.items()}
    if any(binding[k]!=v for k,v in before.items()):fail('SOURCE_CLOSED_METADATA_BYTES')
    m=read(paths['run_manifest_sha256']);c=read(paths['terminal_sha256']);lease=read(lp)
    if (output.get('run_root')!=str(root) or output.get('uid')!=os.getuid()
            or output.get('run_manifest_sha256')!=binding['run_manifest_sha256']
            or m.get('code',{}).get('vla_commit')!=b.FAILED_SOURCE or lease.get('status')!='CLOSED'
            or lease.get('run_uuid')!=root.name or lease.get('terminal_receipt_sha256')!=b.CLOSURE_SHA
            or c.get('status')!='CLOSED_B1_HEARTBEAT_FAILURE_1829_B1024_PRESERVED'
            or c.get('job_id')!=b.FAILED_JOB or c.get('latest_complete_step')!=1024
            or c.get('committed_updates_without_new_complete_state')!=805
            or c.get('prior_complete_state_preserved') is not True or c.get('retired_or_deleted_files')!=0
            or c.get('settlement',{}).get('updates')!=7720 or c['settlement'].get('unknown_effects')!=[]):
        fail('EXACT_CLOSED_FAILURE_WITH_FULL_B1024_NO_SUCCESS_RELABEL')
    e=c.get('effects',{})
    if any(e.get(k)!=v for k,v in dict(job_id=b.FAILED_JOB,request_id=b.FAILED_REQUEST,source_sha=b.FAILED_SOURCE,
            arm='B_COVERAGE',start_native_step=0,last_native_step=1829,optimizer_updates_committed=1829,
            recipe_sha256=b.RECIPE,query_attempts=29,pending_updates=[],unknown_query_intents=[],effect_journal_complete=True).items()):
        fail('UNCHANGED_PAID1829_EFFECTS')
    complete=full_state(root,'B_COVERAGE',1024,recipe,dataset,b.SEED_SHA)
    if complete.get('origin_job_id')!=b.FAILED_JOB or complete.get('origin_source_sha')!=b.FAILED_SOURCE:
        fail('ORIGINAL_B1024_ORIGIN')
    result,files=probes(root,'B_COVERAGE')
    probe=root/'evidence/B_COVERAGE-1024/TRAIN_PROBES/RESULT.json'
    if sha(probe)!=b.PROBE_SHA or before!={k:sha(p) for k,p in paths.items()}:fail('ORIGINAL_PROBES_AND_SOURCE_UNCHANGED')
    return dict(schema='DEC1_B_FAILED1829_FULL1024_IMPORT_PROOF_V1',source=dict(binding),
        authority_release=b.RELEASE,grant_sha256=b.GRANT_SHA,training_lineage_key=b.LINEAGE,
        global_step=1024,actual_consumed_cursor=32768,recipe_sha256=b.RECIPE,
        checkpoint_members=complete['files'],probe_files=files,new_updates=0,paid_failed_step_preserved=1829,
        old_failed_costs_preserved=True,new_probes=0)


def cold_binding(root,request,input_):
    if not b.enabled(request) or b.lineage(request)!=b.LINEAGE or request.get('expected_native_start_step')!=1024:
        fail('EXPLICIT_NEW_B1024_REQUEST')
    binding=input_(BINDING);proof=input_(PROOF);valid_binding(binding)
    p=Path(root)/'training/B_COVERAGE/checkpoint-1024'
    if (p.exists() or proof.get('source')!=binding or proof.get('global_step')!=1024
            or proof.get('actual_consumed_cursor')!=32768 or proof.get('training_lineage_key')!=b.LINEAGE
            or proof.get('authority_release')!=b.RELEASE or proof.get('grant_sha256')!=b.GRANT_SHA
            or proof.get('recipe_sha256')!=b.RECIPE or proof.get('new_updates')!=0 or proof.get('new_probes')!=0):
        fail('ONLY_COLD_PENDING_EXACT_IMPORTED_B1024')
    return binding


def verify_copy(root):
    from .a40_training_context import sealed
    root=Path(root);binding=sealed(root,BINDING);proof=sealed(root,PROOF);valid_binding(binding)
    m=read(root/'RUN_MANIFEST.json')
    if (proof.get('source')!=binding or proof.get('authority_release')!=b.RELEASE
            or proof.get('grant_sha256')!=b.GRANT_SHA or proof.get('training_lineage_key')!=b.LINEAGE
            or proof.get('global_step')!=1024 or proof.get('actual_consumed_cursor')!=32768
            or proof.get('recipe_sha256')!=b.RECIPE or proof.get('new_updates')!=0 or proof.get('new_probes')!=0
            or any(m.get('execution',{}).get('sealed_inputs',{}).get(n)!=sha(root/n) for n in (BINDING,PROOF))):
        fail('SEALED_NEW_MANIFEST_AND_EXACT_FULL_IMPORT_PROOF')
    cp=root/'training/B_COVERAGE/checkpoint-1024/CHECKPOINT_COMPLETE.json'
    if sha(owned(cp,root))!=b.SEED_SHA or read(cp).get('files')!=proof.get('checkpoint_members'):
        fail('EXACT_COPIED_ORIGINAL1024_COMPLETE_MANIFEST')
    result,files=probes(root,'B_COVERAGE')
    if files!=proof.get('probe_files') or result!=sealed(root,'plans/COMPLETED_1024_TRAIN_PROBES.json'):
        fail('EXACT_ORIGINAL_TWELVE_PROBE_FILES')
    return dict(status='PASS_COPIED_ORIGINAL_B1024_AND_PROBES_NOT_TRAINING_ADMISSION',global_step=1024,
        actual_consumed_cursor=32768,checkpoint_complete_sha256=b.SEED_SHA,new_updates=0,new_probes=0)


def stage(root,*,guard,allocation):
    from .a40_training_context import current_directory,sealed
    from .segmented_training import limits
    from .checkpoint_storage import POLICY,_parent_files
    from scripts.pi0294.pv1.neural import train_dataset
    from scripts.pi0294.pv1.pbs_owner import publish_release_record as once
    root=Path(root);guard();directory=current_directory(root,allocation);request=read(directory/'REQUEST.json')
    if (not os.environ.get('PBS_JOBID') or not b.enabled(request) or limits(request)!=(1024,2048)
            or request['additional_output_upper_bytes']!=70000000000):fail('EXACT_FUNDED_NATIVE_B1024_TO2048')
    b.admitted(root,read(root/'RUN_MANIFEST.json'),request)
    binding=sealed(root,BINDING);dataset=train_dataset(root);dataset.initialize()
    recipe=sealed(root,'plans/PRIMARY_LEARNING_RECIPES.json')['recipes']['B_COVERAGE']
    proof=audit_closed_source(binding,recipe=recipe,dataset=dataset)
    if proof!=sealed(root,PROOF):fail('ACTUAL_SOURCE_EQUALS_PRESEALED_FULL_PROOF')
    parent=Path(sealed(root,'plans/PV1_TRAIN_INPUTS.private.json')['CORR_parent_checkpoint'])
    frozen=sealed(root,POLICY)['members'];parent_files=_parent_files(root,parent)
    if len(frozen)!=1:fail('ONE_ORIGINAL_FROZEN_PARENT_SHARD')
    name,expected=next(iter(frozen.items()));fp=owned(parent/name,parent)
    if parent_files.get(name)!=expected or sha(fp)!=expected or proof['checkpoint_members'].get(name)!=expected:
        fail('NEVER_LINK_TRAINED_SHARD_AS_FROZEN_PARENT')
    cp=root/'training/B_COVERAGE/checkpoint-1024'
    if cp.exists():fail('NO_PARTIAL_IMPORT_OVERWRITE_OR_REPLAY')
    before=fp.stat();once(root/'control/B_RECOVERY1024_IMPORT_INTENT.json',dict(job_id=os.environ['PBS_JOBID'],
        source_proof_sha256=sha(root/PROOF),new_updates=0));cp.mkdir(parents=True,mode=0o700)
    def copy(src,dst):
        guard();dst.parent.mkdir(parents=True,exist_ok=True)
        with owned(src).open('rb') as inp,dst.open('xb') as out:
            shutil.copyfileobj(inp,out,1024*1024);out.flush();os.fsync(out.fileno())
        dst.chmod(0o400)
    for n in proof['checkpoint_members']:
        guard();dst=cp/n
        if n==name:
            if before.st_dev!=cp.stat().st_dev:fail('SAME_FILESYSTEM_FROZEN_LINK')
            os.link(fp,dst)
        else:copy(Path(binding['root'])/'training/B_COVERAGE/checkpoint-1024'/n,dst)
    copy(Path(binding['root'])/'training/B_COVERAGE/checkpoint-1024/CHECKPOINT_COMPLETE.json',cp/'CHECKPOINT_COMPLETE.json')
    for n in proof['probe_files']:copy(Path(binding['root'])/n,root/n)
    after=fp.stat()
    if (after.st_ino,after.st_mode,after.st_mtime_ns)!=(before.st_ino,before.st_mode,before.st_mtime_ns):fail('PARENT_NOT_MUTATED')
    # Full original model/optimizer/scheduler/RNG/sampler/statistics audit on
    # the actual copied members before any model is loaded.
    full_state(root,'B_COVERAGE',1024,recipe,dataset,b.SEED_SHA)
    result=verify_copy(root);guard();once(root/'control/B_RECOVERY1024_IMPORT_RESULT.json',dict(result,
        job_id=os.environ['PBS_JOBID'],frozen_link_is_not_independent_backup=True));return result
