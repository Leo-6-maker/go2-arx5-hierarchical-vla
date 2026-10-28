"""Read-only provenance for the one preserved recovery32 full state.

This does not copy, link, chmod, retire, launch, load tensors or admit training.
The registered cold owner must separately fund staging and seal the returned
proof in its new manifest. Existing allocation/source/lease gates still apply.
"""
import json
import os
from pathlib import Path

from scripts.pi0294.native_dataset import sha
from scripts.pi0294.pv1.training import checked_resume
from .alternate_accounting import digest
from .authority import R0
from .lost_save_recovery import LINEAGE, RELEASE, RECIPE

PROOF='plans/RECOVERY32_FULL_STATE_IMPORT.private.json'
EXPECTED=dict(
    job_id='1474352.stdct-mgmt-02',
    source_sha='78427dd1e655c01b298712606d5ba05d260b9bd4',
    request_id='a8a42617-da53-42ee-871e-6e8a24248b35',
    request_sha256='d46bd544d8b1d6711ce2dea278993534ada653e3cd6b4bea746a8ed950a97b8f',
    run_manifest_sha256='99ebfe3fd012746ead8889bfab65accfa3f331bf9ee3c59485df10278a88b89b',
    terminal_sha256='f5b4342813cc1e3e0c98496a407aecadc5549214e2e3c4065a7ed4ca660d40cb',
    lease_sha256='0fdf6f6a974649613ce463c948ae22979fb481e467e9047000abaffc4760e67e',
    complete_sha256='1c50579c172367894fe83921d84e4053803dd451c1c3e693d7881f1ff673e6f8',
    manifest_sha256='4aa1d671fb4b1862af28ea9f9c39c3ca0df7f00eee92bcd512467b1da3e6311b')


def fail(reason):raise PermissionError('DEC1_RECOVERY32_INTAKE_'+reason)


def owned(path,root=None):
    path=Path(path)
    if (path.is_symlink() or not path.is_file() or path.stat().st_uid!=os.getuid()
            or root is not None and not path.resolve().is_relative_to(Path(root).resolve())):
        fail('OWN_REGULAR_BOUND_MEMBER')
    return path


def read(path):return json.loads(owned(path).read_bytes())


def state(root,recipe,dataset):
    root=Path(root);checkpoint=root/'training/A_CONTROL/checkpoint-32'
    if root.is_symlink() or root.stat().st_uid!=os.getuid() or checkpoint.is_symlink():
        fail('OWN_SOURCE_OR_DESTINATION_ROOT')
    complete_path=owned(checkpoint/'CHECKPOINT_COMPLETE.json',root)
    if (sha(complete_path)!=EXPECTED['complete_sha256'] or digest(recipe)!=RECIPE
            or dataset.manifest_sha!=EXPECTED['manifest_sha256']):
        fail('EXACT_SAVED32_COMPLETE_RECIPE_AND_ORIGINAL_DATA')
    complete=read(complete_path)
    if (complete.get('global_step')!=32 or complete.get('full_resume') is not True
            or complete.get('origin_job_id')!=EXPECTED['job_id']
            or complete.get('origin_source_sha')!=EXPECTED['source_sha']):
        fail('REAL_NATIVE32_NOT_PARENT_INFERENCE_OR_LOST1024')
    for name in complete['files']:
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or p.as_posix()!=name or name in ('','.','CHECKPOINT_COMPLETE.json'):
            fail('CANONICAL_RELATIVE_FULL_STATE_MEMBER')
        owned(checkpoint/p,checkpoint)
    if checked_resume(root,'A_CONTROL',checkpoint,recipe,dataset)!=32:
        fail('MATURE_FULL_RESUME_CONTRACT')
    if sha(complete_path)!=EXPECTED['complete_sha256']:
        fail('COMPLETE_MANIFEST_CHANGED_DURING_READ')
    return complete


def inspect_closed_source(root,*,recipe,dataset):
    """Audit registered closed source bytes; never manufacture a new lease."""
    root=Path(root);manifest_path=owned(root/'RUN_MANIFEST.json',root)
    terminal_path=owned(root/'TERMINAL_RECONCILIATION.json',root)
    binding_path=owned(root/'OUTPUT_BINDING.private.json',root)
    binding=read(binding_path);lease_path=owned(binding['lease'])
    paths=(manifest_path,terminal_path,binding_path,lease_path)
    before={str(p):sha(p) for p in paths}
    if (sha(manifest_path)!=EXPECTED['run_manifest_sha256']
            or sha(terminal_path)!=EXPECTED['terminal_sha256']
            or sha(lease_path)!=EXPECTED['lease_sha256']):
        fail('EXACT_ACTUAL_MANIFEST_TERMINAL_AND_CLOSED_LEASE')
    m=read(manifest_path);terminal=read(terminal_path);lease=read(lease_path)
    if (binding.get('run_root')!=str(root) or binding.get('uid')!=os.getuid()
            or binding.get('run_manifest_sha256')!=EXPECTED['run_manifest_sha256']
            or m['code'].get('vla_commit')!=EXPECTED['source_sha']
            or lease.get('status')!='CLOSED' or lease.get('run_uuid')!=root.name
            or lease.get('terminal_receipt_sha256')!=EXPECTED['terminal_sha256']
            or terminal.get('status')!='CLOSED_FIRST32_COST_STOP_FULL_STATE_PRESERVED_POSTSAVE_ENGINEERING_FAILURE'
            or terminal.get('job_id')!=EXPECTED['job_id'] or terminal.get('recoverable_full_state') is not True
            or terminal.get('global_step')!=32 or terminal.get('settlement',{}).get('unknown_effects')!=[]):
        fail('CLOSED_KNOWN_EFFECTS_WITH_FULL_NATIVE_STATE')
    effects=terminal['effects']
    if (effects.get('job_id')!=EXPECTED['job_id'] or effects.get('request_id')!=EXPECTED['request_id']
            or effects.get('request_sha256')!=EXPECTED['request_sha256']
            or effects.get('source_sha')!=EXPECTED['source_sha'] or effects.get('recipe_sha256')!=RECIPE
            or effects.get('arm')!='A_CONTROL' or effects.get('start_native_step')!=0
            or effects.get('last_native_step')!=32 or effects.get('optimizer_updates_committed')!=32
            or effects.get('pending_updates')!=[] or effects.get('unknown_query_intents')!=[]
            or effects.get('effect_journal_complete') is not True):
        fail('EXACT_SETTLED32_NOT_CUMULATIVE1056_OR_UNCERTAIN_UPDATE')
    complete=state(root,recipe,dataset)
    if before!={str(p):sha(p) for p in paths}:fail('SOURCE_METADATA_CHANGED_DURING_READ')
    return dict(schema='DEC1_RECOVERY32_READONLY_FULL_STATE_IMPORT_PROOF_V1',
        status='PASS_CLOSED_SOURCE_FULL_MEMBER_BYTES_NOT_TRAINING_ADMISSION',
        authority_release=RELEASE,training_lineage_key=LINEAGE,source=dict(EXPECTED),
        global_step=32,actual_consumed_cursor=1024,recipe_sha256=RECIPE,statistics_sha256=R0,
        members=dict(complete['files']),new_GPU_PBS_models_queries_updates_physics=0)


def verify_cold_import(root,*,recipe,dataset):
    """Recheck the exact own-root copy against its immutable owner proof."""
    root=Path(root);proof_path=owned(root/PROOF,root);m=read(root/'RUN_MANIFEST.json')
    if (proof_path.stat().st_mode&0o222 or m.get('authority',{}).get('recovery_authorization_release')!=RELEASE
            or m.get('execution',{}).get('sealed_inputs',{}).get(PROOF)!=sha(proof_path)):
        fail('NEW_MANIFEST_MUST_SEAL_THE_EXACT_READONLY_SOURCE_PROOF')
    proof=read(proof_path)
    expected=dict(schema='DEC1_RECOVERY32_READONLY_FULL_STATE_IMPORT_PROOF_V1',
        status='PASS_CLOSED_SOURCE_FULL_MEMBER_BYTES_NOT_TRAINING_ADMISSION',authority_release=RELEASE,
        training_lineage_key=LINEAGE,source=EXPECTED,global_step=32,actual_consumed_cursor=1024,
        recipe_sha256=RECIPE,statistics_sha256=R0,new_GPU_PBS_models_queries_updates_physics=0)
    if set(proof)!=set(expected)|{'members'} or any(proof.get(k)!=v for k,v in expected.items()):
        fail('EXACT_CLOSED_SOURCE32_PROOF_NOT_GENERIC_PARENT_OR_RESTART')
    complete=state(root,recipe,dataset)
    if complete['files']!=proof['members']:fail('COPIED_COMPLETE_MEMBER_SET_MUST_MATCH_SOURCE')
    return dict(status='PASS_COLD_OWN_FULL_STATE32_BYTES_NOT_ALLOCATION_OR_CUDA_RESUME',
        global_step=32,actual_consumed_cursor=1024,checkpoint_complete_sha256=EXPECTED['complete_sha256'],
        source_proof_sha256=sha(proof_path),old_lost1024_costs_not_reset=True,
        new_GPU_PBS_models_queries_updates_physics=0)


def stage_in_allocation(root,*,recipe,dataset,guard,allocation):
    """Create the exact copy only inside the funded, admitted native job."""
    import shutil
    from scripts.pi0294.pv1.pbs_owner import publish_release_record as once
    from .a40_training_context import current_directory,sealed
    # Allocation itself is checked by the caller before torch/premodel. This
    # guard binds the fresh owner, request, deadline and active native lease.
    root=Path(root);guard()
    if not os.environ.get('PBS_JOBID'):fail('COPY_ONLY_INSIDE_FUNDED_NATIVE_ALLOCATION')
    directory=current_directory(root,allocation)
    request=read(directory/'REQUEST.json')
    from .segmented_training import limits
    if (limits(request)!=(32,2048) or request['training_block']['arm']!='A_CONTROL'
            or request['additional_output_upper_bytes']!=70000000000):fail('FUNDED_EXACT_A32_IMPORT_SCOPE')
    binding=sealed(root,'plans/RECOVERY32_SOURCE_BINDING.private.json')
    if binding.get('checkpoint_complete_sha256')!=EXPECTED['complete_sha256']:fail('SOURCE32_BINDING')
    source=Path(binding['root']);proof=inspect_closed_source(source,recipe=recipe,dataset=dataset)
    if proof!=sealed(root,PROOF):fail('SEALED_SOURCE_PROOF_MUST_MATCH_ACTUAL_CLOSED_SOURCE')
    checkpoint=root/'training/A_CONTROL/checkpoint-32'
    if checkpoint.exists():fail('NO_PARTIAL_IMPORT_RETRY_OR_CHECKPOINT_OVERWRITE')
    from .checkpoint_storage import POLICY,_parent_files,_owned_file
    policy=sealed(root,POLICY);inputs=sealed(root,'plans/PV1_TRAIN_INPUTS.private.json')
    parent=Path(inputs['CORR_parent_checkpoint']);parent_files=_parent_files(root,parent)
    frozen=policy['members']
    if len(frozen)!=1:fail('ONE_EXACT_SEALED_FROZEN_SHARD_ONLY')
    name,expected=next(iter(frozen.items()))
    # Policy values are digests, as in the existing post-save implementation.
    expected=expected['sha256'] if isinstance(expected,dict) else expected
    if proof['members'].get(name)!=expected or parent_files.get(name)!=expected:
        fail('FROZEN_IMPORTED_MEMBER_EQUALS_PROTECTED_ORIGINAL_PARENT')
    member=_owned_file(parent/name,parent);guard()
    if sha(member)!=expected:fail('ACTUAL_PARENT_FROZEN_BYTES')
    original=member.stat()
    once(root/'control/RECOVERY32_IMPORT_INTENT.json',dict(job_id=os.environ['PBS_JOBID'],
        source_complete_sha256=EXPECTED['complete_sha256'],proof_sha256=sha(root/PROOF),new_updates=0))
    checkpoint.parent.mkdir(parents=True,exist_ok=True);checkpoint.mkdir(mode=0o700)
    for n in proof['members']:
        guard();destination=checkpoint/n;destination.parent.mkdir(parents=True,exist_ok=True)
        if n==name:
            if member.stat().st_dev!=checkpoint.stat().st_dev:fail('SAME_FILESYSTEM_FROZEN_LINK_ONLY')
            os.link(member,destination)
        else:
            with owned(source/'training/A_CONTROL/checkpoint-32'/n).open('rb') as src,destination.open('xb') as dst:
                shutil.copyfileobj(src,dst,1024*1024);dst.flush();os.fsync(dst.fileno())
            destination.chmod(0o400)
    raw=(source/'training/A_CONTROL/checkpoint-32/CHECKPOINT_COMPLETE.json').read_bytes()
    with (checkpoint/'CHECKPOINT_COMPLETE.json').open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    (checkpoint/'CHECKPOINT_COMPLETE.json').chmod(0o400)
    if (member.stat().st_mode,member.stat().st_ino,member.stat().st_mtime_ns)!=(original.st_mode,original.st_ino,original.st_mtime_ns):
        fail('PROTECTED_PARENT_MODE_INODE_AND_MTIME_UNCHANGED')
    result=verify_cold_import(root,recipe=recipe,dataset=dataset);guard()
    once(root/'control/RECOVERY32_IMPORT_RESULT.json',dict(result,job_id=os.environ['PBS_JOBID'],
        frozen_shard_hardlink_not_independent_backup=name,source_state_preserved=True))
    return result
