"""Admit the exact consumed bank of a closed sibling, never an arbitrary cache."""
import os
from pathlib import Path

from scripts.pi0292.durable import read
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.e1.scope import STATS
from .decision import AUTHORITY

PLAN='plans/REUSED_NEW_BANK_BINDING.private.json'


def preflight(root,manifest,inputs,stage):
    """Reject stale metadata-only bank bindings before reserving a GPU job."""
    root=Path(root)
    for name in {'RUN_MANIFEST.json','OUTPUT_BINDING.private.json',*manifest['execution']['sealed_inputs']}:
        path=root/name
        if (path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve())
                or path.stat().st_uid!=os.getuid() or path.stat().st_mode&0o222):
            raise PermissionError('PV1_PREQUEUE_FINAL_BINDING_MUST_BE_OWN_IMMUTABLE_FILE:'+name)
    target=Path(inputs['new_dataset'])
    if target.is_symlink() or not target.is_dir() or target.stat().st_uid!=os.getuid():
        raise PermissionError('PV1_PREQUEUE_NEW_BANK_OWN_DIRECTORY')
    hashes={}
    for name in ('DATASET_MANIFEST.json','READY.json','statistics.json','PAYLOAD_MANIFEST.json'):
        path=target/name
        if path.is_symlink() or not path.is_file() or path.stat().st_uid!=os.getuid():
            raise PermissionError('PV1_PREQUEUE_COMPLETE_BANK_METADATA_REQUIRED:'+name)
        hashes[name]=sha(path)
    ready=read(target/'READY.json')
    if (hashes['DATASET_MANIFEST.json']!=stage['new_bank_manifest_sha256']
            or hashes['READY.json']!=stage['new_bank_ready_sha256']
            or hashes['statistics.json']!=STATS
            or ready['payload_manifest_sha256']!=hashes['PAYLOAD_MANIFEST.json']):
        raise PermissionError('PV1_PREQUEUE_FIXED_BANK_METADATA_HASHES')
    parents=read(target/'DATASET_MANIFEST.json')['episodes']
    raw=read(inputs['new_raw_bindings'])
    if set(raw)!={row['id'] for row in parents}:
        raise PermissionError('PV1_PREQUEUE_EXACT_BANK_RAW_BINDING_MEMBERS')
    for row in parents:
        folder=target/'raw'/row['id'];numeric=target/row['file']
        if (not folder.is_dir() or folder.is_symlink() or not numeric.is_file() or numeric.is_symlink()
                or not numeric.resolve().is_relative_to(target.resolve())
                or Path(raw[row['id']]).resolve()!=folder.resolve()):
            raise PermissionError('PV1_PREQUEUE_RAW_AND_NUMERIC_MUST_USE_ACTUAL_BANK')
    reuse=None
    if not target.resolve().is_relative_to(root.resolve()):
        reuse=validate(root,target,read(root/'plans/PV1_HF_BANK_DELIVERY.json'),stage)
    return dict(status='PASS_PREQUEUE_IMMUTABLE_COMPLETE_BANK_BINDING_NOT_FULL_PAYLOAD_REHASH',
                metadata_sha256=hashes,parent_episodes=len(parents),
                source_child_uuid=None if reuse is None else reuse['source_child_uuid'],
                full_member_hashes_still_required_inside_allocation=True)


def validate(root,target,delivery,stage):
    root,target=Path(root),Path(target);manifest=read(root/'RUN_MANIFEST.json');path=root/PLAN
    from . import evaluation_platform as evaluation
    from scripts.pi0294.dec1 import a40_training_context as context
    if evaluation.cold_profile(manifest) and context.CONSUMERS in manifest['execution']['sealed_inputs']:
        # Reuse the already paid, closed Vanda CPU readers. Their original
        # validators check actual leases, registry, receipts and data paths;
        # they do not grant a training or evaluation allocation.
        bound=root/context.CONSUMERS
        if (bound.is_symlink() or bound.stat().st_mode&0o222
                or sha(bound)!=manifest['execution']['sealed_inputs'][context.CONSUMERS]):
            raise PermissionError('PV1_VANDA_DEV_PRESEALED_CLOSED_CPU_READERS')
        actual=context.verify_consumers(root);records=context.consumer_records(root)
        source=Path(records['new139']['root'])
        if (source.parent.resolve()!=root.parent.resolve() or source==root
                or target.resolve()!=(source/'datasets/FROZEN_TRAIN_BANK').resolve()
                or sha(source/'plans/PV1_HF_BANK_DELIVERY.json')!=sha(root/'plans/PV1_HF_BANK_DELIVERY.json')
                or read(source/'plans/PV1_HF_BANK_DELIVERY.json')!=delivery
                or sha(target/'READY.json')!=stage['new_bank_ready_sha256']
                or sha(target/'statistics.json')!=STATS):
            raise PermissionError('PV1_VANDA_DEV_EXACT_CLOSED_CPU_BANK_AND_DELIVERY')
        return dict(status='PASS_PRESEALED_CLOSED_VANDA_CPU_BANK_REUSE_NOT_YET_REHASHED',
            binding_sha256=sha(bound),source_child_uuid=source.name,
            closed_CPU_readers=actual,dataset_manifest_sha256=stage['new_bank_manifest_sha256'],
            fixed_HF_revision=delivery['revision'],new_download_bytes=0,new_extraction_bytes=0,
            prior_consumption_does_not_replace_current_hash_verification=True)
    if (path.is_symlink() or path.stat().st_mode&0o222
            or manifest['execution']['sealed_inputs'].get(PLAN)!=sha(path)):
        raise PermissionError('PV1_REUSE_REQUIRES_PRESEALED_EXACT_BANK_BINDING')
    plan=read(path);source=Path(plan['source_child_root']);parent=root.parent.parent
    if (plan['schema']!='PI0294_PV1_CLOSED_SIBLING_BANK_REUSE_V1'
            or source==root or source.is_symlink() or not source.is_dir()
            or source.parent.resolve()!=(parent/'children').resolve()
            or source.stat().st_uid!=os.getuid() or target.is_symlink()
            or target.resolve()!=(source/'datasets/FROZEN_TRAIN_BANK').resolve()
            or not target.is_dir() or target.stat().st_uid!=os.getuid()):
        raise PermissionError('PV1_REUSE_ONLY_REGISTERED_SAME_CAMPAIGN_SIBLING_BANK')
    m=read(source/'RUN_MANIFEST.json');binding=read(source/'OUTPUT_BINDING.private.json')
    lease=read(binding['lease']);consumed=source/'evidence/NEW_BANK_CONSUMPTION_READY.private.json'
    adoption=read(parent/'PV1_ADOPTION_AND_RESOURCE_BINDING.json')
    if (sha(source/'RUN_MANIFEST.json')!=plan['source_manifest_sha256']
            or adoption['authority_release']!=AUTHORITY or plan['authority_release']!=AUTHORITY
            or plan['campaign_run_uuid']!=adoption['run_uuid'] or m['authority']['protocol_ref']!=AUTHORITY
            or sha(source/'OUTPUT_BINDING.private.json')!=m['execution']['output_binding_sha256']
            or binding['run_uuid']!=source.name or m['run_uuid']!=source.name
            or Path(binding['run_root']).resolve()!=source.resolve()
            or binding['uid']!=os.getuid() or lease['status']!='CLOSED' or lease['authority_release']!=AUTHORITY
            or lease['run_uuid']!=source.name or lease['all_registered_workers_terminal'] is not True
            or consumed.is_symlink() or consumed.stat().st_mode&0o222
            or sha(consumed)!=plan['source_consumption_receipt_sha256']):
        raise PermissionError('PV1_REUSE_SOURCE_CLOSED_TERMINAL_PINNED_IDENTITY')
    campaign=read(parent/'HOPPER_RESOURCE_LEDGER.json')
    owned=[v for v in campaign['PBS_submissions'] if v['run_uuid']==source.name]
    if (campaign['campaign_run_uuid']!=adoption['run_uuid'] or campaign['authority_release']!=AUTHORITY
            or not owned or any(v['status']!='NATIVE_TERMINAL' for v in owned)
            or campaign['unknown_effects'] or campaign['pending_updates']):
        raise PermissionError('PV1_REUSE_SOURCE_NATIVE_JOBS_TERMINAL_NO_UNKNOWN_EFFECTS')
    for p in [*(source/'worker_processes').glob('*.json'),*(source/'aux_worker_processes').glob('*.json')]:
        worker=read(p);stat=Path('/proc',str(worker['pid']),'stat');fields=stat.read_text().split() if stat.exists() else None
        if fields and fields[2]!='Z' and fields[21]==str(worker['proc_start']):
            raise PermissionError('PV1_REUSE_SOURCE_REGISTERED_WORKER_STILL_ACTIVE')
    proof=read(consumed)
    if (proof['status']!='PASS_ACTUAL_FULL_NEW_BANK_CONSUMPTION' or proof['full_member_hashes_verified'] is not True
            or proof['dataset_manifest_sha256']!=stage['new_bank_manifest_sha256']
            or proof['download_revision']!=delivery['revision'] or proof['statistics_sha256']!=STATS
            or proof['delivery_binding_sha256']!=sha(source/'plans/PV1_HF_BANK_DELIVERY.json')
            or read(source/'plans/PV1_HF_BANK_DELIVERY.json')!=delivery
            or sha(target/'DATASET_MANIFEST.json')!=stage['new_bank_manifest_sha256']
            or sha(target/'READY.json')!=stage['new_bank_ready_sha256']
            or sha(target/'statistics.json')!=STATS
            or plan['later_data_hot_injected'] is not False or plan['read_only_shared_input'] is not True):
        raise PermissionError('PV1_REUSE_EXACT_QUALIFIED_REVISION_MANIFEST_STATS_NO_HOT_DATA')
    return dict(status='PASS_PRESEALED_CLOSED_SIBLING_BANK_REUSE_NOT_YET_REHASHED',
        binding_sha256=sha(path),source_child_uuid=source.name,
        source_manifest_sha256=plan['source_manifest_sha256'],source_consumption_receipt_sha256=sha(consumed),
        dataset_manifest_sha256=proof['dataset_manifest_sha256'],fixed_HF_revision=proof['download_revision'],
        new_download_bytes=0,new_extraction_bytes=0,prior_consumption_does_not_replace_current_hash_verification=True)
