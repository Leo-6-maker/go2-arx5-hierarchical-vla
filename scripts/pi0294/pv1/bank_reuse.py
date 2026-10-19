"""Admit the exact consumed bank of a closed sibling, never an arbitrary cache."""
import os
from pathlib import Path

from scripts.pi0292.durable import read
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.e1.scope import STATS
from .decision import AUTHORITY

PLAN='plans/REUSED_NEW_BANK_BINDING.private.json'


def validate(root,target,delivery,stage):
    root,target=Path(root),Path(target);manifest=read(root/'RUN_MANIFEST.json');path=root/PLAN
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
