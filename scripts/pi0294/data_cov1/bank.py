"""Immutable stage banks, separate from PV1 and with no invented READY state."""
import os
from pathlib import Path
import re

import numpy as np

from scripts.pi0292.durable import read
from scripts.pi0292.hf_bank import contained
from scripts.pi0294.e1.scope import STATS
from scripts.pi0294.fd1 import put,sha
from scripts.pi0294.pv1.data_delivery import pack_archive,safe_json
from scripts.pi0294.scene_cards import digest
from .mixed_batch import qualify_all
from .reports import stage_reports
from .scope import sealed_file

SCHEMA='PI0294_DATA_COV1_ADMITTED_STAGE_BANK_V1'
LOCAL_READY='DATA_COV1_LOCAL_STAGE_BANK_QUALIFIED_NOT_HF_DELIVERED'


def public_json(value):
    safe_json(value)
    if isinstance(value,dict):
        for key,item in value.items():public_json(key);public_json(item)
    elif isinstance(value,(list,tuple)):
        for item in value:public_json(item)
    elif isinstance(value,str) and value.startswith(('/var/cache/','/tmp/','/run/','/media/','/data/')):
        raise PermissionError('DATA_COV1_PUBLIC_BANK_FORBIDS_HOST_BOUND_PATHS')


def manifest(groups,stage,captures,exports,*,actual_cards,population_sha256,adoption_sha256):
    reports=stage_reports(groups,stage,captures,exports,actual_cards=actual_cards)
    if (not reports['stage_gate']['count_gate_pass'] or not reports['first_four_gate']['count_gate_pass']
            or not re.fullmatch('[0-9a-f]{64}',population_sha256)
            or not re.fullmatch('[0-9a-f]{64}',adoption_sha256)):
        raise PermissionError('DATA_COV1_NO_STAGE_BANK_BEFORE_FIXED_FIRST_FOUR_AND_STAGE_GATES')
    rows=[]
    admitted={r['id']:r for r in reports['all_attempts']['rows'] if r['admitted']}
    for identifier in sorted(admitted):
        row=dict(exports[identifier]);row.update(actual_geometry_fingerprint=admitted[identifier]['actual_geometry_fingerprint'],
            actual_geometry_revision=admitted[identifier]['actual_geometry_revision'])
        rows.append(row)
    value=dict(schema=SCHEMA,domain='FIXED_BASE_ARX5_TASK_SIM',stage=stage,statistics_sha256=STATS,
        statistics_refit=False,population_plan_sha256=population_sha256,adoption_sha256=adoption_sha256,
        all_attempts_sha256=digest(reports['all_attempts']),stage_gate_sha256=digest(reports['stage_gate']),
        first_four_gate_sha256=digest(reports['first_four_gate']),stage_gate=reports['stage_gate'],
        first_four_gate=reports['first_four_gate'],episodes=rows,later_rows_hot_injected=False,
        observations_and_labels='REAL_TWO_RGB_ACTUAL_STATE_AND_ACTUALLY_COMMITTED_ABSOLUTE_EEF10',
        evaluation_rows_used_for_training=False,PV1_bank_changed=False,selected='NONE',deployment='NONE')
    public_json(value)
    return value,reports


def validate_manifest(value):
    if (value.get('schema')!=SCHEMA or value.get('statistics_sha256')!=STATS
            or value.get('statistics_refit') is not False or value.get('later_rows_hot_injected') is not False
            or value.get('evaluation_rows_used_for_training') is not False or value.get('PV1_bank_changed') is not False
            or value.get('stage') not in ('P24','P72','P144') or not value.get('episodes')
            or value.get('stage_gate',{}).get('count_gate_pass') is not True
            or value.get('first_four_gate',{}).get('count_gate_pass') is not True
            or value.get('stage_gate_sha256')!=digest(value['stage_gate'])
            or value.get('first_four_gate_sha256')!=digest(value['first_four_gate'])):
        raise PermissionError('DATA_COV1_FROZEN_LOCAL_BANK_EXACT_STAGE_QUALIFICATION')
    ids=set();raws=set()
    for row in value['episodes']:
        if (row.get('split')!='TRAIN' or row.get('family')!='DATA_COV1'
                or not re.fullmatch(r'PI0294-DATA_COV1-TRAIN-AP[012]-(?:\+Y|-Y)-(?:NEAR|FAR)-[0-9]{3}-T[12]-C04',row['id'])
                or row['id'] in ids or row['source_raw_manifest_sha256'] in raws
                or row['file']!='numeric/'+row['id']+'.npz' or row['nominal_clearance_m']!=.004
                or row['windows']!=601 or row.get('full_task_success') is not True
                or row.get('independent_raw_admitted') is not True or row.get('admitted_for_training') is not True
                or row['status']!='PASS_SAVED_TEACHER_EXPORT_AND_OFFICIAL_CHAIN'
                or row['codec']['status']!='PASS_FIXED_ORIGINAL_V3_0_NO_SILENT_CLIP'
                or not re.fullmatch('[0-9a-f]{64}',row['actual_geometry_fingerprint'])):
            raise PermissionError('DATA_COV1_ONLY_UNIQUE_ACTUALLY_ADMITTED_FULL_TRAIN_PARENTS')
        ids.add(row['id']);raws.add(row['source_raw_manifest_sha256'])
    public_json(value)
    return value


def parent_files(root,row,guard):
    root=Path(root);numeric=contained(root,row['file']);sealed_file(numeric,row['sha256'])
    with np.load(numeric,allow_pickle=False) as z:
        if (z['actual_state10'].shape!=(641,10) or z['committed_absolute10'].shape!=(640,10)
                or z['valid_window'].shape!=(640,) or not np.array_equal(z['valid_window'],np.arange(640)<601)
                or z['rgb_paths'].shape!=(641,2) or z['rgb_sha256'].shape!=(641,2)
                or z['label_representation'].item()!='ABSOLUTE_COMMITTED_EEF10'):
            raise ValueError('DATA_COV1_FULL_NUMERIC_AND_VALID_H40_WINDOW_IDENTITY')
        names=z['rgb_paths'].copy();hashes=z['rgb_sha256'].copy()
    members={row['file']:numeric};seen=set()
    for index,(paths,digests) in enumerate(zip(names,hashes)):
        for camera,(relative,expected) in enumerate(zip(paths,digests)):
            guard();name=str(relative);prefix=('body_rgb','wrist_rgb')[camera]
            if name!=prefix+'/'+str(index).zfill(4)+'.png':
                raise ValueError('DATA_COV1_EXACT_641_ACTUAL_FRAME_NAMES')
            public='raw/'+row['id']+'/'+name;p=contained(root,public);sealed_file(p,str(expected))
            if public in seen:raise ValueError('DATA_COV1_DUPLICATE_PHYSICAL_FRAME_MEMBER')
            seen.add(public);members[public]=p
    return members


def assemble(root,value,sources,processor,official_source,cosmos,guard):
    """Caller provides a registered PACK scope, reservation and deadline guard."""
    from .scope import validate
    root=Path(root);m,_=validate(root,'PACK');validate_manifest(value)
    if (m['stage']!=value['stage'] or sha(root/'plans/statistics.json')!=STATS
            or any(m['limits'][k] for k in ('Teacher_episodes','engine_starts','physics_ticks','GR00T_queries','training_updates','PBS_submissions'))):
        raise PermissionError('DATA_COV1_REGISTERED_PACK_ZERO_PHYSICS_AND_NEURAL_BUDGET')
    if set(sources)!={r['id'] for r in value['episodes']}:
        raise PermissionError('DATA_COV1_EXACT_CLOSED_EXPORT_SOURCE_SET')
    out=root/'datasets/FROZEN_TRAIN_BANK'
    if out.exists():raise FileExistsError('DATA_COV1_PRESERVE_FIRST_BANK_INCLUDING_PARTIAL')
    bindings=[];allmembers={}
    for row in value['episodes']:
        guard();source=Path(sources[row['id']]);child=source.parent.parent
        if (child.parent.resolve()!=root.parent.resolve() or source.is_symlink()
                or not source.resolve().is_relative_to(child.resolve())):
            raise PermissionError('DATA_COV1_ONLY_SAME_CAMPAIGN_REGISTERED_SIBLING_EXPORT')
        binding=read(child/'OUTPUT_BINDING.private.json');ledger=read(child/'RESOURCE_LEDGER.json')
        if (binding['uid']!=os.getuid() or binding['host']!=os.uname().nodename
                or Path(binding['run_root']).resolve()!=child.resolve()
                or read(binding['lease'])['status']!='CLOSED'
                or ledger['status']!='TERMINAL_DATA_COV1_SAVED_EXPORT' or ledger['unknown_effects']):
            raise PermissionError('DATA_COV1_CLOSED_QUALIFIED_EXPORT_ONLY_NO_ACTIVE_MEMBER_MUTATION')
        receipt=source/'EXPORT_RECEIPT.json';sealed_file(receipt);saved=read(receipt)
        if any(row.get(k)!=v for k,v in saved.items()):
            raise PermissionError('DATA_COV1_BANK_ROW_MUST_EQUAL_ORIGINAL_QUALIFIED_EXPORT')
        sealed_file(source/'statistics.json',STATS)
        allmembers.update(parent_files(source,row,guard))
        bindings.append(dict(id=row['id'],export_root=str(source),export_receipt_sha256=sha(receipt)))
    batch=qualify_all(bindings,processor,official_source,cosmos,guard)
    guard();out.mkdir();os.link(root/'plans/statistics.json',out/'statistics.json')
    for name,p in allmembers.items():
        guard();target=out/name;target.parent.mkdir(parents=True,exist_ok=True);os.link(p,target)
    put(out/'DATASET_MANIFEST.json',value);put(out/'CROSS_PARENT_QUALIFICATION.json',batch)
    files=[]
    for p in sorted(out.rglob('*')):
        if p.is_file():guard();files.append(dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)))
    put(out/'PAYLOAD_MANIFEST.json',files)
    ready=dict(schema='PI0294_DATA_COV1_LOCAL_STAGE_READY_V1',status=LOCAL_READY,stage=value['stage'],
        dataset_manifest_sha256=sha(out/'DATASET_MANIFEST.json'),payload_manifest_sha256=sha(out/'PAYLOAD_MANIFEST.json'),
        cross_parent_qualification_sha256=sha(out/'CROSS_PARENT_QUALIFICATION.json'),statistics_sha256=STATS,
        files=len(files),logical_payload_bytes=sum(f['bytes'] for f in files),HF_delivery_claim=False,
        downstream_native_consumption_claim=False,later_rows_hot_injected=False)
    put(out/'READY.json',ready)
    return ready


def bank_members(bank,guard):
    bank=Path(bank)
    if bank.is_symlink() or not bank.is_dir() or bank.stat().st_uid!=os.getuid():
        raise PermissionError('DATA_COV1_REAL_OWN_STAGE_BANK_DIRECTORY_REQUIRED')
    value=read(bank/'DATASET_MANIFEST.json');validate_manifest(value)
    ready=read(bank/'READY.json')
    if (ready['status']!=LOCAL_READY or ready['dataset_manifest_sha256']!=sha(bank/'DATASET_MANIFEST.json')
            or ready['payload_manifest_sha256']!=sha(bank/'PAYLOAD_MANIFEST.json')
            or ready['cross_parent_qualification_sha256']!=sha(bank/'CROSS_PARENT_QUALIFICATION.json')
            or sha(bank/'statistics.json')!=STATS):
        raise PermissionError('DATA_COV1_FULL_LOCAL_BANK_PROOF_REQUIRED_BEFORE_PACKAGE')
    mixed=read(bank/'CROSS_PARENT_QUALIFICATION.json')
    ids={r['id'] for r in value['episodes']}
    batches=mixed.get('batches',[])
    if (mixed['status']!='PASS_EVERY_PARENT_IN_BOUNDED_CROSS_PARENT_OFFICIAL_CPU_BATCHES'
            or mixed['distinct_parent_count']!=len(value['episodes'])
            or not batches or {i for b in batches for i in b.get('distinct_parent_ids',[])}!=ids
            or any(b.get('status')!='PASS_ACTUAL_CROSS_PARENT_OFFICIAL_CPU_BATCH'
                or b.get('parent_count')!=2 or b.get('deterministic_repeat') is not True
                or b.get('model_forward') is not False for b in batches)):
        raise PermissionError('DATA_COV1_EVERY_ADMITTED_PARENT_NEEDS_NATIVE_MIXED_BATCH_QUALIFICATION')
    members={n:contained(bank,n) for n in ('DATASET_MANIFEST.json','statistics.json','CROSS_PARENT_QUALIFICATION.json')}
    for row in value['episodes']:members.update(parent_files(bank,row,guard))
    entries=read(bank/'PAYLOAD_MANIFEST.json');expected={n:dict(path=n,bytes=p.stat().st_size,sha256=sha(p)) for n,p in members.items()}
    if (len(entries)!=len(expected) or any(expected.get(r['path'])!=r for r in entries)
            or ready['files']!=len(entries) or ready['logical_payload_bytes']!=sum(r['bytes'] for r in entries)):
        raise ValueError('DATA_COV1_FULL_PAYLOAD_MEMBER_SET_SIZE_HASH_REQUIRED')
    if {p.relative_to(bank).as_posix() for p in bank.rglob('*') if p.is_file()}!=set(members)|{'PAYLOAD_MANIFEST.json','READY.json'}:
        raise PermissionError('DATA_COV1_NO_XML_PRIVATE_METADATA_OR_VENDOR_ASSETS_IN_BANK')
    for name in ('PAYLOAD_MANIFEST.json','READY.json'):members[name]=contained(bank,name)
    return value,ready,members


def package(bank,output,guard,*,maximum_archive_bytes):
    value,ready,members=bank_members(bank,guard);output=Path(output)
    if output.exists():raise FileExistsError('DATA_COV1_PRESERVE_EXISTING_PACKAGE_NO_BLIND_RETRY')
    if type(maximum_archive_bytes) is not int or maximum_archive_bytes<=0:
        raise PermissionError('DATA_COV1_EXPLICIT_RESERVED_ARCHIVE_BOUND')
    if sum(p.stat().st_size for p in members.values())+len(members)*4096+1024**2>maximum_archive_bytes:
        raise PermissionError('DATA_COV1_ARCHIVE_BOUND_INCLUDES_REGULAR_TAR_HEADER_OVERHEAD')
    output.mkdir();index={}
    for i in range(0,len(value['episodes']),8):
        ids={r['id'] for r in value['episodes'][i:i+8]}
        subset={n:p for n,p in members.items() if any(n=='numeric/'+identifier+'.npz' or n.startswith('raw/'+identifier+'/') for identifier in ids)}
        name='parents-'+str(i//8).zfill(3)+'.tar.gz';index[name]=pack_archive(output/name,subset,guard)
        if sum(p.stat().st_size for p in output.iterdir())>maximum_archive_bytes:
            raise PermissionError('DATA_COV1_ACTUAL_ARCHIVE_STORAGE_BOUND')
    index['metadata.tar.gz']=pack_archive(output/'metadata.tar.gz',{n:p for n,p in members.items() if '/' not in n},guard)
    receipt=dict(schema='PI0294_DATA_COV1_ARCHIVE_INDEX_V1',stage=value['stage'],archives=index,
        dataset_manifest_sha256=ready['dataset_manifest_sha256'],source_local_ready_sha256=sha(Path(bank)/'READY.json'),
        statistics_sha256=STATS,only_new_DATA_COV1_stage=True,PV1_or_old_models_reuploaded=False,
        READY_LAST_not_yet_published=True,HF_fixed_revision_readback_not_yet_performed=True)
    public_json(receipt);put(output/'ARCHIVE_MEMBER_MANIFEST.json',receipt)
    if sum(p.stat().st_size for p in output.iterdir())>maximum_archive_bytes:
        raise PermissionError('DATA_COV1_INDEX_INCLUSIVE_ARCHIVE_STORAGE_BOUND')
    return receipt
