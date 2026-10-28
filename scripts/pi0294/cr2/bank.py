"""Fixed-R1 CR2 bank: closed raw, paired audit, real readers, no new physics."""
import os
from pathlib import Path
import re
import shutil
import time

from scripts.pi0291_arx5_data.native_dataset import cpu_guard
from scripts.pi0292.durable import OwnerLock, read, write
from scripts.pi0294.cr1.numeric import serial
from scripts.pi0294.cr1.scope import install_import_guard
from scripts.pi0294.data_cov1.bank import parent_files, public_json
from scripts.pi0294.fd1 import put, sha
from scripts.pi0294.native_dataset import NativeDataset
from scripts.pi0294.pv1.data_delivery import pack_archive
from scripts.pi0294.pv1.export import batch_inventory
from scripts.pi0294.scene_cards import digest
from scripts.pi0292.train_adapter import build_processor
from .plan import AUTH, R1, bank_gate, cards
from .runtime import bind_cpu, wait_priority
from .scope import sealed_file, validate

SCHEMA='PI0294_CR2_FIXED_R1_P12_BANK_V1'
LOCAL='CR2_LOCAL_FROZEN_FULL_READER_QUALIFIED_NOT_HF_READY'


def configuration(profile=None):
    if profile is None:
        return dict(auth=AUTH,validate=validate,priority=None,cards=cards,gate=bank_gate,
            gate_status='PASS_CR2_DATA_BANK_GATE',schema=SCHEMA,bank='CR2_FIXED_R1_P12',
            family='CR2_FIXED_R1',local=LOCAL,adoption='CR2_ADOPTION_AND_SCOPE.json',
            operation='CR2_ASSEMBLE_AND_PACK_FIXED_R1_P12',attempts='ALL24_PAIRED_ATTEMPTS.json',
            output_cap=48*1024**3,archive_cap=7*1024**3,
            pack_status='PASS_CR2_LOCAL_BANK_AND_PACK_NOT_HF_READY',
            local_schema='PI0294_CR2_LOCAL_FROZEN_V1',index_schema='PI0294_CR2_ARCHIVE_INDEX_V1')
    from scripts.pi0294.vap1.bank import configuration as vap1_configuration
    return vap1_configuration(profile)


def manifest(groups, rows, *, population_sha256, adoption_sha256, profile=None):
    cfg=configuration(profile);gate=cfg['gate'](rows,groups)
    if gate['status']!=cfg['gate_status']:
        raise PermissionError('CR2_COMPLETE_ORIGINAL_24_DENOMINATOR_AND_BANK_GATE_REQUIRED')
    if any(not re.fullmatch('[0-9a-f]{64}',v) for v in (population_sha256,adoption_sha256)):
        raise PermissionError('CR2_EXACT_NATIVE_POPULATION_AND_ADOPTION_IDENTITIES')
    indexed={c['id']:c for c in cfg['cards'](groups)};episodes=[];raws=set()
    for row in sorted(rows,key=lambda r:r['id']):
        if not row['admitted']:continue
        card=indexed[row['id']];reader=row['official_reader']['R1'];codec=row['R1']
        if (row.get('task')!=card['task'] or row.get('scene_group')!=card['group_id']
                or row.get('status')!='PAIRED_FIXED_R0_R1_SAVED_DATA_COMPLETE'
                or codec.get('status')!='PASS_FIXED_ORIGINAL_V3_0_NO_SILENT_CLIP'
                or codec.get('actual_statistics_sha256')!=R1
                or reader.get('statistics_sha256')!=R1
                or reader.get('actual_anchors')!=[0,80,160,320,480,600]
                or any(reader.get(k) is not True for k in ('actual_dual_camera_frames',
                    'mask_H40_EEF10_zero_padding','deterministic_repeat'))
                or not re.fullmatch('[0-9a-f]{64}',row['source_raw_manifest_sha256'])
                or row['source_raw_manifest_sha256'] in raws):
            raise PermissionError('CR2_ONLY_DISTINCT_ACTUALLY_QUALIFIED_R1_TRAIN_PARENTS')
        raws.add(row['source_raw_manifest_sha256'])
        episodes.append(dict(id=card['id'],split='TRAIN',task=card['task'],family=cfg['family'],
            scene_group=card['group_id'],appearance_profile=card['appearance_profile'],direction=card['direction'],
            distance_bin=card['distance_bin'],windows=601,file='numeric/'+card['id']+'.npz',
            raw_root='raw/'+card['id'],nominal_clearance_m=.004,source_raw_manifest_sha256=row['source_raw_manifest_sha256'],
            paired_receipt_identity=digest(row),transition=row['transition'],full_task_success=True))
    value=dict(schema=cfg['schema'],domain='FIXED_BASE_ARX5_TASK_SIM',bank=cfg['bank'],statistics_sha256=R1,
        population_sha256=population_sha256,adoption_sha256=adoption_sha256,gate=gate,
        all_24_paired_attempts_identity=digest(rows),episodes=episodes,statistics_refit=False,
        PV1_bank_changed=False,later_rows_hot_injected=False,evaluation_rows_used_for_training=False,
        policy_success_claim=False,selected='NONE',deployment='NONE')
    if profile is not None:
        value['all_paired_attempts_identity']=value.pop('all_24_paired_attempts_identity')
    public_json(value)
    return value


def closed_child(parent, root, role, state, *, profile=None):
    cfg=configuration(profile)
    root=Path(root);parent=Path(parent)
    if root.is_symlink() or root.parent.resolve()!=(parent/'children').resolve():
        raise PermissionError('CR2_ONLY_REGISTERED_SAME_CAMPAIGN_SIBLING')
    binding=read(root/'OUTPUT_BINDING.private.json');m=read(root/'RUN_MANIFEST.json');ledger=read(root/'RESOURCE_LEDGER.json')
    if (binding['uid']!=os.getuid() or binding['host']!=os.uname().nodename
            or Path(binding['run_root']).resolve()!=root.resolve() or m['role']!=role
            or m['authority']['protocol_release']!=cfg['auth'] or read(binding['lease'])['status']!='CLOSED'
            or ledger['status']!=state or ledger['unknown_effects'] or (root/'STOP.json').exists()):
        raise PermissionError('CR2_EXACT_CLOSED_KNOWN_EFFECT_SOURCE_ONLY')
    return root


def raw_file(path, expected=None):
    """Historical capture files keep their real modes; never chmod their inodes."""
    p=Path(path)
    if p.is_symlink() or not p.is_file() or p.stat().st_uid!=os.getuid():
        raise PermissionError('CR2_CLOSED_RAW_OWN_REGULAR_MEMBER_ONLY')
    before=p.stat();actual=sha(p);after=p.stat()
    identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    if identity(before)!=identity(after) or expected is not None and actual!=expected:
        raise ValueError('CR2_CLOSED_RAW_CONTENT_OR_IDENTITY_CHANGED')
    return actual


def copy_raw(source, target, expected):
    raw_file(source,expected)
    target=Path(target)
    if target.exists():raise FileExistsError('CR2_PRESERVE_FIRST_BANK_COPY')
    with Path(source).open('rb') as src,target.open('xb') as dst:
        shutil.copyfileobj(src,dst)
    target.chmod(0o400)
    raw_file(source,expected);sealed_file(target,expected)


def qualify_all(bank, value, processor, inputs, guard, *, profile=None):
    cfg=configuration(profile)
    dataset=NativeDataset(bank,inputs['official_source'],inputs['cosmos'],for_optimization=True,
        raw_bindings={r['id']:str(bank/r['raw_root']) for r in value['episodes']})
    dataset.processor=processor;lookup={(r['id'],a):i for i,(r,a) in enumerate(dataset.rows)}
    ids=[r['id'] for r in value['episodes']];pairs=[ids[i:i+2] for i in range(0,len(ids)-1,2)]
    if len(ids)%2:pairs.append(ids[-2:])
    batches=[]
    for pair in pairs:
        if cfg['priority'] is None:wait_priority(bank.parent.parent,guard)
        else:wait_priority(bank.parent.parent,guard,cfg['priority'])
        samples=[]
        for identifier in pair:
            for anchor in (0,80,160,320,480,600):
                guard();samples.append(dataset[lookup[identifier,anchor]]['processed'])
        batch=batch_inventory(processor.collator(samples),len(samples))
        if batch!=batch_inventory(processor.collator(samples),len(samples)):
            raise ValueError('CR2_BANK_CROSS_PARENT_DETERMINISTIC_REPEAT')
        batches.append(dict(distinct_parent_ids=pair,anchors=[0,80,160,320,480,600],
            actual_dual_camera_frames=True,batch_sha256=digest(batch),samples=len(samples),deterministic_repeat=True))
    if {i for b in batches for i in b['distinct_parent_ids']}!=set(ids):
        raise ValueError('CR2_EVERY_ADMITTED_PARENT_MUST_ENTER_REAL_MIXED_BATCH')
    return dict(status='PASS_ALL_ADMITTED_CR2_PARENTS_REAL_OFFICIAL_MIXED_BATCHES',
        distinct_parent_count=len(ids),maximum_samples_per_batch=12,batches=batches,
        model_forward=False,new_physics=0,new_constructors=0)


def run(root, *, profile=None):
    cfg=configuration(profile)
    r=Path(root);m,inputs=cfg['validate'](r,'PACK');work=read(r/'plans/WORK_PLAN.private.json')
    ledger=read(r/'RESOURCE_LEDGER.json');assert ledger['status']=='RESERVED' and not ledger['unknown_effects']
    if work.get('operation')!=cfg['operation']:
        raise PermissionError('CR2_EXPLICIT_NEW_BANK_PACK_PLAN')
    lock=OwnerLock(r/'CR2_WORKER.lock');began=time.process_time();failure=None;bind_cpu(m)
    parent=r.parent.parent;out=r/'datasets'/cfg['bank'];last_storage=0.;peak=0
    output_limit=work['maximum_output_peak_bytes']
    if type(output_limit) is not int or not 0<output_limit<=cfg['output_cap']:
        lock.close();raise PermissionError('CR2_ORIGINAL_ALL_OUTPUT_PEAK_LIMIT')
    def guard():
        nonlocal last_storage,peak
        if (time.time()>=m['limits']['operation_deadline_epoch'] or time.process_time()-began>=m['limits']['CPU_s']-60
                or any((p/'STOP.json').exists() for p in (r,parent)) or shutil.disk_usage(r).free<40*1024**3):
            raise PermissionError('CR2_PACK_FINITE_CPU_WALL_STORAGE_OR_STOP')
        if time.monotonic()-last_storage>=15:
            seen=set();size=0
            for p in parent.rglob('*'):
                if p.is_symlink():continue
                if p.is_file():
                    s=p.stat();key=(s.st_dev,s.st_ino)
                    if key not in seen:size+=max(s.st_size,s.st_blocks*512);seen.add(key)
            peak=max(peak,size);last_storage=time.monotonic()
            if peak>output_limit:raise PermissionError('CR2_ACTUAL_ALL_OUTPUT_UNIQUE_INODE_PEAK_LIMIT')
    try:
        install_import_guard();cpu_guard()
        if cfg['priority'] is None:wait_priority(r,guard)
        else:wait_priority(r,guard,cfg['priority'])
        ledger['status']='RUNNING';write(r/'RESOURCE_LEDGER.json',ledger)
        sealed_file(work['static_plan_path'],work['static_plan_sha256']);static=read(work['static_plan_path'])
        rows=[];sources={};cases={}
        for source in work['closed_exports']:
            child=closed_child(parent,source['root'],'EXPORT','TERMINAL_EXPORT',profile=profile)
            result=child/'evidence/PAIRED_EXPORT_RESULT.private.json';sealed_file(result,source['result_sha256'])
            rows.extend(read(result)['rows'])
            for case in read(child/'plans/WORK_PLAN.private.json')['cases']:
                if case['id'] in cases:raise PermissionError('CR2_NO_DUPLICATE_CAPTURE_OR_EXPORT_PARENT')
                cases[case['id']]=case;sources[case['id']]=child
        if set(cases)!={row['id'] for row in rows}:raise PermissionError('CR2_EXACT_ALL_ATTEMPTS_AND_SOURCE_BINDINGS')
        if cfg.get('qualify_attempts') is not None:
            rows=cfg['qualify_attempts'](rows,cases,guard)
        value=manifest(static['groups'],rows,population_sha256=work['static_plan_sha256'],
            adoption_sha256=m['authority']['adoption_sha256'],profile=profile)
        if len(cases)>400:raise PermissionError('CR2_SAVED_PARENT_READ_LIMIT')
        guard()
        raw_upper=0
        for row in value['episodes']:
            case=cases[row['id']];raw=Path(case['source_episode'])
            closed_child(parent,raw.parent.parent,'CAPTURE','TERMINAL_CAPTURE',profile=profile)
            raw_file(raw/'FILE_MANIFEST.json',case['raw_manifest_sha256'])
            raw_upper+=sum(member['bytes'] for member in read(raw/'FILE_MANIFEST.json'))
        archive_bound=work['maximum_archive_bytes']
        if type(archive_bound) is not int or not 0<archive_bound<=cfg['archive_cap']:
            raise PermissionError('CR2_BOUNDED_ARCHIVES_AND_FULL_READBACK_TRANSFER_RESERVE')
        # Keep a conservative pre-write envelope, including copies, numeric
        # members, tar overhead and control metadata, not just periodic samples.
        if peak+2*raw_upper+archive_bound+1024**3>output_limit:
            raise PermissionError('CR2_PREWRITE_ALL_OUTPUT_AND_ARCHIVE_HEADROOM')
        out.mkdir();os.link(r/'plans/R1_statistics.json',out/'statistics.json')
        for row in value['episodes']:
            guard();identifier=row['id'];case=cases[identifier];raw=Path(case['source_episode'])
            closed_child(parent,raw.parent.parent,'CAPTURE','TERMINAL_CAPTURE',profile=profile)
            raw_file(raw/'FILE_MANIFEST.json',case['raw_manifest_sha256'])
            raw_file(raw/'AUDIT.json',case['audit_sha256'])
            source=sources[identifier]/'datasets/R1'/identifier
            receipt=source/'VERSION_READER_RECEIPT.json';sealed_file(receipt);saved=read(receipt)
            if (saved['id']!=identifier or saved['version']!='R1' or saved['statistics_sha256']!=R1
                    or saved['codec']!=next(x for x in rows if x['id']==identifier)['R1']):
                raise PermissionError('CR2_EXACT_ORIGINAL_R1_EXPORT_RECEIPT')
            row['sha256']=saved['numeric_sha256'];row['version_reader_receipt_sha256']=sha(receipt)
            numeric=source/'numeric'/(identifier+'.npz');sealed_file(numeric,row['sha256'])
            target=out/row['file'];target.parent.mkdir(exist_ok=True);os.link(numeric,target)
            raw_entries={entry['path']:entry for entry in read(raw/'FILE_MANIFEST.json')}
            for camera in ('body_rgb','wrist_rgb'):
                dest=out/'raw'/identifier/camera;dest.mkdir(parents=True)
                for i in range(641):
                    guard();name=camera+'/'+str(i).zfill(4)+'.png'
                    copy_raw(raw/name,dest/(str(i).zfill(4)+'.png'),raw_entries[name]['sha256'])
            parent_files(out,row,guard)
        public_json(value);public_json(rows)
        put(out/'DATASET_MANIFEST.json',value);put(out/cfg['attempts'],rows)
        processor=build_processor(inputs['official_source'],inputs['cosmos'],read(out/'statistics.json'))
        if serial(processor.state_action_processor.norm_params)!=read(r/'plans/FIXED_EFFECTIVE_PARAMS.private.json')['candidate_effective_params']:
            raise ValueError('CR2_BANK_ACTUAL_FIXED_R1_PROCESSOR_EFFECTIVE_PARAMS')
        mixed=qualify_all(out,value,processor,inputs,guard,profile=profile);put(out/'CROSS_PARENT_QUALIFICATION.json',mixed)
        members={p.relative_to(out).as_posix():p for p in out.rglob('*') if p.is_file()}
        entries=[dict(path=n,bytes=p.stat().st_size,sha256=sha(p)) for n,p in sorted(members.items())]
        put(out/'PAYLOAD_MANIFEST.json',entries);members['PAYLOAD_MANIFEST.json']=out/'PAYLOAD_MANIFEST.json'
        local=dict(schema=cfg['local_schema'],status=cfg['local'],statistics_sha256=R1,
            dataset_manifest_sha256=sha(out/'DATASET_MANIFEST.json'),payload_manifest_sha256=sha(out/'PAYLOAD_MANIFEST.json'),
            actual_cross_parent_sha256=sha(out/'CROSS_PARENT_QUALIFICATION.json'),parents=len(value['episodes']),
            HF_READY=False,later_rows_hot_injected=False)
        put(out/'LOCAL_FROZEN.json',local);members['LOCAL_FROZEN.json']=out/'LOCAL_FROZEN.json'
        for name in (cfg['attempts'],'DATASET_MANIFEST.json','CROSS_PARENT_QUALIFICATION.json','LOCAL_FROZEN.json'):
            public_json(read(out/name))
        bound=archive_bound
        upper=sum(p.stat().st_size for p in members.values())+len(members)*4096+1024**2
        if upper>bound:raise PermissionError('CR2_ARCHIVE_PAYLOAD_AND_HEADERS_EXCEED_BOUND')
        output=r/'delivery';output.mkdir();archives={}
        for i in range(0,len(value['episodes']),8):
            ids={x['id'] for x in value['episodes'][i:i+8]}
            subset={n:p for n,p in members.items() if any(n=='numeric/'+x+'.npz' or n.startswith('raw/'+x+'/') for x in ids)}
            name='parents-'+str(i//8).zfill(3)+'.tar.gz';archives[name]=pack_archive(output/name,subset,guard)
        archives['metadata.tar.gz']=pack_archive(output/'metadata.tar.gz',{n:p for n,p in members.items() if '/' not in n},guard)
        index=dict(schema=cfg['index_schema'],archives=archives,statistics_sha256=R1,
            dataset_manifest_sha256=local['dataset_manifest_sha256'],source_local_frozen_sha256=sha(out/'LOCAL_FROZEN.json'),
            source_bank=cfg['bank'],old_models_and_banks_reuploaded=False,READY_LAST_not_yet_published=True)
        if profile is None:index['new_CR2_only']=True
        public_json(index);put(output/'ARCHIVE_MEMBER_MANIFEST.json',index)
        size=sum(p.stat().st_size for p in output.iterdir())
        if size>bound:raise PermissionError('CR2_ACTUAL_INDEX_INCLUSIVE_ARCHIVE_BOUND')
        last_storage=0.;guard()
        result=dict(status=cfg['pack_status'],gate=value['gate'],parents=len(value['episodes']),
            archive_bytes=size,archive_index_sha256=sha(output/'ARCHIVE_MEMBER_MANIFEST.json'),
            dataset_manifest_sha256=local['dataset_manifest_sha256'],new_models=0,new_queries=0,new_physics=0,HF_READY=False)
        put(r/'evidence/PACK_RESULT.private.json',result)
        return result
    except BaseException as exc:failure=type(exc).__name__+': '+str(exc);raise
    finally:
        ledger.update(status='FAILED_PRESERVED' if failure else 'TERMINAL_PACK',CPU_s=time.process_time()-began,
            CPU_reserved_s=0,failure=failure,output_peak_bytes=peak)
        write(r/'RESOURCE_LEDGER.json',ledger);lock.close()
