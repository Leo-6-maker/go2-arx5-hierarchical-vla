"""Resume only known-unstarted fixed draws through the unchanged E1 primitive."""
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.pi0292.durable import read, write
from scripts.pi0292.eval_contract import error_metrics, gate_action
from scripts.pi0292.relay_protocol import digest
from scripts.pi0294.e1 import model
from scripts.pi0294.e1.journal import Journal, MutationOutcomeUnknown
from scripts.pi0294.native_dataset import sha
from .probe_compare import PRECISION


def _plans(root):
    expert=read(root/'plans/PROBE_MANIFEST.json');students=read(root/'plans/STUDENT_PROBES.private.json')
    if (expert['TRAIN_only'] is not True or expert['count']!=len(expert['windows'])
            or expert['count']!=96 or len(students['states'])!=12):
        raise PermissionError('ORIGINAL_96_EXPERT_AND_12_STUDENT_TRAIN_PROBES_ONLY')
    ids=[r['id'] for r in expert['windows']+students['states']]
    if len(set(ids))!=108 or any(Path(i).name!=i or i in ['.','..'] for i in ids):
        raise PermissionError('EXACT_UNIQUE_FIXED_PROBE_IDENTITIES')
    if any(row['split']!='TRAIN' for row in expert['windows']):
        raise PermissionError('NO_DEV_TEACHER_PROBES')
    return expert,students


def status(root,label,checkpoint_files):
    root=Path(root);expert,students=_plans(root)
    folder=root/'evidence'/label/'probes';weights=digest(checkpoint_files)
    if folder.is_symlink():raise PermissionError('NO_FIXED_PROBE_DIRECTORY_SYMLINK')
    records={};aggregate_complete=True
    for name,expected in [('ON_EXPERT_RESULTS.json',expert['windows']),('ON_STUDENT_RESULTS.private.json',students['states'])]:
        path=folder/name
        if not path.exists():aggregate_complete=False;continue
        if path.is_symlink():raise PermissionError('NO_FIXED_PROBE_AGGREGATE_SYMLINK')
        value=read(path);rows=value['records'];by_id={r['id']:r for r in rows}
        if len(by_id)!=len(rows) or set(by_id)-{r['id'] for r in expected}:
            raise PermissionError('NO_EXTRA_OR_DUPLICATE_FIXED_PROBE_RESULTS')
        if name=='ON_EXPERT_RESULTS.json':
            if (value['status']!='ACTUAL_NATIVE_TRAIN_PROBES_NOT_TASK_SUCCESS'
                    or value['complete']!=(len(rows)==len(expected))):
                raise PermissionError('EXPLICIT_EXPERT_PROBE_COMPLETENESS_NOT_DIRECTORY_EXISTS')
        elif value['status']!='FIXED_12_TRAIN_OBSERVATIONS_ONLY_NOT_LIVE_OR_TEACHER_LABELS':
            raise PermissionError('ORIGINAL_STUDENT_PROBE_RESULT_KIND')
        aggregate_complete &= len(rows)==len(expected)
        records.update(by_id)
    journal=Journal(root/'reservations/probe-generations',root.name)
    missing=[];completed=[]
    expected_ids={r['id'] for r in expert['windows']+students['states']}
    if folder.exists() and {p.stem for p in folder.glob('*.npz')}-expected_ids:
        raise PermissionError('UNDECLARED_FIXED_PROBE_ARRAYS')
    for row in expert['windows']+students['states']:
        identifier=row['id'];path=folder/(identifier+'.npz');message='probe-'+label+'-'+identifier
        receipt=journal.path(message)
        if not receipt.exists():
            if path.exists() or identifier in records:
                raise MutationOutcomeUnknown('FIXED_DRAW_ARRAY_OR_RESULT_WITHOUT_DURABLE_INTENT')
            missing.append(identifier);continue
        item=read(receipt)
        if item['status']!='COMPLETE':
            raise MutationOutcomeUnknown('FIXED_DRAW_INTENT_UNKNOWN_NO_RESAMPLING')
        if path.is_symlink() or not path.is_file():
            raise ValueError('COMPLETED_FIXED_DRAW_ARRAY_MISSING_NO_RESAMPLING')
        with np.load(path,allow_pickle=False) as saved:
            prediction=saved['physical_H40'];noise=json.loads(saved['noise_fingerprint_json'].item())
            precision=json.loads(saved['inference_precision_json'].item());observation=saved['observation_fingerprint'].item()
            if (prediction.shape!=(40,10) or not np.isfinite(prediction).all()
                    or saved['checkpoint_files_fingerprint'].item()!=weights
                    or int(saved['noise_seed'].item())!=row['noise_seed']
                    or any(precision.get(k)!=v for k,v in PRECISION.items())):
                raise ValueError('SAVED_FIXED_PROBE_WEIGHT_SEED_PRECISION_OR_ARRAY_CHANGED')
            if 'episode_id' in row:
                if (hashlib.sha256(saved['state'].tobytes()).hexdigest()!=row['state_sha256']
                        or hashlib.sha256(saved['expert_absolute10'].tobytes()).hexdigest()!=row['absolute_H40_sha256']):
                    raise ValueError('FIXED_EXPERT_LABEL_AND_STATE_IDENTITY')
            elif observation!=digest({k:v for k,v in row['payload'].items() if k!='snapshot_uuid'}):
                raise ValueError('FIXED_STUDENT_INPUT_IDENTITY')
            record=records.get(identifier)
            if record is not None:
                if (record['noise']!=noise or record['inference_precision']!=precision
                        or record.get('seed',record.get('noise_seed'))!=row['noise_seed']):
                    raise ValueError('FIXED_PROBE_AGGREGATE_METADATA_CHANGED')
                if 'episode_id' in row:
                    if (any(record[k]!=row[k] for k in ['episode_id','task','anchor','split'])
                            or record['errors']!=error_metrics(prediction,saved['expert_absolute10'],saved['state'])
                            or record['gate']!=gate_action(prediction,saved['state'],row['anchor']*.05,identifier)):
                        raise ValueError('FIXED_EXPERT_RESULT_CONTEXT_OR_METRICS_CHANGED')
                elif (record['new_query'] is not True or record['physical_H40']!=prediction.tolist()
                        or record['gate']!=gate_action(prediction,row['payload']['state_eef10'],row['tick']*.002,identifier)):
                    raise ValueError('FIXED_STUDENT_RESULT_CHANGED')
        args=dict(label=label,id=identifier,seed=row['noise_seed'],observation_fingerprint=observation,
            checkpoint_files_fingerprint=weights,precision='ORIGINAL_E0_BF16_AUTOCAST_FP32_MATMUL_HIGHEST')
        meta=journal.completed(message,args)
        if (sha(path)!=meta['arrays_sha256'] or meta['noise']!=noise or meta['inference_precision']!=precision
                or (identifier in records and records[identifier]['arrays_sha256']!=meta['arrays_sha256'])):
            raise ValueError('SAVED_FIXED_DRAW_JOURNAL_ARRAY_OR_AGGREGATE_CHANGED')
        completed.append(identifier)
    return dict(complete=aggregate_complete and len(completed)==108,known_unstarted_ids=missing,
        completed_draw_ids=completed,unknown_effects=0,new_queries_required=len(missing),
        aggregate_rebuild_required=not aggregate_complete,checkpoint_files_fingerprint=weights)


def complete(root,label,instance,dataset,ledger,guard):
    root=Path(root);before=status(root,label,instance.identity['files'])
    if before['complete']:return dict(on_expert=96,on_student=12,new_queries=0,reused_draws=108)
    expert,students=_plans(root);folder=root/'evidence'/label/'probes';folder.mkdir(parents=True,exist_ok=True)
    indices={(r['id'],anchor):i for i,(r,anchor) in enumerate(dataset.rows)}
    records=[]
    for row in expert['windows']:
        guard();state,target,images,text=dataset.raw(indices[row['episode_id'],row['anchor']])
        request=model.payload(state,images,text)
        physical,meta=model.fixed_draw(root,label,row['id'],instance,request,row['noise_seed'],
            folder/(row['id']+'.npz'),ledger,expert=target,state=state)
        records.append(dict(id=row['id'],task=row['task'],episode_id=row['episode_id'],anchor=row['anchor'],split='TRAIN',
            errors=error_metrics(physical,target,state),gate=gate_action(physical,state,row['anchor']*.05,row['id']),
            native_expert_roundtrip_maximum_absolute_difference=meta['native_expert_roundtrip_maximum_absolute_difference'],
            expert_clip=meta['expert_clip'],noise=meta['noise'],seed=row['noise_seed'],
            inference_precision=meta['inference_precision'],arrays_sha256=meta['arrays_sha256']))
        write(folder/'ON_EXPERT_RESULTS.json',dict(status='ACTUAL_NATIVE_TRAIN_PROBES_NOT_TASK_SUCCESS',records=records,complete=len(records)==96))
    student_records=[]
    for row in students['states']:
        guard();physical,meta=model.fixed_draw(root,label,row['id'],instance,row['payload'],row['noise_seed'],folder/(row['id']+'.npz'),ledger)
        student_records.append(dict(id=row['id'],new_query=True,noise=meta['noise'],noise_seed=row['noise_seed'],physical_H40=physical.tolist(),
            inference_precision=meta['inference_precision'],arrays_sha256=meta['arrays_sha256'],
            gate=gate_action(physical,row['payload']['state_eef10'],row['tick']*.002,row['id'])))
    write(folder/'ON_STUDENT_RESULTS.private.json',dict(status='FIXED_12_TRAIN_OBSERVATIONS_ONLY_NOT_LIVE_OR_TEACHER_LABELS',records=student_records))
    after=status(root,label,instance.identity['files'])
    if not after['complete']:raise PermissionError('FIXED_PROBES_NOT_COMPLETE_AFTER_BOUNDED_RESUME')
    return dict(on_expert=96,on_student=12,new_queries=before['new_queries_required'],reused_draws=len(before['completed_draw_ids']))


def panel_work(root,label,checkpoint_files,prior_rows,spec):
    from .plans import remaining_panel
    remaining=remaining_panel(prior_rows,spec)
    fixed=status(root,label,checkpoint_files) if label.endswith('TRAIN4') else None
    return dict(remaining=remaining,probes=fixed,
        skip_before_model_load=not remaining and (fixed is None or fixed['complete']))
