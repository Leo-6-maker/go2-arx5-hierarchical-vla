"""Actual original-Processor target round trip; no policy model or optimizer."""
import argparse,json,time
from pathlib import Path
import numpy as np

# Declared before target measurements, for the existing FP32 EEF representation.
FP32_COMPONENT_ATOL = 1e-6


def dataset_type(family):
    if family=='TARGET_SHORT_V1':
        from go2arx5_vla.world_model.target_gr00t_dataset_v1 import TargetNativeDataset
        return TargetNativeDataset
    if family=='TASK_LEARNING_CONTINUOUS_V1':
        from go2arx5_vla.world_model.task_learning_native_dataset_v1 import TaskLearningNativeDataset
        return TaskLearningNativeDataset
    raise ValueError('EXPLICIT_ORIGINAL_SHORT_OR_NEW_CONTINUOUS_DATASET_FAMILY')


def qualify(root,source,cosmos,*,verify_transfer=False,transfer_site='VANDA_A40',
            dataset_family='TARGET_SHORT_V1'):
    from go2arx5_vla.world_model.target_gr00t_dataset_v1 import TargetNativeDataset,need
    from go2arx5_vla.world_model.storage import digest,write_json
    from scripts.pi0292.train_adapter import OFFICIAL_SHA
    from scripts.pi0292.eval_contract import expert_decode
    from scripts.pi0294.native_train import tree_identity
    root=Path(root);original=root/'PROCESSOR_TRAINING_ADMISSION.json'
    need(transfer_site in('VANDA_A40','HOPPER_SINGLE_GPU'),'EXPLICIT_ACTUAL_TRANSFER_SITE')
    transferred='hopper'if transfer_site=='HOPPER_SINGLE_GPU'else'vanda'
    expected=json.loads(original.read_bytes()) if verify_transfer else None
    path=root/('PROCESSOR_'+transferred.upper()+'_RECONSTRUCTION.json' if verify_transfer else original.name)
    need(not path.exists(),'PRESERVE_EXISTING_PROCESSOR_ADMISSION')
    start=time.monotonic();cpu=time.process_time()
    data=dataset_type(dataset_family)(root,source,cosmos)
    data.initialize()
    records=[]
    # Every kept TRAIN anchor, not sampled adjacent frames mixed across splits.
    for index,(row,anchor)in enumerate(data.rows):
        state,commands,images,text=data.raw(index)
        decoded,meta=expert_decode(data.processor,state,commands)
        error=float(np.max(np.abs(decoded-commands)))
        need(np.isfinite(error)and error<=FP32_COMPONENT_ATOL,'OFFICIAL_FP32_ROUNDTRIP_AT_FIXED_TOLERANCE')
        processed=data[index]
        batch=data.processor.collator([processed['processed']])
        action=batch['inputs']['action'];mask=batch['inputs']['action_mask']
        need(tuple(action.shape)==(1,40,132)and tuple(mask.shape)==(1,40,132)
             and bool((mask[...,:10]==1).all())and bool((mask[...,10:]==0).all()),
             'ORIGINAL_H40_EEF10_MASK_AND_TIME_SEMANTICS')
        records.append(dict(episode=row['id'],anchor=anchor,max_abs_error=error,
            RGB_pairs=len(images)//2,language_matches=bool(text),decode_calls=1,
            actual_collated_batch_identity=tree_identity(batch)))
    if verify_transfer:
        need(expected['manifest_sha256']==data.manifest_sha and expected['statistics_sha256']==data.statistics_sha
             and expected['processed_TRAIN_windows']==len(records),'SAME_EXACT_TRANSFERRED_PROCESSOR_INPUTS')
        need(expected.get('dataset_family','TARGET_SHORT_V1')==dataset_family,
             'SAME_EXPLICIT_DATASET_FAMILY_ACROSS_SITES')
        for a,b in zip(expected['records'],records):
            need(a['episode']==b['episode'] and a['anchor']==b['anchor']
                 and a['actual_collated_batch_identity']==b['actual_collated_batch_identity'],
                 'ACTUAL_CROSS_SITE_IMAGE_TEXT_STATE_ACTION_MASK_BATCH_IDENTITY')
    processor=root/('target-processor-'+transferred+'-v1'if verify_transfer else'target-processor-v1')
    data.processor.save_pretrained(processor)
    files={str(p.relative_to(processor)):digest(p)for p in processor.rglob('*')if p.is_file()}
    result=dict(schema='WM02_TARGET_GR00T_PROCESSOR_ADMISSION_V1',
        status='PASS_ACTUAL_CROSS_SITE_TARGET_PROCESSOR_RECONSTRUCTION'if verify_transfer else'PASS_ACTUAL_OFFICIAL_PROCESSOR_ROUNDTRIP',manifest_sha256=data.manifest_sha,
        statistics_sha256=data.statistics_sha,dataset_family=dataset_family,
        upstream_sha=OFFICIAL_SHA,TRAIN_only_statistics=True,
        processed_TRAIN_windows=len(records),maximum_abs_error=max(r['max_abs_error']for r in records),
        fixed_component_tolerance=FP32_COMPONENT_ATOL,records=records,
        processor_files=files,processor_directory=str(processor),
        train_and_deploy_processor='SAME_SAVED_TARGET_PROCESSOR_V1',
        original_Core_admission_sha256=digest(original)if verify_transfer else None,
        actual_cross_site_processed_batch_identity=verify_transfer,
        transfer_site=transfer_site if verify_transfer else None,
        new_policy_models=0,new_optimizer_updates=0,new_physics=0,new_PBS=0,
        wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu)
    write_json(path,result);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','source','cosmos'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--verify-transfer',action='store_true')
    p.add_argument('--transfer-site',choices=('VANDA_A40','HOPPER_SINGLE_GPU'),default='VANDA_A40')
    p.add_argument('--dataset-family',choices=('TARGET_SHORT_V1','TASK_LEARNING_CONTINUOUS_V1'),
                   default='TARGET_SHORT_V1')
    a=p.parse_args(argv);r=qualify(a.dataset,a.source,a.cosmos,verify_transfer=a.verify_transfer,
        transfer_site=a.transfer_site,dataset_family=a.dataset_family)
    print(json.dumps({k:v for k,v in r.items()if k!='records'}));return 0

if __name__=='__main__':raise SystemExit(main())
