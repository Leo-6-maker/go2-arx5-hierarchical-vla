"""Actual original-Processor target round trip; no policy model or optimizer."""
import argparse,json,time
from pathlib import Path
import numpy as np

# Declared before target measurements, for the existing FP32 EEF representation.
FP32_COMPONENT_ATOL = 1e-6


def qualify(root,source,cosmos):
    from go2arx5_vla.world_model.target_gr00t_dataset_v1 import TargetNativeDataset,need
    from go2arx5_vla.world_model.storage import digest,write_json
    from scripts.pi0292.train_adapter import OFFICIAL_SHA
    from scripts.pi0292.eval_contract import expert_decode
    root=Path(root);path=root/'PROCESSOR_TRAINING_ADMISSION.json'
    need(not path.exists(),'PRESERVE_EXISTING_PROCESSOR_ADMISSION')
    start=time.monotonic();cpu=time.process_time()
    data=TargetNativeDataset(root,source,cosmos)
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
            RGB_pairs=len(images)//2,language_matches=bool(text),decode_calls=1))
    processor=root/'target-processor-v1'
    data.processor.save_pretrained(processor)
    files={str(p.relative_to(processor)):digest(p)for p in processor.rglob('*')if p.is_file()}
    result=dict(schema='WM02_TARGET_GR00T_PROCESSOR_ADMISSION_V1',
        status='PASS_ACTUAL_OFFICIAL_PROCESSOR_ROUNDTRIP',manifest_sha256=data.manifest_sha,
        statistics_sha256=data.statistics_sha,upstream_sha=OFFICIAL_SHA,TRAIN_only_statistics=True,
        processed_TRAIN_windows=len(records),maximum_abs_error=max(r['max_abs_error']for r in records),
        fixed_component_tolerance=FP32_COMPONENT_ATOL,records=records,
        processor_files=files,processor_directory=str(processor),
        train_and_deploy_processor='SAME_SAVED_TARGET_PROCESSOR_V1',
        new_policy_models=0,new_optimizer_updates=0,new_physics=0,new_PBS=0,
        wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu)
    write_json(path,result);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','source','cosmos'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(argv);r=qualify(a.dataset,a.source,a.cosmos)
    print(json.dumps({k:v for k,v in r.items()if k!='records'}));return 0

if __name__=='__main__':raise SystemExit(main())
