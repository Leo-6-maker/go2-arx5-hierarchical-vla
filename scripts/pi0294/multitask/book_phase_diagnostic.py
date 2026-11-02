"""Fixed saved TRAIN32, same raw inputs and three declared noise replicates."""
import hashlib
from pathlib import Path

KIND='V7_BOOK_PHASE_MATCHED_TRAIN32_DIAGNOSTIC'


def validate_manifest(m,p):
    from collections import Counter
    if (m['kind']!=KIND or p['schema']!='BOOK_PHASE_MATCHED_TRAIN32_DIAGNOSTIC_V1'
            or p['noise_replicates']!=[0,1,2] or p['phase_metadata_student_input']is not False
            or p['labels_changed']is not False or p['physics_integrations']!=0 or p['optimizer_updates']!=0
            or len(p['windows'])!=32 or len({(r['id'],r['anchor'])for r in p['windows']})!=32
            or any(r['split']!='TRAIN'for r in p['windows'])
            or Counter(r['category']for r in p['windows'])!=dict(grasp_establishment=8,holding_carry=8,legal_release=8,actual_early_recovery=8)):
        raise PermissionError('FIXED_TRAIN32_PHASE_DIAGNOSTIC_ONLY')
    limits=dict(model_loads=1,optimizer_updates=0,physics_integrations=0,generation_queries=96,RGB_pairs=0,source_constructors=0)
    if any(m['limits'].get(k)!=v for k,v in limits.items()):raise PermissionError('FINITE_TRAIN32_NO_PHYSICS_OR_OPTIMIZATION')


def run(root,instance,guard):
    import numpy as np
    import torch
    from scripts.pi0292.durable import read,write
    from scripts.pi0294.native_dataset import NativeDataset,sha
    from scripts.pi0294.e1.model import payload
    from scripts.pi0294.e1.artifacts import atomic_npz
    from scripts.pi0294.e1.smoke import put
    from scripts.pi0292.eval_contract import error_metrics,gate_action
    root=Path(root);p=read(root/'TRAIN32_DIAGNOSTIC_PLAN.json');validate_manifest(read(root/'RUN_MANIFEST.json'),p)
    bindings=instance.bindings
    data=NativeDataset(bindings['diagnostic_dataset'],bindings['source'],bindings['cosmos'],for_optimization=False)
    if data.statistics_sha!=instance.identity['statistics_sha256']:raise PermissionError('MATCHED_ORIGINAL_BOOK_STATISTICS')
    lookup={(r['id'],a):(r,a)for r,a in data.rows}
    data.rows=[lookup[r['id'],r['anchor']]for r in p['windows']];data.initialize()
    if any(r['split']!='TRAIN'for r,a in data.rows):raise PermissionError('TRAIN_DIAGNOSTICS_ONLY')
    records=[];memory=[]
    capacity_request=read(root/'RUN_MANIFEST.json').get('GPU_inference_qualification')
    def measure_capacity(phase):
        if capacity_request is None:return
        from .book_gpu_capacity import INFERENCE_REQUEST,entry_requirement
        if capacity_request!=INFERENCE_REQUEST:raise PermissionError('EXACT_INFERENCE_CAPACITY_REQUEST')
        entry_requirement(read(root/'RUN_MANIFEST.json'))
        torch.cuda.synchronize();free,total=torch.cuda.mem_get_info()
        memory.append(dict(phase=phase,free_bytes=free,total_bytes=total,
            allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
            maximum_allocated_bytes=torch.cuda.max_memory_allocated(),maximum_reserved_bytes=torch.cuda.max_memory_reserved()))
        passed=min(r['free_bytes']for r in memory)>=capacity_request['minimum_headroom_bytes']
        write(root/'GPU_INFERENCE_CAPACITY.json',dict(measurements=memory,
            minimum_free_bytes=min(r['free_bytes']for r in memory),capacity_pass=passed,
            completed_generations=len(records),optimizer_updates=0,physics_integrations=0,
            training_admitted=False,free_is_shared_device_observation=True))
        if not passed:raise PermissionError('INFERENCE_REQUIRES_ONE_GIB_ACTUAL_HEADROOM')
    if capacity_request is not None:torch.cuda.reset_peak_memory_stats()
    measure_capacity('AFTER_ACTUAL_MODEL_AND_PARENT_RESTORE')
    for i,window in enumerate(p['windows']):
        state,expert,images,text=data.raw(i);request=payload(state,images,text)
        for replicate in p['noise_replicates']:
            guard();key=f'{i:02}-{replicate}'
            seed=int.from_bytes(hashlib.sha256(f"BOOK_PHASE_TRAIN32_V1/{window['id']}/{window['anchor']}/{replicate}".encode()).digest()[:8],'big')%(2**63)
            put(root/(key+'.INTENT.json'),dict(window=window,seed=seed,replicate=replicate))
            previous=instance.model.action_head.num_inference_timesteps
            try:
                instance.model.action_head.num_inference_timesteps=4
                value=instance.generate(request,seed)
            finally:instance.model.action_head.num_inference_timesteps=previous
            predicted=value['physical_H40'];path=root/(key+'.npz')
            atomic_npz(path,physical_H40=predicted,normalized=value['normalized'],actual_state10=state,actual_completed_expert_H40=expert)
            row=dict(window=window,replicate=replicate,seed=seed,arrays_sha256=sha(path),noise=value['noise'],
                image_SHA256=[r['sha256']for r in request['images']],errors=error_metrics(predicted,expert,state),
                original_gate=gate_action(predicted,state,0.,key),
                predicted_grip_prefix8=predicted[:8,9].tolist(),expert_grip_prefix8=expert[:8,9].tolist(),
                first_grip_signed_error=float(predicted[0,9]-expert[0,9]),
                prefix8_grip_signed_error_mean=float(np.mean(predicted[:8,9]-expert[:8,9])),
                first_grip_absolute_error=float(abs(predicted[0,9]-expert[0,9])))
            put(root/(key+'.RESULT.json'),row);records.append(row)
            measure_capacity('AFTER_GENERATION_'+str(len(records)))
            write(root/'DIAGNOSTIC_PROGRESS.json',dict(completed_generations=len(records),total=96))
    result=dict(status='COMPLETE_MATCHED_TRAIN32_PHASE_DIAGNOSTIC',records=records,generation_queries=96,
                TRAIN_windows=32,noise_replicates=3,optimizer_updates=0,physics_integrations=0,formal_DEV_conditions=0,
                low_gripper_prediction_is_not_physical_grasp_success=True)
    write(root/'TRAIN32_DIAGNOSTIC_RESULT.json',result)
    return result
