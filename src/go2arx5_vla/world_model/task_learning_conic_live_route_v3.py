"""Two preregistered new-live-loader parity queries, never physical commands."""
import json,time
from pathlib import Path
import numpy as np
from .storage import digest


def run(s,root,guard):
    from scripts.pi0292.durable import write
    from scripts.pi0294.numerics import prepare_environment,configure_torch,MODE
    from scripts.wm02.infer_fusion_paused_vla import _load_frozen,_generate_once
    from scripts.pi0294.e1.model import payload
    from .task_learning_conic_policy_v3 import install_ordinary_model_delta,action_packet
    from .target_gr00t_core_v1 import trim_unused_logits
    from .task_learning_native_dataset_v1 import TaskLearningNativeDataset
    from scripts.pi0294.local_backbone import local_model_name
    from scripts.pi0294.neural import COSMOS_REVISION
    root=Path(root);started=time.monotonic();cpu=time.process_time();guard()
    pin=s['sealed_C40'];file=Path(pin['path'])
    if digest(file)!=pin['sha256']:raise ValueError('Exact sealed C500 ordinary40 result required')
    records=json.loads(file.read_bytes())['records']
    if len(records)!=40 or [records[i]['phase_audit_only']for i in(0,3)]!=['PREGRASP','LIFT']:
        raise ValueError('Fixed existing C40 indices0and3 required')
    prepare_environment(MODE)
    import torch
    configure_torch(torch,MODE);torch.set_num_threads(4)
    instance,torch=_load_frozen(s,root);trim_unused_logits(instance.model)
    custody=install_ordinary_model_delta(instance.model,s['C500_package'],s['checkpoint_files'])
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
    instance.processor=Gr00tN1d7Processor.from_pretrained(Path(custody['parent_package']['path'])/'processor',
        model_name=local_model_name(s['registered_root'],s['neural_bindings']),local_files_only=True,
        transformers_loading_kwargs=dict(local_files_only=True,trust_remote_code=True,
            cache_dir=s['neural_bindings']['cosmos_hub'],revision=COSMOS_REVISION))
    instance.processor.eval();instance.identity=dict(package=s['C500_package'],parent_package=custody['parent_package'],numeric_mode=MODE)
    data=TaskLearningNativeDataset(s['dataset'],s['neural_bindings']['source'],s['neural_bindings']['cosmos'])
    data.processor=instance.processor;lookup={(r['id'],a):i for i,(r,a)in enumerate(data.rows)}
    rows=[]
    for index in(0,3):
        guard();old=records[index];original=file.parent/old['sample_file']
        if digest(original)!=old['sha256']:raise ValueError('Sealed ordinary prediction changed')
        state,_,images,text=data.raw(lookup[old['episode'],old['anchor']])
        write(root/f'QUERY_{index}_INTENT.private.json',dict(seed=old['seed'],episode=old['episode'],anchor=old['anchor'],epoch=time.time()))
        value,precision,timing=_generate_once(instance,payload(state,images,text),old['seed'],torch)
        packet=action_packet(value,'C500_PLUS_CONIC');target=root/f'QUERY_{index}_ACTUAL.private.npz';np.savez(target,**packet)
        with np.load(original,allow_pickle=False)as z:
            exact={key:np.array_equal(value[key],z[key])for key in('normalized','physical_H40')}
            maximum={key:float(np.max(np.abs(value[key].astype(float)-z[key].astype(float))))for key in exact}
        replay_index=index*3;replay=Path(s['Core_conic_root'])/'replay-1'/f'C-{replay_index:03d}.private.npz'
        with np.load(replay,allow_pickle=False)as z:
            fitted_exact='explicit_conic_H40'in packet and np.array_equal(packet['explicit_conic_H40'],z['decoded'])
        row=dict(index=index,phase=old['phase_audit_only'],episode=old['episode'],anchor=old['anchor'],seed=old['seed'],
            original_prediction_sha256=old['sha256'],live_prediction_sha256=digest(target),raw_exact=exact,
            maximum_absolute_error=maximum,conic_exact_to_Core_CPU=fitted_exact,precision=precision,noise=value['noise'],**timing)
        rows.append(row);write(root/f'QUERY_{index}_RESULT.private.json',row)
        write(root/'PROGRESS.private.json',dict(completed_queries=len(rows),requested_queries=2,model_loads=1,optimizer_updates=0,physics_integrations=0))
    result=dict(schema='CORE_C500_NEW_LIVE_LOADER_PARITY2_ACTUAL_V3',requested_queries=2,actual_queries=2,actual_model_loads=1,
        all_raw_bit_exact=all(all(r['raw_exact'].values())for r in rows),all_conic_bit_exact=all(r['conic_exact_to_Core_CPU']for r in rows),
        rows=rows,C500_package=s['C500_package'],model_custody=custody,source_sha=s['source_sha'],
        fit_calls=2,solver_calls=4,optimizer_updates=0,physics_integrations=0,native_execution_admission=False,
        original_cross_hardware_REJECT_preserved=True,original_DEV80_TEST=0,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        wall_s=time.monotonic()-started,CPU_s=time.process_time()-cpu)
    write(root/'PARITY_ACTUAL.private.json',result)
    if not(result['all_raw_bit_exact']and result['all_conic_bit_exact']):raise ValueError('NEW_LIVE_LOADER_PARITY_FAILED_PRESERVE_BOTH_QUERIES_NO_PROMOTION')
    return result
