"""New B0 weight/target-Processor control, separate from the original B0."""
import json,time
from pathlib import Path
import numpy as np
from .storage import digest

PROFILE='CORE_B0_FIXED_TARGET_PROCESSOR_REPLAY_V3_1'
PROCESSOR_PACKAGE='7feffa8dd11b5c9ce963f2e0f2f2613196c1a3ae8e5039056377bd5e25e02250'


def run(s,root,guard):
    from scripts.pi0292.durable import write
    from scripts.pi0294.numerics import prepare_environment,configure_torch,MODE
    from scripts.wm02.infer_fusion_paused_vla import _load_frozen,_generate_once
    from scripts.pi0294.e1.model import payload
    from scripts.pi0294.e1.journal import Journal
    from scripts.pi0294.e1.artifacts import atomic_npz
    from .target_gr00t_core_v1 import trim_unused_logits
    from .task_learning_native_dataset_v1 import TaskLearningNativeDataset
    from .task_learning_core_local_v1 import compare_saved_generations
    from .task_learning_local_analysis_v1 import geometry
    from scripts.pi0294.local_backbone import local_model_name
    from scripts.pi0294.neural import COSMOS_REVISION
    root=Path(root);started=time.monotonic();cpu=time.process_time();guard()
    if s['profile']!=PROFILE or s['max_queries']!=80 or s['processor_package']['sha256']!=PROCESSOR_PACKAGE:
        raise ValueError('Separate fixed-Processor B0 replay80 contract required')
    parent=Path(s['processor_package']['path']);manifest=json.loads((parent/'PACKAGE_MANIFEST.json').read_bytes())
    if digest(parent/'PACKAGE_MANIFEST.json')!=PROCESSOR_PACKAGE:raise ValueError('Original SFT2000 Processor package changed')
    for name,pin in manifest['files'].items():
        if name.startswith('processor/')and digest(parent/name)!=pin:raise ValueError('Saved target Processor member changed')
    selection=Path(s['Core40_selection']['path'])
    if digest(selection)!=s['Core40_selection']['sha256']:raise ValueError('Sealed original TRAIN40 selection changed')
    records=json.loads(selection.read_bytes())['records']
    if len(records)!=40:raise ValueError('Exactly40 existing TRAIN states and original seeds required')
    prepare_environment(MODE)
    import torch
    configure_torch(torch,MODE);torch.set_num_threads(4)
    # Load original CORR2048 B0 weights. No SFT or LAST2 tensor is installed.
    instance,torch=_load_frozen(s,root);trim_unused_logits(instance.model)
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
    instance.processor=Gr00tN1d7Processor.from_pretrained(parent/'processor',
        model_name=local_model_name(s['registered_root'],s['neural_bindings']),local_files_only=True,
        transformers_loading_kwargs=dict(local_files_only=True,trust_remote_code=True,
            cache_dir=s['neural_bindings']['cosmos_hub'],revision=COSMOS_REVISION))
    instance.processor.eval();instance.identity=dict(weights='ORIGINAL_CORR2048_B0_NO_SFT_INSTALL',processor_package=s['processor_package'],numeric_mode=MODE)
    write(root/'B0_TARGET_PROCESSOR_MODEL_LOADED.private.json',dict(identity=instance.identity,actual_model_loads=1,original_B0_identity_replaced=False))
    data=TaskLearningNativeDataset(s['dataset'],s['neural_bindings']['source'],s['neural_bindings']['cosmos']);data.processor=instance.processor
    if data.manifest_sha!='ff94fc77c5a8309dc1460032b308c2f213c96c088449ee1821b3e46e045107b0':raise ValueError('Original TRAIN100 required')
    lookup={(e['id'],a):i for i,(e,a)in enumerate(data.rows)};completed=0;requested=80
    for repeat in [1,2]:
        folder=root/f'replay-{repeat}';folder.mkdir();rows=[]
        for index,old in enumerate(records):
            guard();state,expert,images,text=data.raw(lookup[old['episode'],old['anchor']]);file=folder/f'prediction-{index:03d}.npz'
            journal=Journal(folder/f'query-{index:03d}-journal',f'replay{repeat}-query{index}')
            def generate():
                value,precision,timing=_generate_once(instance,payload(state,images,text),old['seed'],torch)
                if value['normalized'].shape!=(1,40,132)or value['physical_H40'].shape!=(40,10):raise ValueError('Original complete H40 required')
                if not all(np.isfinite(value[k]).all()for k in ['normalized','physical_H40']):raise ValueError('Nonfinite original output')
                atomic_npz(file,normalized=value['normalized'],physical_H40=value['physical_H40'],actual_state10=state,
                    expert_policy_H40=expert,noise_seed=np.asarray(old['seed'],dtype=np.int64))
                return dict(episode=old['episode'],anchor=old['anchor'],seed=old['seed'],sample_file=file.name,sha256=digest(file),
                    phase_audit_only=old['phase_audit_only'],precision=precision,noise=value['noise'],envelope=geometry(value['physical_H40'],state),**timing)
            result=journal.once('GENERATE',dict(episode=old['episode'],anchor=old['anchor'],seed=old['seed'],processor_package=s['processor_package']),generate)
            rows.append(result);completed+=1
            write(folder/'RESULT.private.json',dict(records=rows,actual_queries=len(rows),policy='B0_FIXED_TARGET_PROCESSOR_NOT_ORIGINAL_B0',original_full_inference=True))
            write(root/'PROGRESS.private.json',dict(actual_model_loads=1,completed_queries=completed,requested_queries=requested,updates=0,physics_integrations=0))
    exact=compare_saved_generations(root/'replay-1',root/'replay-2',root/'SAME_CARD_ACTUAL.private.json')
    result=dict(schema='CORE_B0_FIXED_TARGET_PROCESSOR_REPLAY80_ACTUAL_V3_1',status='PASS_ACTUAL_NEW_B0_TARGET_PROCESSOR_SAME_CARD40_TWICE',
        requested_queries=80,actual_queries=completed,actual_model_loads=1,matched_unique_states=40,all_bit_exact=exact['all_bit_exact'],
        checkpoint_manifest_sha256=s['checkpoint_manifest_sha256'],processor_package=s['processor_package'],GPU_uuid=s['GPU_uuid'],numeric_mode=MODE,
        original_B0_weights_unchanged=True,SFT_or_LAST2_weights_installed=False,original_B0_Processor_control_preserved=True,
        weights_control_identity='NEW_B0_FIXED_TARGET_PROCESSOR_NOT_OLD_B0_PROMOTION',
        original_envelope_pass=sum(r['envelope']['original_envelope_pass']for r in rows),original_envelope_requested=40,
        optimizer_updates=0,physics_integrations=0,original_DEV80_TEST=0,expert_is_model_input=False,
        native_execution_admission=False,original_cross_hardware_REJECT_preserved=True,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        wall_s=time.monotonic()-started,CPU_s=time.process_time()-cpu)
    write(root/'REPLAY80_ACTUAL.private.json',result);return result
