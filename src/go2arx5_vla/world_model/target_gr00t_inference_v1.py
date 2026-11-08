"""Four sealed real TRAIN queries for target Processor/weight diagnosis.

This worker has no native world, optimizer, task controller or submission code.
The original runtime, FUSION scope and neural lock still admit the GPU process.
Original B0 keeps its own Processor. This is the separately named target-
Processor shadow, or reconstruction of an actually trained production package.
"""
import json,os,time
from pathlib import Path
from .storage import digest,write_json
from . import target_gr00t_profile_v1 as profile

PROFILE='wm02-target-gr00t-sealed-inference-v1'
KIND='TARGET_GR00T_SEALED_INFERENCE_V1'
PARENT_SHADOW='PARENT_WITH_TARGET_PROCESSOR_SHADOW_FIT4'
PACKAGE='FORMAL_PACKAGE_RECONSTRUCTION_TRAIN4'
# Declared before either paired A40/Core result. This is a cross-hardware
# numerical gate, separate from the stricter same-A40 package equality gate.
CROSS_SITE_ATOL=dict(XYZ_m=1e-4,rot6d_component=1e-3,gripper=1e-3,normalized_component=1e-2)


def compare_cross_site(first,second):
    import numpy as np
    a,b=Path(first),Path(second)
    one=json.loads((a/'RESULT.json').read_bytes())['records']
    two=json.loads((b/'RESULT.json').read_bytes())['records']
    profile.need(len(one)==len(two)==4,'FOUR_MATCHED_A40_CORE_TRAIN_OBSERVATIONS')
    maximum={k:0. for k in CROSS_SITE_ATOL}
    for x,y in zip(one,two):
        profile.need(all(x[k]==y[k]for k in ('episode','split','anchor','seed','noise')),
                     'EXACT_CROSS_SITE_OBSERVATION_AND_NOISE')
        profile.need(x['model_identity']['dataset_sha256']==y['model_identity']['dataset_sha256']
            and x['model_identity']['statistics_sha256']==y['model_identity']['statistics_sha256']
            and x['model_global_step']==y['model_global_step'],'EXACT_CROSS_SITE_MODEL_DATA_STATISTICS')
        with np.load(a/(x['episode']+'.npz'),allow_pickle=False)as za, np.load(b/(y['episode']+'.npz'),allow_pickle=False)as zb:
            left,right=za['physical_H40'],zb['physical_H40']
            profile.need(left.shape==right.shape==(40,10) and np.isfinite(left).all()
                and np.isfinite(right).all(),'FINITE_CROSS_SITE_PHYSICAL_H40')
            delta=np.abs(left-right)
            values=dict(XYZ_m=float(delta[:,:3].max()),rot6d_component=float(delta[:,3:9].max()),
                gripper=float(delta[:,9].max()))
            left,right=za['normalized'],zb['normalized']
            profile.need(left.shape==right.shape and np.isfinite(left).all()
                and np.isfinite(right).all(),'FINITE_CROSS_SITE_NORMALIZED_H40')
            values['normalized_component']=float(np.abs(left-right).max())
            maximum={k:max(maximum[k],v)for k,v in values.items()}
    profile.need(all(maximum[k]<=v for k,v in CROSS_SITE_ATOL.items()),'A40_CORE_NUMERICAL_GATE_FAILED')
    return dict(status='PASS_ACTUAL_A40_CORE_MATCHED_INPUT_NOISE_NUMERICAL_GATE',
        absolute_component_tolerances=CROSS_SITE_ATOL,actual_maximum_absolute_error=maximum,
        actual_matched_queries=4,native_task_success='NOT_ESTABLISHED_BY_NUMERICAL_AGREEMENT')


def validate_spec(s):
    from scripts.pi0294.numerics import MODE
    need=profile.need
    need(s.get('profile')==PROFILE and s.get('kind')==KIND
        and s.get('execution_site')=='CORE_FUSION_NEURAL'and s.get('physics')is False
        and s.get('device')=='cuda'and s.get('max_queries')==4
        and s.get('GR00T_updates')==s.get('WM_updates')==s.get('Actor_updates')==0
        and s.get('numeric_mode')==MODE and s.get('baseline_changed')is False
        and s.get('checkpoint_manifest_sha256')==profile.PARENT,
        'FOUR_SEALED_QUERIES_NO_PHYSICS_NO_TRAINING_NO_BASELINE_CHANGE')
    mode=s.get('target_inference_mode');need(mode in(PARENT_SHADOW,PACKAGE),'EXPLICIT_TARGET_INFERENCE_MODE')
    root=Path(s['target_dataset']);manifest=json.loads((root/'DATASET_MANIFEST.json').read_bytes())
    admission=json.loads((root/'PROCESSOR_TRAINING_ADMISSION.json').read_bytes())
    need(digest(root/'DATASET_MANIFEST.json')==s['target_dataset_sha256']
        and digest(root/'statistics.json')==s['target_statistics_sha256']
        and admission.get('status')=='PASS_ACTUAL_OFFICIAL_PROCESSOR_ROUNDTRIP'
        and admission.get('manifest_sha256')==s['target_dataset_sha256']
        and admission.get('statistics_sha256')==s['target_statistics_sha256']
        and manifest.get('TRAIN_only_statistics')is True
        and manifest.get('TEST_results_read')is False
        and manifest.get('WM_or_HOLD_or_diagnostic_relabelled')is False,
        'EXACT_ADMITTED_TARGET_DATA_AND_TRAIN_ONLY_PROCESSOR')
    if mode==PARENT_SHADOW:
        need(manifest.get('bundle_kind')=='FIT4_DIAGNOSTIC'
            and len([r for r in manifest['episodes']if r['split']=='TRAIN'])==4
            and s.get('candidate_package')is None,
            'PARENT_TARGET_PROCESSOR_CONTROL_NOT_ORIGINAL_B0_OR_TRAINED_CANDIDATE')
    else:
        from .target_gr00t_delivery_v1 import verify
        descriptor=s['candidate_package'];m=verify(descriptor['path'],descriptor['sha256'],s['checkpoint_files'])
        proof=s['package_reconstruction_proof'];path=Path(proof['path'])
        need(digest(path)==proof['sha256'],'EXACT_ACTUAL_A40_RECONSTRUCTION_PROOF')
        q=json.loads(path.read_bytes())
        need(manifest.get('bundle_kind')=='FORMAL'and m['recipe']['dataset_sha256']==s['target_dataset_sha256']
            and m['recipe']['statistics_sha256']==s['target_statistics_sha256']
            and (q.get('package')or{}).get('sha256')==descriptor['sha256']
            and q.get('status')=='PASS_ACTUAL_INDEPENDENT_PARENT_DELTA_RECONSTRUCTION'
            and q.get('actual_matched_queries')==4 and q.get('absolute_component_tolerance')==1e-6
            and all(v<=1e-6 for v in q['max_abs_error'].values()),
            'ACTUAL_PRODUCTION_PACKAGE_AND_A40_PREDICTIONS_NOT_CPU_FIXTURE')
    return manifest


def validate_gpu(s,actual):
    from .fusion_scope import validate
    from .pilot1_scope import GPU_UUIDS
    from .pilot1_storage import local_path
    validate_spec(s)
    profile.need(s.get('GPU_uuid')in GPU_UUIDS and actual['uuid']==s['GPU_uuid']
        and os.environ.get('CUDA_VISIBLE_DEVICES')==s['GPU_uuid']
        and type(s.get('checkpoint_parameter_bytes'))is int and s['checkpoint_parameter_bytes']>0
        and s['checkpoint_parameter_bytes']<=s.get('allocator_limit_bytes',0)<actual['free_bytes']-1024**3,
        'ORIGINAL_SHARED_GPU_UUID_ALLOCATOR_AND_HEADROOM')
    validate(s)
    local_path(s,s['target_dataset']);local_path(s,s['checkpoint'])
    if s.get('candidate_package'):local_path(s,s['candidate_package']['path'])
    path=Path(s['calibration_result']);result=json.loads(path.read_bytes())
    lease=json.loads((path.parent/'control/LEASE.json').read_bytes())
    profile.need(digest(path)==s['calibration_result_sha256']and lease.get('status')=='CLOSED'
        and result.get('schema')=='WM02_FUSION_FIRST_ACTUAL_CALIBRATION_GROUP_V1'
        and result.get('exit_code')==0 and result.get('actual_method_evaluations')==4
        and result.get('actual_native_ticks')==5*4103,'ORIGINAL_FIRST_NATIVE_CALIBRATION_REQUIRED')


def infer(s,output):
    from scripts.pi0294.numerics import prepare_environment,configure_torch,MODE
    from scripts.pi0294.native_dataset import official_import
    validate_spec(s);root=Path(output);start=time.monotonic();cpu=time.process_time()
    for name,pin in s['checkpoint_files'].items():
        profile.need(digest(Path(s['checkpoint'])/name)==pin,'FROZEN_PARENT_FOR_TARGET_SHADOW:'+name)
    prepare_environment(MODE);official_import(s['neural_bindings']['source'])
    import torch
    from transformers import AutoModel
    import gr00t.model.gr00t_n1d7.gr00t_n1d7
    from scripts.pi0294.local_backbone import local_model_name
    from scripts.pi0294.neural import COSMOS_REVISION
    from .runtime import cuda_uuid
    profile.need(torch.cuda.device_count()==1 and cuda_uuid(torch.cuda.get_device_properties(0).uuid)==s['GPU_uuid'],
                 'ONE_ACTUAL_DECLARED_CORE_GPU')
    configure_torch(torch,MODE);torch.set_num_threads(4)
    free,total=torch.cuda.mem_get_info()
    profile.need(s['allocator_limit_bytes']<free-1024**3,'FRESH_SHARED_DEVICE_HEADROOM')
    torch.cuda.set_per_process_memory_fraction(s['allocator_limit_bytes']/total,0)
    torch.cuda.reset_peak_memory_stats()
    bindings=s['neural_bindings'];alias=local_model_name(s['registered_root'],bindings)
    kwargs=dict(local_files_only=True,trust_remote_code=True,cache_dir=bindings['cosmos_hub'],revision=COSMOS_REVISION)
    write_json(root/'MODEL_LOAD_INTENT.private.json',dict(parent_manifest_sha256=profile.PARENT,model_loads_requested=1))
    model,loading=AutoModel.from_pretrained(s['checkpoint'],local_files_only=True,trust_remote_code=True,
        output_loading_info=True,model_name=alias,transformers_loading_kwargs=kwargs,
        tune_llm=False,tune_visual=False,tune_projector=True,tune_diffusion_model=True,tune_vlln=True,
        state_dropout_prob=0.,backbone_trainable_params_fp32=True,load_bf16=False)
    write_json(root/'MODEL_CONSTRUCTED.private.json',dict(actual_model_loads=1,loading=loading))
    profile.need(not any(loading.get(k)for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
        and model.action_head.num_inference_timesteps==4,'EXACT_ORIGINAL_PARENT_FOUR_STEP_SAMPLER')
    from .target_gr00t_dataset_v1 import TargetNativeDataset
    data=TargetNativeDataset(s['target_dataset'],bindings['source'],bindings['cosmos'])
    data.initialize()
    if s['target_inference_mode']==PACKAGE:
        from .target_gr00t_delivery_v1 import install
        from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
        package=s['candidate_package'];install(model,package['path'],package['sha256'],s['checkpoint_files'])
        data.processor=Gr00tN1d7Processor.from_pretrained(Path(package['path'])/'processor',
            model_name=alias,local_files_only=True,transformers_loading_kwargs=kwargs)
        data.processor.eval()
    model.requires_grad_(False);model.to('cuda').eval()
    profile.need(data.processor.embodiment_id_mapping['new_embodiment']==10,'ORIGINAL_EMBODIMENT_MAPPING')
    parameter_bytes=sum(p.numel()*p.element_size()for p in model.parameters())
    profile.need(parameter_bytes==s['checkpoint_parameter_bytes'],'CORE_ORIGINAL_MODEL_STORAGE_NOT_SILENT_CAST')
    from scripts.pi0294.e1.model import E1ResidentModel
    from .target_gr00t_probes_v1 import measure
    instance=E1ResidentModel.__new__(E1ResidentModel)
    instance.root,instance.bindings,instance.model,instance.processor=root,bindings,model,data.processor
    instance.identity=dict(mode=s['target_inference_mode'],global_step=2000 if s.get('candidate_package')else 0,
        dataset_sha256=data.manifest_sha,statistics_sha256=data.statistics_sha,
        parent_manifest_sha256=profile.PARENT,candidate_package=s.get('candidate_package'))
    profile.need(time.time()<s['deadline_epoch']and not(Path(s['campaign_root'])/'STOP.json').exists(),
                 'FINITE_TARGET_SHADOW_BEFORE_QUERY')
    records=measure(root,'sealed-TRAIN4',instance,data)
    write_json(root/'RESULT.private.json',dict(status='ACTUAL_CORE_TARGET_SEALED_TRAIN4_INFERENCE',
        source_sha=s['source_sha'],target_inference_mode=s['target_inference_mode'],model_identity=instance.identity,
        GPU_uuid=s['GPU_uuid'],numeric_mode=MODE,
        actual_model_loads=1,actual_VLA_queries=len(records),actual_optimizer_updates=0,
        GR00T_updates=0,WM_updates=0,Actor_updates=0,native_integrations=0,
        Processor_decode_calls=len(records),original_baseline_changed=False,
        original_source_B0=False,
        target_native_execution=False,raw_H40_clipped=False,records=records,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu))
