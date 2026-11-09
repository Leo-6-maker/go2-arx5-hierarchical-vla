"""Forty fixed real TRAIN probes for a delivered continuous-task SFT package."""
import ast,hashlib,json
from pathlib import Path
from . import target_gr00t_profile_v1 as original
from .stable_task_scene_v1 import pinned
from .storage import digest

PROFILE='wm02-task-learning-package-core-v1'
KIND='TASK_LEARNING_PACKAGE_CORE_RECONSTRUCTION_V1'
MODE='CONTINUOUS_TASK_PACKAGE_RECONSTRUCTION_TRAIN40'
MEMORY='CORE_ORDINARY_FULL_CATEGORY_GPU_UNUSED_LOGITS1_FP32_V1'
MEMORY_IMPLEMENTATION='6df341b03ac76c147d7d3aa19b0281546b4a9087bc7ef7e3e7bc612257a3f572'


def memory_implementation():
    from . import target_gr00t_core_v1 as core
    source=Path(core.__file__).read_text()
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='trim_unused_logits')
    return hashlib.sha256(ast.get_source_segment(source,node).encode()).hexdigest()


def validate_memory(s):
    """Core05's CPU-retained category tables do not admit full GPU tables."""
    memory=pinned(s['memory_execution_reference']);proof=memory.get('proof')or{}
    need=original.need
    need(s.get('memory_execution')==MEMORY and memory_implementation()==MEMORY_IMPLEMENTATION
        and memory.get('schema')=='GR00T_CORE_ORDINARY_FROZEN_PARENT_PLUS_DELTA_ACTUAL_V1'
        and len(memory.get('source_sha',''))==40
        and memory.get('worker_exit_code')==0 and memory.get('lease_status')=='CLOSED'
        and memory.get('all_owned_children_terminal')is True
        and proof.get('status')=='PASS_ACTUAL_ORDINARY_FULL_CATEGORY_GPU_FIT200_RECONSTRUCTION'
        and proof.get('ordinary_category_tables_on_GPU')is True
        and proof.get('parameter_bytes')==s.get('checkpoint_parameter_bytes')==12576064000
        and proof.get('all_parameter_dtypes')=='torch.float32'
        and proof.get('full_original_backbone_generations')==4
        and proof.get('all132_H40_outputs_bit_exact')is True
        and proof.get('physical_outputs_bit_exact')is True and proof.get('precision_relaxed')is False
        and 0<proof.get('peak_allocated_bytes',0)<=s.get('allocator_limit_bytes',0),
        'ACTUAL_ORDINARY_FULL_CATEGORY_GPU_MEMORY_EXECUTION_NOT_CPU_TABLE_QUALIFICATION')
    return memory


def validate_spec(s):
    from scripts.pi0294.numerics import MODE as numeric
    from .task_learning_native_dataset_v1 import TaskLearningNativeDataset
    from .target_gr00t_delivery_v1 import verify
    need=original.need
    need(s.get('profile')==PROFILE and s.get('kind')==KIND
        and s.get('target_inference_mode')==MODE and s.get('max_queries')==40
        and s.get('execution_site')=='CORE_FUSION_NEURAL'and s.get('device')=='cuda'
        and s.get('physics')is False and s.get('baseline_changed')is False
        and s.get('GR00T_updates')==s.get('WM_updates')==s.get('Actor_updates')==0
        and s.get('numeric_mode')==numeric and s.get('checkpoint_manifest_sha256')==original.PARENT,
        'FIXED_CONTINUOUS_PACKAGE_PROBES_WITHOUT_NATIVE_WORLD_OR_UPDATES')
    data=TaskLearningNativeDataset(s['target_dataset'],s['neural_bindings']['source'],s['neural_bindings']['cosmos'])
    root=Path(s['target_dataset']);admission=json.loads((root/'PROCESSOR_TRAINING_ADMISSION.json').read_text())
    need(data.manifest['bundle_kind']=='FORMAL_TASK_LEARNING_V1'
        and data.manifest_sha==s['target_dataset_sha256']and data.statistics_sha==s['target_statistics_sha256']
        and admission.get('status')=='PASS_ACTUAL_OFFICIAL_PROCESSOR_ROUNDTRIP'
        and admission.get('dataset_family')=='TASK_LEARNING_CONTINUOUS_V1'
        and admission.get('manifest_sha256')==data.manifest_sha
        and admission.get('statistics_sha256')==data.statistics_sha,
        'ACTUAL_NEW_CONTINUOUS_DATA_AND_ORIGINAL_CORE_PROCESSOR')
    package=s['candidate_package'];m=verify(package['path'],package['sha256'],s['checkpoint_files'])
    q=pinned(s['package_reconstruction_proof'])
    need(m['profile']=='TASK_LEARNING_GR00T_V1'and m.get('training_site')=='HOPPER_SINGLE_GPU'
        and m['recipe']['dataset_sha256']==data.manifest_sha
        and m['recipe']['statistics_sha256']==data.statistics_sha
        and q.get('status')=='PASS_ACTUAL_INDEPENDENT_PARENT_DELTA_RECONSTRUCTION'
        and(q.get('package')or{}).get('sha256')==package['sha256']
        and q.get('actual_matched_queries')==40 and q.get('absolute_component_tolerance')==1e-6
        and all(v<=1e-6 for v in q['max_abs_error'].values()),
        'ACTUAL_HOPPER_ORDINARY_PARENT_DELTA_AND_FORTY_MATCHED_PHASE_PROBES')
    validate_memory(s)
    return data.manifest
