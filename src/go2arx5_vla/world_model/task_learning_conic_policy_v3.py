"""Explicit composed C500 policy utilities; native admission is separate.

The neural process retains raw GR00T output and runs the declared CPU conic
solver. The physics process needs only its existing NumPy/control environment;
it verifies the packet and original full H40 gates without importing SciPy,
Clarabel or Torch. These utilities neither start workers nor grant execution.
"""
import json
from pathlib import Path
import numpy as np
from .storage import digest
from . import task_learning_grasp_v1 as task
from .task_learning_conic_qualification_v3 import identity

PROTOCOL='CORE_LOCAL_C500_EXPLICIT_CONIC_DIAGNOSTIC_NOT_DEV_TEST_V3'
EXECUTION_VERSION='H40_ENDPOINT_FIXED_CONIC_COMPOSED_EXECUTION_V3_01'
SOLVER_METHOD='H40_ENDPOINT_FIXED_CONIC_L2_OFFLINE_V3_01'
METHODS=('C500_RAW','C500_PLUS_CONIC')
C500_PACKAGE_SHA256='0b248af6ffe503c778c43e5f68c03eb9be6266b4c493ff82a9e1668dead8a8d3'
SFT2000_PACKAGE_SHA256='7feffa8dd11b5c9ce963f2e0f2f2613196c1a3ae8e5039056377bd5e25e02250'


def verify_C500_package(package,checkpoint_files):
    """Reuse the real completed40 reconstruction; do not rerun its model calls."""
    from .stable_task_scene_v1 import pinned
    from .target_gr00t_delivery_v1 import verify as verify_parent
    from .task_learning_conic_qualification_v3 import DATA,STATS
    root=Path(package['path']);manifest_path=root/'PACKAGE_MANIFEST.json'
    task.need(root.resolve()==root and package['sha256']==C500_PACKAGE_SHA256
        and digest(manifest_path)==C500_PACKAGE_SHA256,'CONIC_EXACT_C500_PACKAGE')
    m=json.loads(manifest_path.read_bytes())
    task.need(m['schema']=='TASK_LEARNING_CORE_LOCAL_LAST2_PACKAGE_V1'
        and m['parent_global_step']==2000 and m['adaptation_step']==500
        and m['parent_package']['sha256']==SFT2000_PACKAGE_SHA256
        and m['dataset_sha256']==DATA and m['statistics_sha256']==STATS
        and m['standalone_without_parent_package']is False and m['cross_hardware_qualification']is False
        and m['old_FIT4_identity_reused']is False,'CONIC_NEW_C500_NOT_FULL_SFT2500_OR_FIT4')
    for name,pin in m['files'].items():
        file=root/name
        task.need(not Path(name).is_absolute()and '..'not in Path(name).parts
            and file.resolve().is_relative_to(root)and digest(file)==pin,'CONIC_C500_MEMBER:'+name)
    contract=json.loads((root/'PARAMETER_CONTRACT.json').read_bytes())
    task.need(contract['variant']=='INTERFACES_LAST2_DIT'and contract['trainable_numel']==77847684
        and contract['target_category']==10,'CONIC_EXACT_INTERNAL_LAST2_PARAMETERS')
    proof=pinned(m['reconstruction']);producer=Path(m['reconstruction']['path']).parent
    spec_path=producer/'EXECUTION_SPEC.json';s=json.loads(spec_path.read_bytes())
    lease=json.loads((producer/'control/LEASE.json').read_bytes());worker=json.loads((producer/'WORKER_EXIT.private.json').read_bytes())
    task.need(lease['status']=='CLOSED'and lease['worker_exit_code']==worker['exit_code']==0
        and lease['spec_sha256']==digest(spec_path)and worker.get('failure')is None
        and s['operation']=='REBUILD'and s['package']==m['parent_package']
        and proof['schema']=='TASK_LEARNING_CORE_LOCAL_LAST2_ORDINARY_RECONSTRUCTION_V1'
        and proof['actual_full_original_generations']==proof['matched_cached_generations']==len(proof['records'])==40
        and proof['all_H40_and_physical_bit_exact']is True
        and all(all(a['exact'].values())for a in proof['records'])
        and proof['parent_package']==m['parent_package']and proof['trained_checkpoint']==s['trained_checkpoint']
        and proof['cross_hardware_qualified']is False and proof['original_DEV_or_TEST_results']is False,
        'CONIC_REUSE_ACTUAL_CLOSED_C500_ORDINARY40_PROOF')
    ordinary=producer/'original-full-inference';result=json.loads((ordinary/'RESULT.private.json').read_bytes())
    task.need(result['original_full_inference']is True and result['actual_queries']==len(result['records'])==40
        and result['adaptation_step']==500,'CONIC_REAL_FULL_MODEL_NOT_FEATURE_SWAP')
    for row in result['records']:
        file=ordinary/row['sample_file'];task.need(Path(row['sample_file']).name==row['sample_file']and digest(file)==row['sha256'],
            'CONIC_PINNED_C500_ORDINARY_PREDICTION')
        with np.load(file,allow_pickle=False)as z:
            task.need(z['physical_H40'].shape==(40,10)and z['normalized'].shape==(1,40,132)
                and np.isfinite(z['physical_H40']).all()and np.isfinite(z['normalized']).all(),
                'CONIC_ORIGINAL_FULL_H40_SINGLE_DECODE')
    parent=verify_parent(m['parent_package']['path'],SFT2000_PACKAGE_SHA256,checkpoint_files)
    task.need(parent['recipe']['dataset_sha256']==DATA and parent['recipe']['statistics_sha256']==STATS,
        'CONIC_ORIGINAL_SFT2000_DATA_AND_PROCESSOR')
    return dict(schema='CONIC_C500_ORIGINAL_MODEL_CUSTODY_REUSED_V3',package=package,parent_package=m['parent_package'],
        ordinary40_proof=m['reconstruction'],reconstruction_spec=dict(path=str(spec_path),sha256=digest(spec_path)),
        GPU_uuid=s['GPU_uuid'],precision='ORIGINAL_BF16_AUTOCAST_FP32_PARAMETERS',denoising_steps=4,
        actual_original_full_generations_reused=40,new_model_calls=0,new_learning_updates=0,
        physical_execution_admission=False,original_cross_hardware_REJECT_preserved=True)


def install_ordinary_model_delta(model,package,checkpoint_files):
    """Same verified ordinary weight mapping as the completed C500 rebuild."""
    proof=verify_C500_package(package,checkpoint_files)
    from .target_gr00t_delivery_v1 import install
    install(model,proof['parent_package']['path'],SFT2000_PACKAGE_SHA256,checkpoint_files)
    import torch
    from .task_learning_features_v1 import install_ordinary_diagnostic_delta
    root=Path(package['path']);values=torch.load(root/'target_parameters.pt',map_location='cpu',weights_only=True)
    install_ordinary_diagnostic_delta(model,values,json.loads((root/'PARAMETER_CONTRACT.json').read_bytes()))
    model.requires_grad_(False);model.eval()
    return proof


def action_packet(value,method):
    """Run in the neural worker; preserve the original raw/normalized arrays."""
    if method not in METHODS:raise ValueError('Explicit preregistered C500 method required')
    raw=np.asarray(value['physical_H40']);normalized=np.asarray(value['normalized'])
    if raw.shape!=(40,10)or normalized.shape!=(1,40,132)or not np.isfinite(raw).all()or not np.isfinite(normalized).all():
        raise ValueError('Original full finite GR00T and single Processor decode required')
    packet=dict(physical_H40=raw,normalized=normalized)
    if method=='C500_RAW':return packet
    from .task_learning_conic_decoder_v3 import fit
    decoded,audit=fit(raw)
    return seal_conic_packet(value,decoded,audit)


def seal_conic_packet(value,decoded,audit):
    """Encode an already computed result; useful for sealed interface replay."""
    raw=np.asarray(value['physical_H40']);audit=dict(audit)
    packet=dict(physical_H40=raw,normalized=np.asarray(value['normalized']))
    audit.update(execution_version=EXECUTION_VERSION,raw_identity=identity(raw),
        decoded_identity=None if decoded is None else identity(decoded),
        raw_output_preserved=True,object_truth_or_expert_inputs_used=False)
    packet['conic_fit_audit_json']=np.asarray(json.dumps(audit,sort_keys=True,allow_nan=False))
    if decoded is not None:packet['explicit_conic_H40']=decoded
    # A solver rejection is returned as an actual completed neural query. The
    # native compiler issues a controlled proposal rejection; no raw fallback.
    return packet


def compile_proposal(action,pin,*,method,source_from_target,query_state10,actual_TCP,
                     consumed_reference_TCP,base_hold):
    """Pure compilation and original gates; no simulator calls or side effects."""
    from .task_learning_student_v1 import StudentProposalRejected,compile_proposal as original
    if method not in METHODS:raise ValueError('Explicit preregistered C500 method required')
    if method=='C500_RAW':
        program,proposal=original(action,pin,source_from_target=source_from_target,query_state10=query_state10,
            actual_TCP=actual_TCP,consumed_reference_TCP=consumed_reference_TCP,base_hold=base_hold)
        proposal.update(composed_policy_method=method,explicit_decoder_applied=False,GR00T_raw_H40_consumed=True)
        return program,proposal
    def reject(why):raise StudentProposalRejected('TASK_LEARNING_STUDENT_FULL_PROPOSAL_REJECTED: EXPLICIT_CONIC_'+why)
    task.need(digest(action)==pin,'EXPLICIT_CONIC_PACKET_ACTION_BYTES_CHANGED')
    with np.load(action,allow_pickle=False)as packet:
        raw=packet['physical_H40'].copy();normalized=packet['normalized']
        task.need('conic_fit_audit_json'in packet,'EXPLICIT_CONIC_PACKET_AUDIT_MISSING')
        audit=json.loads(str(packet['conic_fit_audit_json']))
        task.need(audit.get('version')==SOLVER_METHOD and audit.get('execution_version')==EXECUTION_VERSION
            and audit.get('raw_identity')==identity(raw)and audit.get('raw_output_preserved')is True
            and audit.get('object_truth_or_expert_inputs_used')is False,'EXPLICIT_CONIC_PACKET_METHOD_AND_RAW_IDENTITY')
        if not(audit.get('solver_success')is True and audit.get('translation',{}).get('status')=='Solved'
            and audit.get('gripper',{}).get('status')=='Solved'and 'explicit_conic_H40'in packet):reject('SOLVER_REJECTED_NO_FALLBACK')
        decoded=packet['explicit_conic_H40'].copy()
    task.need(raw.shape==decoded.shape==(40,10)and normalized.shape==(1,40,132)
        and all(np.isfinite(a).all()for a in(raw,decoded,normalized))
        and identity(decoded)==audit.get('decoded_identity'),'EXPLICIT_CONIC_PACKET_COMPLETE_FINITE_SAVED_OUTPUT_IDENTITY')
    task.need(np.array_equal(decoded[[0,-1]],raw[[0,-1]])and np.array_equal(decoded[:,3:9],raw[:,3:9]),
        'EXPLICIT_CONIC_PACKET_FIXED_ENDPOINTS_AND_ROTATIONS')
    from scripts.pi0292.eval_contract import matrix10
    from scripts.pi0291_arx5_data.native_dataset import pose10
    from .fusion_observation import rigid
    frame=np.asarray(source_from_target,float);rigid(frame)
    task.need(np.allclose(matrix10(query_state10),frame@np.asarray(actual_TCP),rtol=0,atol=1e-9),
        'EXPLICIT_CONIC_PACKET_SAME_CURRENT_CAPTURED_STATE_AND_NATIVE_TCP')
    try:
        target=pose10(np.linalg.inv(frame)@matrix10(decoded),decoded[:,9])
        program=task.FiftyMsProgram(target,actual_TCP,consumed_reference_TCP,base_hold)
    except ValueError as error:reject('ORIGINAL_FULL_H40_GATE:'+str(error))
    return program,dict(raw_policy_H40=raw.tolist(),explicit_conic_policy_H40=decoded.tolist(),
        mapped_target_H40=target.tolist(),original_Processor_decode_calls=1,action_sha256=pin,
        policy_frame_mapping_count=1,command_clips=0,teacher_fallback=False,expert_training_eligible=False,
        external_inference_adapter=False,composed_policy_method=method,explicit_decoder_applied=True,
        explicit_decoder_execution_version=EXECUTION_VERSION,solver_audit=audit,
        GR00T_raw_H40_consumed=False,composed_H40_requires_original_native_reference_and_consumption_audits=True)
