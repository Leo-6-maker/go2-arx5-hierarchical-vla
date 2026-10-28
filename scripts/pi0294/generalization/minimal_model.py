"""Cold raw-parent loader using the original N1.7/processor/inference paths.

Not an entry point or an allocation producer. No existing owner emits this
new scope's release yet. All actual dispatch remains fail-closed until native
request/manifest/lease/resources and the original owner are integrated.
"""
import hashlib
import json
from pathlib import Path
import time

from . import minimal_study as study
from .minimal_training import RAW_PARENT_FILES_ID, RAW_REVISION, ARMS
from .minimal_dataset import PANEL_SHA

SCOPE = 'ARX5_MINIMAL_MATCHED_RAW_PARENT_FIRST512_V1'
LOAD_STATUS = 'PASS_ARX5_MINIMAL_OWN_NATIVE_MODEL_ALLOCATION_V1'
FILES = {
    'model-00001-of-00002.safetensors': '8a1a1d8a33c99103c7c80c136073c5bb8bfe9ca8f7a970c93c033ea89742906d',
    'model-00002-of-00002.safetensors': 'c3f61940deb2007ba1ad7743013b57f0f8462356151db9655175d7aca2d40661',
    'config.json': '54c0367060cd310d0b3343fe72a589860a8b6e8173810164a4ffd6253f52e689',
    'processor_config.json': '85c1b4690ae090559e79a45193e598b65d6146eedf14750884da65e6d31032be',
    'embodiment_id.json': 'abc4a749389837416a102d9455a8c089336f2e417afb18565703b9944974bc35',
    'model.safetensors.index.json': '407804ea5a62f4f8823f48811ae0edbb82fac101e9cf4d7273e6e2f692bb4d59',
    'statistics.json': 'c97b1b07a82a8a8858771d56d278732d97dd8eae506032c146c77eb53828afc9',
}


def stat_identity(path):
    st = Path(path).stat()
    return dict(dev=st.st_dev, inode=st.st_ino, bytes=st.st_size,
                mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns)


def audit_raw_parent(checkpoint, *, guard):
    """Full input audit inside the caller's existing bounded native lifecycle.

    Do not call via a short observation RPC and assume timeout means exit.
    The owner must persist its ordinary once receipt before model release.
    This function never changes weights/statistics or supplies admission.
    """
    study.need(callable(guard), 'EXISTING_OWNER_GUARD_REQUIRED')
    root = Path(checkpoint); rows = {}; start = time.monotonic()
    for name, expected in FILES.items():
        guard()
        last_guard = time.monotonic()
        p = root/name; before = stat_identity(p); digest = hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda: f.read(4 * 1024**2), b''):
                if time.monotonic() - last_guard >= 5:
                    guard(); last_guard = time.monotonic()
                digest.update(chunk)
        study.need(stat_identity(p) == before and digest.hexdigest() == expected,
                   'EXACT_FULL_RAW_PARENT_BYTES_REQUIRED:' + name)
        rows[name] = dict(sha256=expected, stat=before)
    return dict(status='ACTUAL_FULL_RAW_PARENT_SEVEN_FILES_VERIFIED', revision=RAW_REVISION,
                raw_parent_files_identity=RAW_PARENT_FILES_ID, files=rows,
                wall_seconds=time.monotonic() - start, models_loaded=0)


def unchanged_raw_parent(checkpoint, audit):
    """Reuse completed full-stream bytes only while all native stat keys match."""
    study.need(audit.get('status') == 'ACTUAL_FULL_RAW_PARENT_SEVEN_FILES_VERIFIED'
               and audit.get('revision') == RAW_REVISION and audit.get('models_loaded') == 0
               and audit.get('raw_parent_files_identity') == RAW_PARENT_FILES_ID
               and set(audit.get('files', {})) == set(FILES), 'ACTUAL_FULL_RAW_AUDIT_NOT_HEADER_OR_TIMEOUT')
    for name, expected in FILES.items():
        row = audit['files'][name]
        study.need(row['sha256'] == expected and row['stat'] == stat_identity(Path(checkpoint)/name),
                   'RAW_PARENT_CHANGED_AFTER_FULL_AUDIT:' + name)


def check_release(allocation, request, *, request_bytes, job_id, hostname, now):
    """Distinct scope, fresh same-job release and full-resource proof required.

    Actual registry/source/lease/GLOBAL admission remains in the original owner
    guard. This consumer additionally refuses old C/D/GEN or header-only loads.
    """
    study.need(isinstance(request_bytes, bytes) and json.loads(request_bytes) == request
               and request.get('scope') == SCOPE and request.get('raw_parent_files_identity') == RAW_PARENT_FILES_ID
               and request.get('statistics_sha256') == study.R0 and request.get('panel_sha256') == PANEL_SHA
               and bool(request.get('request_id')) and bool(request.get('submission_nonce')),
               'DISTINCT_MINIMAL_RAW_PARENT_REQUEST_REQUIRED')
    proof = allocation.get('native_proof', {})
    study.need(allocation.get('status') == LOAD_STATUS and allocation.get('job_id') == job_id
               and bool(job_id) and hostname.split('.')[0].lower().startswith('gn-a40-')
               and allocation.get('request_id') == request['request_id']
               and allocation.get('submission_nonce') == request['submission_nonce']
               and allocation.get('request_sha256') == hashlib.sha256(request_bytes).hexdigest()
               and 0 <= now - allocation.get('observed_epoch', -1e30) <= 120
               and now < allocation.get('allocation_end_epoch', 0)
               and proof.get('complete_queue_verified') is True
               and proof.get('project_eligible') is True
               and proof.get('actual_project_quota_covers_remaining_output') is True
               and proof.get('whole_remaining_package_shortfalls') == {}
               and proof.get('global_unknown_effects') == [],
               'FRESH_SAME_ACTUAL_JOB_NATIVE_RESOURCES_AND_OWNER_RELEASE')


def loader_kwargs(local_backbone_name, cosmos_hub, cosmos_revision):
    # Both arms start with identical flags and forward graph. requires_grad is
    # selected only after strict raw loading, not by different loading flags.
    return dict(local_files_only=True, trust_remote_code=True, output_loading_info=True,
                model_name=str(local_backbone_name), transformers_loading_kwargs=dict(
                    local_files_only=True, trust_remote_code=True, cache_dir=str(cosmos_hub),
                    revision=cosmos_revision), tune_llm=False, tune_visual=False,
                tune_projector=True, tune_diffusion_model=True, tune_vlln=True,
                state_dropout_prob=0., backbone_trainable_params_fp32=True, load_bf16=False)


def validate_loading(model, loading):
    study.need(not any(loading.get(k) for k in
                       ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')),
               'NO_SILENT_NEW_PARAMETERS_OR_PARTIAL_RAW_CHECKPOINT_LOAD')
    config = model.config
    study.need(config.action_horizon == 40 and config.max_action_dim == 132
               and config.max_state_dim == 132 and config.max_num_embodiments == 32
               and model.action_head.num_inference_timesteps == 4
               and model.action_head.state_dropout_prob == 0,
               'EXACT_H40_PADDING132_CATEGORY32_FOUR_STEPS_AND_STATE0')


def load(root, request_path, allocation, bindings, *, checkpoint, audit, mode, guard):
    """Call only from the existing native owner, after actual new-scope release.

    E0 is a frozen offline reference. E1/E2 receive separate parameters from
    the same raw parent. Own512 restore is handled by the original Trainer,
    never by passing an adapted checkpoint as this raw-parent argument.
    """
    import os
    import socket
    study.need(mode in ('E0_RAW',) + ARMS and callable(guard), 'EXPLICIT_RAW_REFERENCE_OR_MATCHED_ARM')
    guard()
    from scripts.pi0294.dec1.a40_training_context import sealed
    root, request_path = Path(root), Path(request_path)
    request_bytes = sealed(root, request_path.relative_to(root).as_posix(), parse=False).read_bytes()
    request = json.loads(request_bytes)
    check_release(allocation, request, request_bytes=request_bytes, job_id=os.environ.get('PBS_JOBID'),
                  hostname=socket.gethostname(), now=time.time())
    from scripts.pi0294.dec1.a40_training_runtime import native_job
    actual_job, _, _ = native_job(request)
    study.need(actual_job == allocation['job_id'], 'INDEPENDENT_NATIVE_JOB_MATCH_BEFORE_MODEL_IMPORT')
    unchanged_raw_parent(checkpoint, audit)
    from scripts.pi0292.durable import read
    from scripts.pi0294.native_dataset import official_import, sha
    study.need(sha(Path(bindings['dataset'])/'statistics.json') == study.R0, 'ORIGINAL_R0_EXTERNAL_TO_RAW_STATS')
    official_import(bindings['source'])
    guard()
    import torch
    from transformers import AutoModel
    import gr00t.model.gr00t_n1d7.gr00t_n1d7
    from scripts.pi0294.local_backbone import local_model_name
    from scripts.pi0294.neural import COSMOS_REVISION
    from scripts.pi0292.train_adapter import build_processor
    from scripts.pi0294.e1.model import E1ResidentModel
    from .minimal_adapter import ParameterContract
    study.need(torch.cuda.device_count() == 1 and 'A40' in torch.cuda.get_device_name(0),
               'ONE_REAL_ALLOCATED_A40_REQUIRED')
    model, info = AutoModel.from_pretrained(checkpoint, **loader_kwargs(
        local_model_name(root, bindings), bindings['cosmos_hub'], COSMOS_REVISION))
    validate_loading(model, info)
    if mode == 'E0_RAW':
        model.requires_grad_(False); contract = None
    else:
        contract = ParameterContract(model, mode, 10)
    model.to('cuda').eval()
    guard()
    instance = E1ResidentModel.__new__(E1ResidentModel)
    instance.root, instance.bindings, instance.model = Path(root), bindings, model
    instance.processor = build_processor(bindings['source'], bindings['cosmos'],
                                         read(Path(bindings['dataset'])/'statistics.json'))
    instance.identity = dict(schema='ARX5_RAW_PARENT_ZERO_STEP_IDENTITY_V1', mode=mode,
                            global_step=0, checkpoint_manifest_sha256=RAW_PARENT_FILES_ID,
                            pretrained_revision=RAW_REVISION, files=dict(FILES),
                            statistics_sha256=study.R0, raw_pretraining_statistics_sha256=FILES['statistics.json'],
                            processor='EXISTING_ARX5_NEW_EMBODIMENT10_R0', full_model_loading_exact=True)
    instance.parameter_contract = contract
    return instance


def unload(instance):
    """Drop wrapper/contract ownership before the original CUDA unload.

    Deleting only instance.model leaves ParameterContract.model alive and can
    overlap two full GR00T models during qualification restore or arm switch.
    The caller must also drop its original Trainer/optimizer before switching.
    """
    contract=getattr(instance,'parameter_contract',None)
    if contract is not None:
        for hook in contract.hooks:hook.remove()
        contract.hooks=[]
        instance.parameter_contract=None
    contract=None
    from scripts.pi0294.e1.model import unload as original
    return original(instance)


def bind_saved_checkpoint(instance,checkpoint,*,guard):
    """Bind subsequent original fixed_draw intents to the actual trained state.

    Training or full qualification restore changes the in-memory parameters.
    Keeping raw-parent file hashes after that mutation would mislabel every
    saved query. Require the complete own state and actual parameter identity,
    then retain raw provenance separately from the current checkpoint bytes.
    This function performs no load, optimizer step or generation.
    """
    study.need(callable(guard),'CURRENT_ORIGINAL_OWNER_GUARD_BEFORE_CHECKPOINT_BIND')
    guard();root=Path(checkpoint)
    from scripts.pi0292.durable import read
    from scripts.pi0294.native_dataset import sha
    from . import minimal_training as training
    from .minimal_adapter import parameter_state_fingerprint
    contract=instance.parameter_contract;contract.validate();complete=read(root/'CHECKPOINT_COMPLETE.json')
    step=complete.get('global_step')
    study.need(instance.identity.get('mode')==contract.arm in ARMS
        and instance.identity.get('pretrained_revision')==RAW_REVISION
        and instance.identity.get('statistics_sha256')==study.R0
        and step in(2,512)and complete.get('training_identity')==contract.arm
        and complete.get('qualification_only')is(step==2)
        and complete.get('full_resume')is True and complete.get('session_terminal')is True
        and complete.get('scheduler_horizon')==1024
        and complete.get('parameter_state_format')=='WRAPPED_CATEGORY_TABLES_FULL_RESUME_ONLY',
        'ONLY_OWN_COMPLETE_STEP2_OR512_NOT_RAW_OR_OTHER_ARM')
    members=training.checkpoint_members(root,complete['files'])
    required={'optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','config.json',
        'SAMPLER_STATE.json','statistics.json','ADAPTER_PARAMETER_CONTRACT.json',
        'FROZEN_PARAMETER_IDENTITY.json','MODEL_PARAMETER_STATE_IDENTITY.json',
        'FULL_TRAINING_STATE_IDENTITY.json'}
    study.need(members==complete['files']and members.get('statistics.json')==study.R0
        and required<=set(members)and any(n.endswith('.safetensors')for n in members)
        and any(n.startswith('processor/')for n in members)
        and read(root/'ADAPTER_PARAMETER_CONTRACT.json')==contract.describe(),
               'CURRENT_COMPLETE_FULL_STATE_MEMBER_BYTES_AND_R0')
    actual=parameter_state_fingerprint(instance.model)
    study.need(actual==read(root/'MODEL_PARAMETER_STATE_IDENTITY.json'),
               'ACTUAL_IN_MEMORY_MODEL_MUST_MATCH_SAVED_OR_RESTORED_CHECKPOINT')
    guard()
    identity=dict(schema='ARX5_MINIMAL_ACTUAL_WRAPPED_CHECKPOINT_IDENTITY_V1',mode=contract.arm,
        global_step=step,checkpoint_manifest_sha256=sha(root/'CHECKPOINT_COMPLETE.json'),
        files=members,statistics_sha256=study.R0,pretrained_revision=RAW_REVISION,
        raw_parent_files_identity=RAW_PARENT_FILES_ID,raw_parent_files=dict(FILES),
        processor='EXISTING_ARX5_NEW_EMBODIMENT10_R0',full_model_loading_exact=True,
        parameter_state_format=complete['parameter_state_format'],qualification_only=step==2)
    instance.identity=identity
    return dict(identity=identity,actual_parameter_identity=actual)
