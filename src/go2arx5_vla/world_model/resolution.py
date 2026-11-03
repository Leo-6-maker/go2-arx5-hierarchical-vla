"""P0 resolution from explicit owner observations, never an authorization issuer."""
from copy import deepcopy


def resolve(binding):
    if binding.get('experiment_id')!='WM02' or binding.get('handoff_sha')!='ddc801f30e288365925c61c37f7ee445ddc29712':
        raise ValueError('Exact WM02 handoff binding required')
    target=deepcopy(binding['target']);blockers=[]
    if target.get('backend')!='GO2_ARX5_MUJOCO_OCS2':
        blockers.append('ACTUAL_TARGET_BACKEND_UNRESOLVED')
    permission=target.get('permission',{})
    # A new plan or even a metadata bool cannot supply native admission.
    if permission.get('native_safety_quarantine_preserved') is not False:
        blockers.append('TARGET_S1_T2_QUARANTINE_NOT_RELEASED')
    if permission.get('status')!='CURRENT_ORIGINAL_VALIDATOR_ADMITTED':
        blockers.append('NO_CURRENT_TARGET_DYNAMIC_ADMISSION')
    if not target.get('frozen_members')or not all(x.get('match')is True for x in target['frozen_members']):
        blockers.append('FROZEN_TARGET_IDENTITY_NOT_VERIFIED')
    if target.get('full_control_restore_available')is not True:
        blockers.append('MPC_WBC_CONTROL_STATE_RESTORE_UNAVAILABLE')
    return dict(schema='WM02_RESOLVED_SPEC_V1',experiment_id='WM02',handoff_sha=binding['handoff_sha'],
        source_sha=binding['source_sha'],target=target,source_diagnostic=deepcopy(binding['source_diagnostic']),
        owner_observation=deepcopy(binding.get('owner_observation',{})),
        target_status='TARGET_BLOCKED'if blockers else'IDENTITY_RESOLVED_RUNTIME_VALIDATION_REQUIRED',
        target_blockers=blockers,target_dynamic_authorized=False,
        original_validator='scripts.pi0294.task_capture_runtime.validate_entry',
        target_native_diagnostics_run=0,model_updates=0,actor_updates=0,
        target_note='Resolved bytes are not a native grant. Live target adapter remains unavailable until original owner admission/restore qualification.')
