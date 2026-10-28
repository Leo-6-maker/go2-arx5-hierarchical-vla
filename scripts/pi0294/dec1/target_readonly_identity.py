"""Readonly supplied-byte inventory of the existing MuJoCo/OCS2 task entry.

No filesystem, runtime imports, validation calls, controller or admission. Hash
observations and expected document pins must come from the actual owner reader.
An identity-complete packet still grants no current DEC1 task qualification.
"""
import hashlib
import json
import math

SCOPE='GR00T_TASK_CAPTURE_FROZEN_WBC_V1'
PROFILE='4080-simulation'
ASSET='FULLBODY_TASK_OVERLAY_V2_WORKING_STROKE'
ROLES=('manifest','input_binding','output_binding','registry','lease','source_seal',
       'window_adoption','startup_config','clock_build','pause_build','adapter_build','task_acceptance','pilot_plan')
ENTRY='scripts.pi0294.task_native_worker.run'
VALIDATOR='scripts.pi0294.task_capture_runtime.validate_entry'


def sha(raw):return hashlib.sha256(raw).hexdigest()


def _object(raw):
    def unique(pairs):
        v={}
        for k,x in pairs:
            if k in v:raise ValueError('DUPLICATE_KEY')
            v[k]=x
        return v
    if type(raw) is not bytes:raise ValueError('EXACT_BYTES_REQUIRED')
    value=json.loads(raw,object_pairs_hook=unique,
        parse_constant=lambda _:(_ for _ in ()).throw(ValueError('NONFINITE_JSON')))
    if type(value) is not dict:raise ValueError('OBJECT_REQUIRED')
    return value


def inspect(documents,expected_sha256,file_sha256,*,now_epoch):
    """Return only identity/missing-field observations, never native GO.

    documents maps existing receipt roles to exact bytes; expected_sha256 is
    independently fixed by caller, not generated from these same bytes. File
    hashes map actual original paths/receipt-relative members to owner readback
    SHA values. No full private path is copied into the returned report.
    """
    missing=[];mismatch=[];docs={};identities={}
    for role in ROLES:
        if role not in documents or role not in expected_sha256:
            missing.append('document:'+role);continue
        try:
            raw=documents[role]
            if sha(raw)!=expected_sha256[role]:raise ValueError('PIN_MISMATCH')
            docs[role]=_object(raw);identities[role]=sha(raw)
        except (TypeError,ValueError):mismatch.append('document:'+role)
    def get(role,*keys):
        try:
            v=docs[role]
            for k in keys:v=v[k]
            if v is None:missing.append('field:'+role+'.'+'.'.join(keys))
            return v
        except (KeyError,TypeError):missing.append('field:'+role+'.'+'.'.join(keys));return None
    def check(role,expected,*keys):
        value=get(role,*keys)
        if value is not None and value!=expected:mismatch.append('field:'+role+'.'+'.'.join(keys))
    def file(label,path,pin):
        if type(path) is not str or not path or type(pin) is not str or len(pin)!=64:
            missing.append('file_identity:'+label)
        elif path not in file_sha256:missing.append('file_readback:'+label)
        elif file_sha256[path]!=pin:mismatch.append('file_readback:'+label)
    check('manifest',SCOPE,'scope');check('manifest',PROFILE,'execution','profile')
    manifest=docs.get('manifest',{});execution=manifest.get('execution',{});code=manifest.get('code',{})
    if type(now_epoch) not in (int,float) or not math.isfinite(now_epoch):missing.append('finite_observation_epoch')
    runtime=code.get('vla_commit');run_id=manifest.get('run_uuid')
    if type(runtime) is not str or len(runtime)!=40 or any(c not in '0123456789abcdef' for c in runtime):
        missing.append('manifest.code.vla_commit')
    if type(run_id) is not str or not run_id:missing.append('manifest.run_uuid')
    asset_root=get('input_binding','asset_root')
    if type(asset_root) is not str or not asset_root.startswith('/'):missing.append('input_binding.asset_root')
    profile=get('manifest','execution','profile_python')
    entry=get('manifest','execution','entry_path');entry_sha=get('manifest','execution','entry_sha256')
    file('standard_entry',entry,entry_sha)
    if type(profile) is not str or not profile:missing.append('profile_python')
    run=manifest.get('run_uuid');check('output_binding',run,'run_uuid');check('lease',run,'run_uuid')
    check('source_seal',code.get('vla_commit'),'source_sha')
    if 'source_seal' in identities:check('manifest',identities['source_seal'],'code','source_seal_sha256')
    if 'output_binding' in identities:check('manifest',identities['output_binding'],'execution','output_binding_sha256')
    if 'window_adoption' in identities:
        check('manifest',identities['window_adoption'],'authority','adoption_sha256')
        check('lease',identities['window_adoption'],'adoption_sha256')
    check('window_adoption',True,'authorizes_execution')
    check('window_adoption',False,'counter_reset')
    check('window_adoption',False,'old_lease_reopened')
    members=get('source_seal','files')
    if type(members) is dict and members:
        for i,(path,digest) in enumerate(members.items()):file('source_member_'+str(i),'source/'+path,digest)
    else:missing.append('source_seal.files')
    check('task_acceptance',ASSET,'version')
    check('task_acceptance',False,'physical_qualification')
    check('task_acceptance',0,'WBC_changes');check('task_acceptance',0,'safety_threshold_changes')
    binding=docs.get('output_binding',{});registry=docs.get('registry',{})
    if 'output_binding' in docs and 'registry' in docs:
        if not any(x.get('role')==binding.get('role') and x.get('path')==binding.get('run_root')
                   for x in registry.get('entries',[])):mismatch.append('registered_output_row')
    initialization=get('startup_config','initialization')
    if type(initialization) is dict:
        pins=initialization.get('pinned_files')
        if type(pins) is not dict or not pins:missing.append('startup_config.initialization.pinned_files')
        else:
            for i,(path,digest) in enumerate(pins.items()):file('startup_pinned_'+str(i),path,digest)
        compiled=initialization.get('compiled_model')
        if not pins or compiled not in pins:missing.append('compiled_model_hash_in_original_pins')
    for role,key,prefix,suffix in (('clock_build','source_pins','native/',''),
            ('clock_build','binary_sha256','clock/',''),('pause_build','binary_sha256','pause/',''),
            ('adapter_build','binaries','adapter/','.so')):
        pins=get(role,key)
        if type(pins) is not dict or not pins:missing.append('members:'+role+'.'+key);continue
        for i,(path,digest) in enumerate(pins.items()):file(role+'_'+key+'_'+str(i),prefix+path+suffix,digest)
    if 'clock_build' in docs and 'pause_build' in docs:
        check('pause_build',docs['clock_build'].get('binary_sha256'),'original_clock_binaries')
    cards=get('pilot_plan','cards')
    if type(cards) is list:
        expected=['TASK-PILOT-'+family+'-'+task for family in ('W0','W1') for task in ('T1','T2')]
        if [c.get('id') for c in cards]!=expected or {c.get('task') for c in cards}!={'T1','T2'}:
            mismatch.append('original_T1_T2_four_pilot_cards')
    window='UNOBSERVED'
    deadline=get('lease','deadline_epoch');lease_status=get('lease','status')
    if type(deadline) in (int,float) and type(now_epoch) in (int,float):
        window='HISTORICAL_OR_CLOSED' if now_epoch>=deadline or lease_status!='ACTIVE' else 'ACTIVE_BYTES_REQUIRES_ORIGINAL_VALIDATOR'
    if manifest.get('scope')!=SCOPE:kind='SUPPLIED_METADATA_IS_NOT_FROZEN_TARGET_TASK_ENTRY'
    elif missing or mismatch:kind='EXISTING_TARGET_ENTRY_IDENTITY_INCOMPLETE'
    else:kind='IDENTITY_BOUND_CURRENT_NATIVE_ADMISSION_UNCHECKED'
    return dict(schema='WDR1_EXISTING_TARGET_READONLY_IDENTITY_V1',status=kind,
        document_sha256=identities,missing=sorted(set(missing)),mismatch=sorted(set(mismatch)),
        historical_window=window,existing_callable=ENTRY,original_validator=VALIDATOR,
        interface=dict(wire='WORLD_EE_JSON_41_POINTS_NO_PREVIEW9',frame='world',tcp_site='x5_ee',
            eef10_converter='scripts.pi0294.base_intent.eef_document',gripper='task_state.write_gripper',
            gripper_units='MECHANICAL_FRACTION_q_div_0.044',cameras=['body_rgb','wrist_rgb'],
            clock='actual state.bin step/time + pause_ack + ROS source MPC',
            ACK='native_receiver accept+installed trace for exact wire SHA',
            stop='receiver CANCEL/TTL->original HOLD; owned native process terminal + physics stop readback'),
        pending_native=['CURRENT_REGISTERED_TARGET_SCOPE_WINDOW_LEASE_AND_RESERVATION',
            'ACTUAL_FROZEN_BINARY_CONFIG_WEIGHTS_AND_PLANT_READBACK',
            'T1_T2_OVERLAY_TCP_GRIPPER_CAMERA_STATE_AND_NATIVE_SAFETY',
            'TARGET_RAW_MAPPED_H40_PREFIX_SAFETY_AND_I_TIME_I_PROGRESS_MAPPING',
            'ACTUAL_REFERENCE_INSTALL_ACK_AND_SAFE_STOP',
            'CURRENT_DEC1_CORR_R0_MODEL_AND_TASK_CONDITION_BINDING'],
        source_training_waits_for_target=False,hold_RMS_repair_required=False,
        native_dispatch_admitted=False,target_task_native_qualified=False,
        executor_manifest_created=False,controller_or_role_changed=False)
