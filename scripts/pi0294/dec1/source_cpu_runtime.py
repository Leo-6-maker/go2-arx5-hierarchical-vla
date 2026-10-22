"""Cold source CPU filesystem admission draft; never registers or launches.

Uses the existing PV1_STAGE_BINDING dispatch seam. Parent global resource
projection originates on LOCAL PC: only its sealed snapshot is on CoreLab.
A real Core child lease AND once native DEC1 reservation must already exist.
"""
import hashlib
import json
import os
from pathlib import Path
import stat
import time
from scripts.pi0294.pv1.scope import validate as validate_shared
from . import source_cpu_intake as intake

ROLE=intake.ROLE
STAGE='PV1_STAGE_BINDING.json'
PARENT_SNAPSHOT='plans/SOURCE_PARENT_RESOURCE_SNAPSHOT.private.json'
DEC1_SNAPSHOT='plans/SOURCE_CORE_DEC1_PRE_RESERVATION_SNAPSHOT.private.json'
PROJECTION='plans/SOURCE_CPU_NATIVE_RESOURCE_PROJECTION.json'
CI='plans/PUBLICATION_AND_CI.private.json'
ASSETS='plans/ASSET_IDENTITY.private.json'
CURRENT_DEC1='authority/DEC1/DEC1_RESOURCE_LEDGER.json'
NATIVE_ADOPTION='authority/DEC1/DEC1_NATIVE_ADOPTION_AND_BUDGET.json'
RESERVATION_SCHEMA='DEC1_SOURCE_CPU_COMPONENT_NATIVE_RESERVATION_V1'
PROJECTION_SCHEMA='DEC1_SOURCE_CPU_NATIVE_RESOURCE_PROJECTION_V1'
CI_SCHEMA='DEC1_SOURCE_CPU_EXACT_CI_BINDING_V1'
ENTRY_SHA256='0c7973e0a3bc0e585e7ab96cfcc9033532a98f56e9c9410882282a291bb7082f'
DOCTOR_SHA256='013e7cea41740b10fdd213e30f6d65b0fbcaee295cce44b624fc61e23fe6d861'
ASSET_CANONICAL_SHA256='edcd253c9f6a1ee09237bba9135957015e3e901d3fa25ff9a8cdd9c8643ec4a0'
REQUIRED_RUNTIME_FILES=(
    'scripts/pi0294/dec1/source_cpu_runtime.py','scripts/pi0294/dec1/source_cpu_intake.py',
    'scripts/pi0294/dec1/source_component_meter.py','scripts/pi0294/dec1/source_component_evidence.py',
    'scripts/pi0294/dec1/source_component_ack.py','scripts/pi0294/dec1/source_episode_wire.py',
    'scripts/pi0294/dec1/source_single_pose_prefix.py','scripts/pi0294/dec1/source_sync_stop.py',
    'scripts/pi0294/dec1/source_component_factory.py',
    'scripts/pi0294/pv1/scope.py','scripts/pi0294/pv1/core_ledger.py',
    'scripts/pi0294/pv1/runtime.py','scripts/pi0294/pv1/core.py')


def _need(ok,reason):
    if not ok:raise PermissionError(reason)


def _sha(raw):return hashlib.sha256(raw).hexdigest()


def _canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def _file(path,*,within=None,readonly=False):
    path=Path(path)
    if within is not None:
        base=Path(within)
        _need(not path.is_absolute() or path.resolve().is_relative_to(base.resolve()),'SOURCE_CPU_PATH_ESCAPE')
        _need(path.resolve().is_relative_to(base.resolve()),'SOURCE_CPU_PATH_ESCAPE')
        relative=path.relative_to(base)
        for part in (path,*[path.parents[i] for i in range(len(relative.parts)-1)]):
            _need(not part.is_symlink(),'SOURCE_CPU_NO_MUTABLE_SOURCE_OR_INPUT_SYMLINK')
    info=path.stat()
    _need(stat.S_ISREG(info.st_mode) and info.st_uid==os.getuid(),'SOURCE_CPU_OWNED_REGULAR_FILE')
    _need(not readonly or not info.st_mode&0o222,'SOURCE_CPU_SEALED_BYTES_MUST_BE_READ_ONLY')
    return path.read_bytes()


def _json(path,*,within=None,readonly=False):
    return intake._object(_file(path,within=within,readonly=readonly),str(path))


def _dims(value,name):return intake._mapping(value,name)


def _reservation(root,m,stage,budget,current,pre,*,closeout,now):
    """Verify native once intent; read-only. Historical/GPU fields untouched."""
    _need(current.get('schema')=='DEC1_NONADDITIVE_RESOURCE_CLASSIFICATION_V1'
        and current.get('authority_release')==intake.AUTHORITY
        and current.get('campaign_run_uuid')==stage['campaign_run_uuid']
        and current.get('original_T0_epoch')==pre['original_T0_epoch']
        and current.get('hard_close_epoch')==intake.HARD_CLOSE,
        'SOURCE_CPU_NATIVE_CORE_CLASSIFICATION_IDENTITY')
    _need(closeout or current.get('unknown_effects')==[],
        'SOURCE_CPU_NATIVE_UNKNOWN_NO_NEW_WORK')
    rows=current.get('source_cpu_reservations')
    _need(type(rows) is dict and budget['request_id'] in rows,
        'SOURCE_CPU_NATIVE_ONCE_CHILD_RESERVATION_REQUIRED')
    row=rows[budget['request_id']]
    _need(row.get('schema')==RESERVATION_SCHEMA and row.get('request_id')==budget['request_id']
        and row.get('authority_release')==intake.AUTHORITY and row.get('role')==ROLE
        and row.get('campaign_run_uuid')==stage['campaign_run_uuid']
        and row.get('child_run_uuid')==root.name and row.get('child_root')==str(root)
        and row.get('child_manifest_sha256')==_sha(_file(root/'RUN_MANIFEST.json',within=root,readonly=True))
        and row.get('component_plan_sha256')==intake.PLAN_SHA256
        and row.get('budget_sha256')==m['execution']['sealed_inputs'][intake.PATHS['budget']]
        and row.get('parent_projection_sha256')==budget['parent_snapshot_sha256']
        and row.get('pre_reservation_DEC1_sha256')==budget['DEC1_snapshot_sha256']
        and row.get('native_projection_receipt_sha256')==budget['native_projection_receipt_sha256'],
        'SOURCE_CPU_EXACT_NATIVE_CHILD_RESERVATION_BINDING')
    allowed=('RESERVED','RUNNING','UNKNOWN','SETTLED') if closeout else ('RESERVED','RUNNING')
    _need(row.get('status') in allowed and (closeout or row.get('unknown_effects')==[]),
        'SOURCE_CPU_RESERVATION_STATE_NO_UNKNOWN_RESTART')
    _need(type(row.get('reserved_epoch')) in (int,float) and type(row['reserved_epoch']) is not bool
        and budget['observed_epoch']<=row['reserved_epoch']<=now
        and row['reserved_epoch']<row['deadline_epoch']
        and row.get('deadline_epoch')==m['limits']['operation_deadline_epoch'],
        'SOURCE_CPU_ORIGINAL_RESERVED_INTERVAL')
    requested=_dims(row.get('requested'),'native.requested')
    remaining=_dims(row.get('remaining'),'native.remaining')
    _need(requested==budget['requested'] and all(remaining[k]<=requested[k] for k in intake.DIMS),
        'SOURCE_CPU_EXACT_NATIVE_TOTAL_AND_REMAINING_RESERVATION')
    aggregate=dict.fromkeys(intake.DIMS,0)
    source_slots=0
    for request_id,item in rows.items():
        _need(type(item) is dict and item.get('request_id')==request_id
            and item.get('role')==ROLE and item.get('schema')==RESERVATION_SCHEMA,
            'SOURCE_CPU_TYPED_SOURCE_RESERVATIONS_ONLY')
        _dims(item.get('requested'),'all_native.requested');left=_dims(item.get('remaining'),'all_native.remaining')
        _need(all(left[k]<=item['requested'][k] for k in intake.DIMS),
            'SOURCE_CPU_NONNEGATIVE_REMAINING_NOT_EXPANDED')
        _need(item.get('status') in ('RESERVED','RUNNING','UNKNOWN','SETTLED'),
            'SOURCE_CPU_TYPED_NATIVE_RESERVATION_STATE')
        if item['status']!='SETTLED':
            _need(closeout or item.get('unknown_effects')==[],
                'SOURCE_CPU_OTHER_UNKNOWN_SOURCE_RESERVATION')
            for key in intake.DIMS:aggregate[key]+=left[key]
        else:_need(not any(left.values()),'SOURCE_CPU_SETTLED_RESERVATION_REMAINS_RESERVED')
        source_slots+=item['requested']['episode_starts']
    _need(source_slots<=4,'SOURCE_CPU_ORIGINAL_FOUR_SOURCE_SLOTS_NOT_MULTIPLIED')
    _need(_dims(current.get('source_cpu_reserved'),'native.aggregate')==aggregate,
        'SOURCE_CPU_ALL_NATIVE_RESERVATIONS_RECONCILE')
    for key,field in (('CoreLab_CPU_s','CPU_s'),('episode_starts','episode_starts'),
            ('physics_ticks','physics_ticks'),('constructors','constructors')):
        value=intake._number(current[field],'current.'+field,key!='CoreLab_CPU_s')
        _need(value>=pre[field] and (closeout or value+aggregate[key]<=intake.CAPS[key]),
            'SOURCE_CPU_CURRENT_COSTS_PRESERVED_AND_BOUNDED:'+key)
    # Local/output/transfer classification needs the native owner's complete
    # actual projection; old Core metadata has no sufficient legacy fields.
    projection=_dims(current.get('source_cpu_current_classification'),'native.complete_current')
    _need(projection['CoreLab_CPU_s']==current['CPU_s']
        and projection['episode_starts']==current['episode_starts']
        and projection['physics_ticks']==current['physics_ticks']
        and projection['constructors']==current['constructors'],
        'SOURCE_CPU_CURRENT_NATIVE_CLASSIFICATION_COUNTERS_MATCH')
    _need(all(projection[k]>=budget['DEC1_used'][k]
        and (closeout or projection[k]+aggregate[k]<=intake.CAPS[k]) for k in intake.DIMS),
        'SOURCE_CPU_COMPLETE_NATIVE_CLASSIFICATION_NO_RESET_OR_EXTRA_ALLOWANCE')



def validate(root,profile='4080-simulation',*,closeout=False):
    """Shared native CPU scope plus DEC1 source-only reservation/readbacks."""
    _need(type(closeout) is bool and profile=='4080-simulation','SOURCE_CPU_FIXED_PROFILE')
    _need(intake.PATHS['stage']==STAGE,'SOURCE_CPU_EXISTING_STAGE_DISPATCH_PATH_REQUIRED')
    root=Path(root).absolute()
    # The shared scope owns real host/profile/UID, unique registry, both leases,
    # source seal/import identity, original13+authorized2 and actual26 assets.
    # Its old roles keep their original branches. No parallel scope is created.
    m,inputs=validate_shared(root,role=ROLE,closeout=closeout)
    parent=root.parent.parent;code=root/m['code']['capsule_relative'];execution=m['execution']
    _need(execution['entry_sha256']==ENTRY_SHA256 and execution['doctor_sha256']==DOCTOR_SHA256,
        'SOURCE_CPU_APPROVED_STANDARD_ENTRY_IDENTITY')
    _need(Path(__file__).resolve()==(code/'scripts/pi0294/dec1/source_cpu_runtime.py').resolve()
        and Path(intake.__file__).resolve()==(code/'scripts/pi0294/dec1/source_cpu_intake.py').resolve(),
        'SOURCE_CPU_ACTUAL_IMPORTED_RUNTIME_AND_INTAKE')
    cpus=m['limits'].get('cpus')
    _need(type(cpus) is list and 1<=len(cpus)<=3
        and all(type(cpu) is int and cpu>=0 for cpu in cpus)
        and len(cpus)==len(set(cpus)) and set(cpus)<=os.sched_getaffinity(0),
        'SOURCE_CPU_ORIGINAL_EPISODE_ONE_TO_THREE_WORKER_CPUS')
    _need(os.environ.get('MUJOCO_GL')=='egl' and os.environ.get('LIBGL_ALWAYS_SOFTWARE')=='1'
        and os.environ.get('__EGL_VENDOR_LIBRARY_FILENAMES')=='/usr/share/glvnd/egl_vendor.d/50_mesa.json'
        and 'MUJOCO_EGL_DEVICE_ID' not in os.environ,'SOURCE_CPU_ORIGINAL_MESA_CPU_RENDERER')
    _need(closeout or not any((root/name).exists() for name in ('STOP.json','SOURCE_PHYSICS_STOP.json')),
        'SOURCE_CPU_CURRENT_CHILD_STOP')
    original_inputs=_json(parent/'INPUT_BINDINGS.private.json',within=parent,readonly=True)
    _need(all(key in inputs and inputs[key]==original_inputs[key] for key in ('asset_root','runtime_python')),
        'SOURCE_CPU_ORIGINAL_PARENT_ASSET_AND_RUNTIME_LOCATORS')
    sealed=execution['sealed_inputs'];_need(type(sealed) is dict,'SOURCE_CPU_SEALED_INPUT_TABLE')
    for name,digest in sealed.items():
        _need(not Path(name).is_absolute() and _sha(_file(root/name,within=root,readonly=True))==digest,
            'SOURCE_CPU_SEALED_INPUT_CHANGED:'+name)
    for name in (PARENT_SNAPSHOT,DEC1_SNAPSHOT,PROJECTION,CI,ASSETS):
        _need(name in sealed,'SOURCE_CPU_REQUIRED_NATIVE_SEALED_INPUT:'+name)
    ci=_json(root/CI,within=root,readonly=True)
    seal=_json(root/m['code']['source_seal_relative'],within=root,readonly=True);members=seal['files']
    _need(ci.get('schema')==CI_SCHEMA and ci.get('source_sha')==m['code']['vla_commit']
        and ci.get('CI_conclusion')=='success'
        and type(ci.get('CI_run_id')) is int and ci['CI_run_id']>0
        and type(ci.get('CI_job_id')) is int and ci['CI_job_id']>0,
        'SOURCE_CPU_ACTUAL_EXACT_RUNTIME_CI_BINDING')
    readback=ci.get('full_readback_file_sha256')
    _need(type(readback) is dict and all(name in members and readback.get(name)==members[name]
        for name in REQUIRED_RUNTIME_FILES),'SOURCE_CPU_EXACT_RUNTIME_FILES_FULL_READBACK')
    _need(_sha(_canonical(_json(root/ASSETS,within=root,readonly=True)))==ASSET_CANONICAL_SHA256,
        'SOURCE_CPU_ORIGINAL_APPROVED_ASSET_TABLE_IDENTITY')
    raw={key:_file(root/name,within=root,readonly=True) for key,name in intake.PATHS.items()}
    parent_raw=_file(root/PARENT_SNAPSHOT,within=root,readonly=True)
    dec1_raw=_file(root/DEC1_SNAPSHOT,within=root,readonly=True)
    native_adoption_raw=_file(parent/NATIVE_ADOPTION,within=parent,readonly=True)
    _need(raw['adoption']==native_adoption_raw,'SOURCE_CPU_ACTUAL_PARENT_DEC1_ADOPTION_BYTES')
    original=_json(parent/'PV1_ADOPTION_AND_RESOURCE_BINDING.json',within=parent,readonly=True)
    actual=intake._object(native_adoption_raw,'actual native DEC1 adoption')
    _need(actual['campaign_run_uuid']==original['run_uuid']
        and actual['original_T0_epoch']==original['T0_epoch']
        and actual['original_hard_close_epoch']==original['hard_closeout_epoch']==intake.HARD_CLOSE
        and original['stop_new_physics_epoch']==original['T0_epoch']+160*3600,
        'SOURCE_CPU_ACTUAL_PARENT_AUTHORITY_CLOCK_NOT_SNAPSHOT_ONLY')
    stage=intake._object(raw['stage'],'stage');budget=intake._object(raw['budget'],'budget')
    projection=_json(root/PROJECTION,within=root,readonly=True)
    _need(projection.get('schema')==PROJECTION_SCHEMA and projection.get('authority_release')==intake.AUTHORITY
        and projection.get('campaign_run_uuid')==original['run_uuid']
        and projection.get('parent_projection_origin')=='EXISTING_LOCAL_PC'
        and projection.get('native_safety_unknown_effects')==[] and projection.get('unknown_effects')==[]
        and projection.get('parent_snapshot_sha256')==_sha(parent_raw)
        and projection.get('DEC1_snapshot_sha256')==_sha(dec1_raw)
        and projection.get('adoption_sha256')==_sha(native_adoption_raw),
        'SOURCE_CPU_REAL_SEALED_NATIVE_PROJECTION_ORIGIN_AND_BYTES')
    for key in ('observed_epoch','valid_until_epoch','DEC1_used','DEC1_reserved',
            'parent_remaining_after_spent_reserved_and_final','site_remaining',
            'source_commissioning_used','source_commissioning_reserved','target_component_reserved'):
        _need(key in projection and projection[key]==budget[key],
            'SOURCE_CPU_NATIVE_PROJECTION_VALUE_CHANGED:'+key)
    active=m['limits'].get('active_cores');other=projection.get('other_registered_CoreLab_cores')
    _need(type(active) is int and len(cpus)<=active<=budget['concurrent_CoreLab_cores']<=4
        and type(other) is int and 0<=other and len(cpus)+other<=budget['concurrent_CoreLab_cores'],
        'SOURCE_CPU_WORKER_PLUS_PROJECTED_CONTROL_WITHIN_ORIGINAL_FOUR')
    frozen=set(intake.ORIGINAL_13)|set(intake.GLUE_2)
    source_files={name:_file(code/name,within=code,readonly=True) for name in frozen}
    intake.validate(m,stage_bytes=raw['stage'],plan_bytes=raw['plan'],population_bytes=raw['population'],
        adoption_bytes=raw['adoption'],parent_bytes=parent_raw,dec1_bytes=dec1_raw,
        budget_bytes=raw['budget'],source_files=source_files,
        # Revalidate the sealed admission snapshot for metadata closeout;
        # never present an expired projection as permission for new work.
        now_epoch=budget['observed_epoch'] if closeout else time.time())
    pre=intake._object(dec1_raw,'pre-reservation Core DEC1')
    current=_json(parent/CURRENT_DEC1,within=parent)
    _reservation(root,m,stage,budget,current,pre,closeout=closeout,now=time.time())
    # The shared core_ledger constructor/guard owns the same typed child meter
    # and limits. This layer adds only the once source budget binding.
    child=_json(root/'RESOURCE_LEDGER.json',within=root)
    _need(child.get('schema')=='PI0294_PV1_CORE_POLICY_LEDGER_V1'
        and child.get('run_uuid')==root.name and child.get('authority_release')==intake.PARENT_AUTHORITY
        and child.get('source_budget_request_id')==budget['request_id'],
        'SOURCE_CPU_REAL_CHILD_METER_ONCE_BUDGET_BINDING')
    return m
