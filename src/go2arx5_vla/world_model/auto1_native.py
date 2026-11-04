"""AUTO1's thin binding to the existing native simulator and physical owner.

No scheduling or native safety replacement: each invocation owns one bounded
stage, retains the existing wrapper/ROS entry and returns actual evidence.
"""
from pathlib import Path
import json,os,time
from contextlib import contextmanager

PHASES=(('cold-1',50),('cold-2',50),('hold-2s-1',1000),('hold-2s-2',1000),('hold-5s-1',2500),('hold-5s-2',2500))

def validate_parent_loader(expected,actual):
    # glibc's loader reads LD_LIBRARY_PATH at exec, before importing rclpy.
    # Setting it later only for native children cannot repair the parent.
    if not expected or actual!=expected:raise PermissionError('AUTO1_NATIVE_PARENT_LOADER_ENV_NOT_BOUND_AT_EXEC')

def phase_limit(phase,previous):
    if phase in ('g2-hold-short','g2-hold-renew','g2-eef-x','g2-eef-y'):
        from .target_reference import dynamic_limit
        return dynamic_limit(phase,previous)
    if phase=='g2-reference-qualification':
        if(previous is None or previous.get('phase')!='hold-5s-2'or previous.get('status')!='NATIVE_STAGE_PASSED'or not previous.get('shutdown_clean')or any(p['exit_code']!=0 for p in previous.get('owned_process_exits',[]))):raise PermissionError('COMPLETE_G1_REQUIRED_BEFORE_REFERENCE_QUALIFICATION')
        return 2503
    names=[n for n,_ in PHASES]
    if phase not in names:raise PermissionError('UNKNOWN_AUTO1_NATIVE_PHASE')
    index=names.index(phase)
    if index and(previous is None or previous.get('phase')!=names[index-1]or previous.get('status')!='NATIVE_STAGE_PASSED'or any(p.get('exit_code')!=0 for p in previous.get('owned_process_exits',[]))):raise PermissionError('PREVIOUS_NATIVE_STAGE_NOT_PASSED')
    return PHASES[index][1]

def validate_scope(spec,lease,registry,*,host,uid,now):
    root=Path(spec['campaign_root']);binding=Path(spec['campaign_binding'])
    from .storage import digest
    if spec.get('diagnostic_debugger')and(spec.get('standby_shutdown_diagnostic')is not True or spec['phase']!='cold-1'):raise PermissionError('DEBUGGER_ONLY_FOR_ZERO_TICK_STANDBY_DIAGNOSIS')
    if (spec['domain']!='WM02_AUTO1_TARGET_COMMISSION_V1'or spec['physics']is not True
            or spec['site_profile']!='4080-simulation'or spec['owner']!='EXISTING_LOCAL_PC'
            or host!='corelab-4080s'or uid!=2009 or spec['host']!=host or spec['uid']!=uid
            or lease.get('status')!='ACTIVE'or lease.get('scope')!='WM02_AUTO1_SIM_ONLY'
            or lease.get('owner')!=spec['owner']or lease.get('run_uuid')!=root.name
            or lease.get('binding_sha256')!=digest(binding)or now>=lease['deadline_epoch']
            or (root/'STOP.json').exists()or not 0<spec['wall_limit_s']<=600):raise PermissionError('AUTO1_REGISTERED_SCOPE_HOST_LEASE_OR_TIME_INVALID')
    if not any(e.get('scope')=='WM02_AUTO1_SIM_ONLY'and e.get('owner')==spec['owner']and e.get('path')==str(root)and e.get('local_lease')==str(root/'control/LEASE.json')for e in registry['entries']):raise PermissionError('AUTO1_ACTUAL_REGISTRY_BINDING_REQUIRED')

@contextmanager
def admitted_target(spec_path,run_id,output):
    from scripts.pi0292.durable import OwnerLock
    from .runtime import admitted_run
    from .storage import digest
    spec=json.loads(Path(spec_path).read_text());campaign=Path(spec['campaign_root'])
    previous=json.loads(Path(spec['previous_stage']).read_text())if spec.get('previous_stage')else None
    if previous is not None:
        if (digest(spec['previous_stage'])!=spec['previous_stage_sha256']or previous.get('campaign_uuid')!=campaign.name
                or previous.get('native_binary_sha256')!=digest(spec['native_binary'])or previous.get('observer_sha256')!=digest(spec['observer_library'])):raise PermissionError('PREVIOUS_STAGE_NOT_BOUND_TO_THIS_CANDIDATE')
    if type(spec['candidate_index'])is not int or not 1<=spec['candidate_index']<=6:raise PermissionError('AUTO1_FINITE_CANDIDATE_COUNT_REQUIRED')
    limit=phase_limit(spec['phase'],previous)
    if spec['max_ticks']!=limit:raise PermissionError('AUTO1_EXACT_PHASE_TICK_LIMIT_REQUIRED')
    validate_scope(spec,json.loads((campaign/'control/LEASE.json').read_text()),json.loads(Path(spec['registry']).read_text()),host=os.uname().nodename,uid=os.getuid(),now=time.time())
    for path,pin in spec['frozen_files'].items():
        if digest(path)!=pin:raise PermissionError('FROZEN_TARGET_INPUT_CHANGED:'+path)
    old=Path(spec['historical_target_lease'])
    if digest(old)!=spec['historical_target_lease_sha256']or json.loads(old.read_text())['status']!='CLOSED':raise PermissionError('HISTORICAL_LEASE_CHANGED')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU_TARGET_COMMISSIONING_ONLY')
    validate_parent_loader(spec['native_environment']['LD_LIBRARY_PATH'],os.environ.get('LD_LIBRARY_PATH'))
    if str(Path(spec['xvfb_run']))not in spec['frozen_files']or not os.access(spec['xvfb_run'],os.X_OK):raise PermissionError('BOUND_EXISTING_XVFB_ENTRY_REQUIRED')
    campaign_lock=OwnerLock(campaign/'OWNER.lock')
    try:
        native_lock=OwnerLock(spec['physical_owner_lock'])
        try:
            with admitted_run(spec_path,run_id,'wm02-auto1-target',output,physics=True):yield spec
        finally:native_lock.close()
    finally:campaign_lock.close()

def task_observer_source(text):
    """Restore the exact pre-existing task pair rule omitted by the generic build."""
    old='    illegal_contact|=!allowed;'
    new='''    auto pair=[&](const char* x,const char* y) {
      return a && b && ((std::string(a)==x && std::string(b)==y) || (std::string(a)==y && std::string(b)==x));
    };
    bool task_pair=pair("task_table","task_cube_geom") || pair("task_left_pad","task_cube_geom") || pair("task_right_pad","task_cube_geom");
    allowed=allowed || task_pair;
    illegal_contact|=!allowed || (task_pair && d->contact[i].dist < -.004);'''
    if text.count(old)!=1:raise ValueError('Exact generic observer task-adaptation anchor required')
    return text.replace(old,new)
