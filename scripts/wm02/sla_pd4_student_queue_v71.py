"""Finite follower for the two existing PD4 arms, then eight original raw rollouts.

Uses the existing Journal, registry, world locks, worker wrappers, file handoff,
native evaluator and shutdown. No retries, training or policy selection.
"""
import argparse
from contextlib import closing
import copy
import json
import os
from pathlib import Path
import subprocess
import time
import traceback


def read(path):
    return json.loads(Path(path).read_bytes())


def cold(root):
    lease=read(root/'control/LEASE.json')
    pid=subprocess.check_output(['systemctl','--user','show','wm02-'+root.name,
        '--property=MainPID','--value'],text=True,timeout=15).strip()
    return lease['status']=='CLOSED'and pid=='0'


def terminal_training(spec):
    from go2arx5_vla.world_model.storage import digest
    from go2arx5_vla.world_model import pd4_student_v71 as p
    from go2arx5_vla.sla_adapter.pd4_config_v71 import MANIFESTS
    q=Path(spec['training_root'])
    if digest(q/'EXECUTION_SPEC.json')!=spec['training_spec_sha256']:
        raise PermissionError('EXISTING_TRAINING_IDENTITY_CHANGED')
    result=read(q/'RESULT.private.json');matching=read(q/'MATCHING_ACTUAL.private.json')
    require_training_summary(result,matching)
    policies={}
    for view in MANIFESTS:
        root=q/'arms'/(q.name+'-'+view.lower())
        if not cold(root):raise PermissionError('BOTH_TRAINING_ARMS_MUST_BE_COLD')
        value=read(root/'RESULT.private.json')
        policy=dict(arm=view,package=value['package'],qualification=dict(
            path=str(root/'RESULT.private.json'),sha256=digest(root/'RESULT.private.json')))
        p.policy_contract(policy)
        policies[view]=policy
    return policies,dict(result_sha256=digest(q/'RESULT.private.json'),
        matching_sha256=digest(q/'MATCHING_ACTUAL.private.json'),matching=matching,
        actual_optimizer_updates=1000,actual_training_visits=32000)


def require_training_summary(result,matching):
    if (result['exit_code']!=0 or result['state']!='BOTH_FIXED500_AND_FRESH_RGB_COMPLETE_OFFLINE_ONLY'
            or len(result['completed_arms'])!=2 or result['partial_arms']
            or(result['actual_optimizer_updates'],result['actual_training_visits'])!=(1000,32000)
            or matching['initial87_bit_exact']is not True or matching['committed_updates_per_arm']!=500
            or matching['actual_visit_sequences_equal']is not True
            or matching['state_dropout_sequences_equal']is not True or matching['physical_requests']!=0):
        raise PermissionError('BOTH_REAL500_MATCHED_START_INPUT_VISITS_AND_DROPOUT_REQUIRED')


def summarize(panel,outcomes):
    from go2arx5_vla.world_model.pd4_student_v71 import panel_contract,VERSION
    rows=panel_contract(panel)
    if [v['native_run']for v in outcomes]!=[v['native_run']for v in rows[:len(outcomes)]]:
        raise PermissionError('FIXED_ORDER_WITHOUT_CONDITION_REPLACEMENT')
    methods={}
    for method in panel['policies']:
        values=[v for v in outcomes if v['method']==method]
        methods[method]=dict(requested=4,attempted=sum(v['attempted']for v in values),
            physical_requests=sum(v['native_world_attempted']for v in values),
            terminal=len(values),task_successes=sum(v['technical_valid']and v['task_success']for v in values),
            valid_task_failures=sum(v['technical_valid']and not v['task_success']for v in values),
            technical_failures=sum(not v['technical_valid']for v in values),
            hard_safety_stops=sum(v['hard_safety_stop']for v in values),
            SLA_queries=sum(v.get('actual_SLA_queries',0)for v in values),
            native_integrations=sum(v.get('actual_native_integrations',0)for v in values))
    return dict(schema=VERSION+'_SUMMARY',requested=8,terminal=len(outcomes),complete=len(outcomes)==8,
        methods=methods,physical_requests=sum(v['native_world_attempted']for v in outcomes),
        optimizer_updates=0,training_visits=0,old_RP8_passed=False,DEV_conditions=0,TEST_conditions=0,
        generalization_claim=False,WBC_changes=False)


def register_conditions(spec,policies):
    from scripts.pi0292.durable import OwnerLock,write
    from scripts.pi0294.night10.native_relay import snapshot,once
    from go2arx5_vla.world_model.storage import digest
    from go2arx5_vla.world_model import pd4_student_v71 as p,stable_task_scene_v1 as scenes
    from go2arx5_vla.world_model.corelab_world_owner_v3 import SCHEMA,PLAN,DOMAIN_POOL,validate_layout
    from go2arx5_vla.sla_adapter.s16_offline_v7 import DATA_SHA
    g=Path(spec['registered_root']);h=g.parent;s=Path(spec['source_root']);c=Path(spec['campaign_root'])
    plan=read(g/'PLAN.private.json');seal=read(s/'SOURCE_STAGED.private.json')
    if digest(g/'PLAN.private.json')!=spec['plan_sha256']or seal['source_sha']!=spec['source_sha']:
        raise PermissionError('PREDECLARED_EIGHT_CONDITIONS_AND_SOURCE')
    panel=dict(plan,policies=policies);p.panel_contract(panel)
    manifest=h/'sla-n20-reference-data-v6-20261114-01/dataset/MANIFEST.private.json'
    if digest(manifest)!=DATA_SHA:raise PermissionError('ORIGINAL_N20_CUSTODY')
    m=read(manifest);originals={}
    for layout in p.LAYOUTS:
        ep=next(e for e in m['episodes']if e['layout']==layout and e['restart']==5)
        originals[layout]=ep['mount']['source_spec']
    end=time.time()+14400
    with closing(OwnerLock(c/'control/corelab-world-owners-v3/registration.lock')):
        if (g/'PANEL.private.json').exists():raise FileExistsError('RECONCILE_NO_REGISTRATION_REPLAY')
        write(g/'PANEL.private.json',panel)
        ports={int(l.split()[3].rsplit(':',1)[1])for l in subprocess.check_output(['ss','-H','-lun'],text=True).splitlines()if ':'in l.split()[3]}
        free=[i for i in sorted(DOMAIN_POOL)if not any(7400+250*i<=v<7400+250*(i+1)for v in ports)]
        if not free:raise BlockingIOError('NO_CURRENT_FREE_ROS_DOMAIN_NO_DISPATCH')
        domain=free[0]
        roots=[str(g/'runs'/(row['native_run']+suffix))for row in plan['assignments']for suffix in ('','-gpu')]
        binding=dict(schema=p.VERSION+'_BINDING',registered_root=str(g),scope=p.SCOPE,
            source_sha=spec['source_sha'],activated_epoch=end-14400,deadline_epoch=end,
            maximum_conditions=8,roots=roots,old_RP8_passed=False,maximum_parallel_worlds=1)
        bp=g/'control/BINDING.private.json';write(bp,binding)
        lease=read(g/'control/LEASE.json');write(g/'control/LEASE.json',dict(lease,binding_sha256=digest(bp)))
        reg,before=snapshot(spec['registry']);descriptors=[]
        nbase=read(g/'NATIVE_BASE.private.json');gbase=read(g/'GPU_BASE.private.json')
        recovered=h/'sla-frozen-controller-byte-custody-v7-20261114-01/go2_x5_ocs2_node'
        from go2arx5_vla.world_model.pd4_prospective_v71 import CONTROLLER
        if digest(recovered)!=CONTROLLER:raise PermissionError('ORIGINAL_CONTROLLER_BYTES_REQUIRED')
        for row in panel['assignments']:
            nr=g/'runs'/row['native_run'];gr=nr.with_name(nr.name+'-gpu')
            for r in (nr,gr):
                r.mkdir()
                for k in ('control','data','logs','tmp','cache'):(r/k).mkdir()
            old=scenes.pinned(originals[row['layout']]);q=p.policy_contract(policies[row['method']]);train=q['spec']
            if read(Path(old['registered_root'])/'control/LEASE.json')['status']!='CLOSED':
                raise PermissionError('ORIGINAL_PRODUCER_NOT_CLOSED')
            wb=dict(schema=SCHEMA,plan_revision=PLAN,worlds=[dict(root=str(nr),cpus=[2,3],ROS_DOMAIN_ID=domain)],
                pool_root=str(g/'runs'),campaign_root=str(c),source_sha=spec['source_sha'],reserved_cpus=[0,1,18,19])
            validate_layout(wb);wp=nr/'WORLD_BINDING.private.json';write(wp,wb)
            shared=dict(source_root=str(s),source_sha=spec['source_sha'],source_files=seal['files'],
                owner='EXISTING_LOCAL_PC',host=os.uname().nodename,uid=os.getuid(),registry=spec['registry'],
                deadline_epoch=end,wall_limit_s=3600,scope=p.SCOPE,core_local_protocol=p.PROTOCOL,
                task_learning_grasp=p.protocol(),target_gr00t_policy=policies[row['method']],
                target_gr00t_query_seeds=row['query_seeds'],target_gr00t_pair_id=row['native_run'],
                target_gr00t_B1_global_step=500,observation_camera_profile=nbase['observation_camera_profile'],
                enforce_cpu_slot_v5=True)
            dedicated=dict(version=p.VERSION,binding=dict(path=str(bp),sha256=digest(bp)),
                original_spec=originals[row['layout']],frozen_N20_manifest=dict(path=str(manifest),sha256=DATA_SHA),old_RP8_passed=False)
            n=copy.deepcopy(nbase)
            for k in ('pilot1_binding','pilot1_binding_sha256','fusion_binding','fusion_binding_sha256',
                    'night1_binding','next1_binding','next2_binding'):n.pop(k,None)
            frozen={k:v for k,v in old['frozen_files'].items()if Path(k).is_file()};frozen[str(recovered)]=CONTROLLER
            n.update(shared,registered_root=str(nr),role=nr.name.upper(),lease=str(nr/'control/LEASE.json'),
                registry_snapshot=str(nr/'control/REGISTRY_SNAPSHOT.private.json'),task_learning_role=p.PROTOCOL,
                task_condition=p.PROTOCOL,experiment_role=p.EXPERIMENT,execution_site=p.NATIVE_SITE,
                cpus=[2,3],ROS_DOMAIN_ID=domain,v3_plan_revision=PLAN,max_ticks=p.ENDPOINT,
                corelab_world_binding_v3=dict(path=str(wp),sha256=digest(wp)),controller_binary=str(recovered),
                native_binary=old['native_binary'],native_binary_source_sha=old['native_binary_source_sha'],
                frozen_files=frozen,model_path=old['model_path'],scene_id=old['scene_id'],data_parent=old['data_parent'],
                task_learning_scene_reference=old['task_learning_scene_reference'],data_role='TRAIN',
                scene_allowlist=old['scene_allowlist'],scene_allowlist_sha256=old['scene_allowlist_sha256'],
                target_gr00t_gpu_root=str(gr),stable_task_panel=str(g/'PANEL.private.json'),
                stable_task_panel_sha256=digest(g/'PANEL.private.json'))
            n[p.KEY]=dedicated
            n['native_environment']=dict(n['native_environment'],PYTHONPATH=str(s/'src')+':'+str(s)+':'+n['native_environment'].get('PYTHONPATH',''),CUDA_VISIBLE_DEVICES='',ROS_DOMAIN_ID=str(domain))
            gpu=copy.deepcopy(gbase)
            for k in ('pilot1_binding','pilot1_binding_sha256','fusion_binding','fusion_binding_sha256',
                    'night1_binding','next1_binding','next2_binding'):gpu.pop(k,None)
            runtime={k:train[k]for k in ('python','upstream_root','upstream_sha','JAX_memory_fraction','initial_function_rule')}
            runtime['compilation_cache_dir']=train['compilation_cache_dir']
            gpu.update(shared,registered_root=str(gr),role=gr.name.upper(),lease=str(gr/'control/LEASE.json'),
                registry_snapshot=str(gr/'control/REGISTRY_SNAPSHOT.private.json'),execution_site=p.GPU_SITE,
                python=train['python'],SLA_runtime=runtime,GPU_uuid=train['GPU_uuid'],cpus=[12,13,14,15],
                allocator_limit_bytes=train['allocator_limit_bytes'],checkpoint=train['parent_params'],checkpoint_files={},
                checkpoint_manifest_path=train['parent_manifest']['path'],checkpoint_manifest_sha256=train['parent_manifest']['sha256'],
                neural_owner_lock=train['neural_owner_lock'],memory_execution_reference=policies[row['method']]['qualification'])
            gpu[p.KEY]=dedicated
            for v,r in ((n,nr),(gpu,gr)):
                for k in ('data','cache','logs','tmp'):v[k]=str(r/k)
            ep=nr/'EPISODE_MANIFEST.private.json';write(ep,scenes.episode(n))
            n.update(episode_manifest=str(ep),episode_manifest_sha256=digest(ep),stage_identity_sha256=digest(ep))
            ns=nr/'EXECUTION_SPEC.json';write(ns,n)
            gpu['producer_native_spec']=dict(path=str(ns),sha256=digest(ns));gs=gr/'EXECUTION_SPEC.json';write(gs,gpu)
            for v,r in ((n,nr),(gpu,gr)):
                if any(e.get('path')==str(r)for e in reg['entries']):raise FileExistsError('FRESH_NATIVE_GPU_ROOT_REQUIRED')
                entry=dict(path=str(r),run_uuid=r.name,role=v['role'],source_sha=spec['source_sha'],scope=p.SCOPE,
                    owner=v['owner'],executor_node=v['host'],local_lease=v['lease'])
                if r==nr:entry.update(corelab_world_binding_sha256=digest(wp),corelab_world_owner=str(c/'control/corelab-world-owners-v3'/('ros-'+str(domain)+'.lock')),ROS_DOMAIN_ID=domain)
                reg['entries'].append(entry)
                write(v['lease'],dict(status='ACTIVE',run_uuid=r.name,spec_sha256=digest(r/'EXECUTION_SPEC.json'),
                    owner=v['owner'],scope=p.SCOPE,deadline_epoch=end))
            descriptors.append(dict(native=dict(path=str(ns),sha256=digest(ns)),gpu=dict(path=str(gs),sha256=digest(gs))))
        once(spec['registry'],reg,g.name.upper()+'-PAIR-REGISTRY',expected_before_sha256=before)
        for pair in descriptors:
            for pin in pair.values():write(Path(pin['path']).parent/'control/REGISTRY_SNAPSHOT.private.json',reg)
        write(g/'REGISTERED_PAIRS.private.json',descriptors)
    return panel,descriptors,end


def run(spec):
    from scripts.pi0292.durable import OwnerLock,write
    from scripts.pi0294.e1.journal import Journal
    from go2arx5_vla.world_model.storage import digest
    from go2arx5_vla.world_model import pd4_student_v71 as p
    from go2arx5_vla.world_model import task_learning_student_v1 as student
    from go2arx5_vla.world_model import task_learning_live_pair_v1 as live
    from go2arx5_vla.world_model.task_learning_student_batch_v1 import execute_pair
    from go2arx5_vla.world_model.task_learning_grasp_native_v1 import episode_config
    from go2arx5_vla.world_model.auto1_native import phase_limit
    from go2arx5_vla.world_model.sla_native_continuation_v2 import wait_seconds
    from go2arx5_vla.world_model.corelab_world_owner_v3 import world_owner
    g=Path(spec['registered_root']);q=Path(spec['training_root']);start=time.time();outcomes=[];panel=None;code=1;error=None;current=None
    j=Journal(g/'control/journal',g.name)
    schedule=spec.get('startup_schedule','CONCURRENT_ORIGINAL_V71')
    def stopped():return (g/'STOP.json').exists()or(Path(spec['campaign_root'])/'STOP.json').exists()
    try:
        if schedule not in ('CONCURRENT_ORIGINAL_V71','SLA_MODEL_READY_BEFORE_NATIVE_V71'):
            raise PermissionError('EXPLICIT_PD4_STARTUP_SCHEDULE_ONLY')
        while time.time()<spec['deadline_epoch']-14520:
            if stopped():raise PermissionError('OWNED_STOP_RETAINED')
            if cold(q):break
            write(g/'PROGRESS.private.json',dict(state='WAIT_BOTH_TRAINING_ARMS_TERMINAL',epoch=time.time(),optimizer_updates=0,physical_requests=0));time.sleep(30)
        else:raise TimeoutError('FINITE_TRAINING_FOLLOWER_NO_RESTART')
        policies,actual=terminal_training(spec);write(g/'TRAINING_GATE_ACTUAL.private.json',actual)
        panel,pairs,end=register_conditions(spec,policies)
        for row,pair in zip(panel['assignments'],pairs,strict=True):
            n=read(pair['native']['path']);gpu=read(pair['gpu']['path']);nr=Path(n['registered_root'])
            # All eight scopes are checked before any model or native process.
            episode_config(n)
            wait_seconds(n)
            # The typed phase limit performs the PD4 producer/prefix audit and
            # live-pair admission. Do not additionally invoke an unrelated old
            # source-pose calibration audit from the construction template.
            if phase_limit(n['phase'],read(n['previous_stage']),n)!=p.ENDPOINT:raise PermissionError('EXACT_ORIGINAL_ENDPOINT')
            live.preflight(n)
        write(g/'PREFLIGHT_ACTUAL.private.json',dict(conditions=8,passed=True,models=0,physical_requests=0))
        for row,pair in zip(panel['assignments'],pairs,strict=True):
            n=read(pair['native']['path']);gpu=read(pair['gpu']['path']);nr=Path(n['registered_root'])
            while time.time()<end-3600:
                if stopped():raise PermissionError('OWNED_STOP_RETAINED')
                free=int(subprocess.check_output(['nvidia-smi','-i',gpu['GPU_uuid'],'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=15).strip())*1024**2
                ram=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()if l.startswith('MemAvailable:'))
                ready=False
                if free>gpu['allocator_limit_bytes']+1024**3 and ram>32*1024**3:
                    try:
                        with closing(OwnerLock(gpu['neural_owner_lock'])),world_owner(n,read(n['registry'])):ready=True
                    except BlockingIOError:pass
                if ready:break
                write(g/'PROGRESS.private.json',dict(state='WAIT_SHARED_GPU_AND_WORLD_NO_MODEL',epoch=time.time(),terminal=len(outcomes),physical_requests=len(outcomes)));time.sleep(30)
            else:raise TimeoutError('FINITE_CAPACITY_WINDOW_UNRUN_SUFFIX_RETAINED')
            current=(row,pair)
            execute_pair(pair,j,g,len(outcomes),native_after_model=schedule=='SLA_MODEL_READY_BEFORE_NATIVE_V71')
            value=student.classify_actual(row,nr,spec['source_sha'],native_body=p.PROTOCOL)
            outcomes.append(value);current=None;write(g/'OUTCOMES.private.json',outcomes);write(g/'SUMMARY.private.json',summarize(panel,outcomes))
            if value['hard_safety_stop']or not value['technical_valid']:
                raise RuntimeError('ACTUAL_SAFETY_OR_ENGINEERING_FAILURE_RETAINED_NO_NEXT_CONDITION')
        code=0
    except BaseException:
        error=traceback.format_exc();write(g/'FAILED.private.json',dict(error=error,automatic_retry=False));traceback.print_exc()
        if current:
            row,pair=current
            try:
                outcomes.append(student.classify_actual(row,Path(pair['native']['path']).parent,spec['source_sha'],native_body=p.PROTOCOL));write(g/'OUTCOMES.private.json',outcomes)
            except BaseException:write(g/'UNRESOLVED_PAIR.private.json',dict(pair=pair,error=traceback.format_exc(),automatic_retry=False))
    finally:
        if panel:
            write(g/'SUMMARY.private.json',summarize(panel,outcomes))
            # Revoke only undispatched permissions; never invent a worker exit.
            for row in panel['assignments']:
                for suffix in ('','-gpu'):
                    r=g/'runs'/(row['native_run']+suffix)
                    if not j.path(r.name.upper()+'-RUN').exists()and not(r/'SUPERVISOR.private.json').exists():
                        lease=read(r/'control/LEASE.json')
                        write(r/'control/LEASE.json',dict(lease,status='CLOSED',worker_exit_code=None,closed_epoch=time.time(),terminal_reason='NEVER_DISPATCHED_PERMISSION_REVOKED'))
        write(g/'RESULT.private.json',dict(schema=p.VERSION+'_ACTUAL',exit_code=code,error=error,
            summary=summarize(panel,outcomes)if panel else None,optimizer_updates=0,training_visits=0,
            wall_s=time.time()-start,outcomes=outcomes,automatic_retry=False))
        write(g/'WORKER_EXIT.private.json',dict(exit_code=code,wall_s=time.time()-start,error=error))
        lease=read(g/'control/LEASE.json');write(g/'control/LEASE.json',dict(lease,status='CLOSED',worker_exit_code=code,closed_epoch=time.time()))
    return code


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--spec',type=Path,required=True);args=parser.parse_args(argv)
    from go2arx5_vla.world_model.storage import digest
    spec=read(args.spec);root=Path(spec['registered_root']);lease=read(root/'control/LEASE.json')
    if (os.getuid()!=2009 or os.uname().nodename!='corelab-4080s'or lease['status']!='ACTIVE'
            or lease['spec_sha256']!=digest(args.spec)or(root/'SUPERVISOR.private.json').exists()):
        raise PermissionError('FRESH_FINITE_CORE_PD4_FOLLOWER_REQUIRED')
    from scripts.pi0292.durable import write
    from scripts.pi0294.e1.journal import Journal
    write(root/'SUPERVISOR.private.json',dict(pid=os.getpid(),epoch=time.time()))
    j=Journal(root/'control/journal',root.name)
    result=j.once('RUN',dict(spec_sha256=digest(args.spec)),lambda:dict(exit_code=run(spec)))
    return result['exit_code']


if __name__=='__main__':raise SystemExit(main())
