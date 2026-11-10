"""Thin panel/group registration for the existing serial native DEV worker.

No policy selection, training, queue polling or automatic continuation lives
here. The existing owner calls each operation once with actual sealed inputs.
"""
import json,os,subprocess,time
from pathlib import Path
from contextlib import closing
from . import task_learning_student_v1 as student,task_learning_grasp_v1 as task
from . import task_learning_student_batch_v1 as batch
from .stable_task_scene_v1 import pinned
from .storage import digest

SCHEMA='TASK_LEARNING_STUDENT_DEV_REGISTRATION_V2'
MEMBER='src/go2arx5_vla/world_model/task_learning_student_registration_v2.py'


def descriptor(path):
    path=Path(path);return dict(path=str(path),sha256=digest(path))


def make_panel(source_sha,step,seed,policies,scenes,prefix):
    """Same DEV layouts, restarts and query noise at every scientific endpoint."""
    from .task_learning_dev_v1 import DEV_OFFSETS
    from .task_learning_scene_v1 import scene_id
    task.need(seed in(20261109,20261110,20261111)and step in(2000,4000,8000)
        and Path(prefix).name==prefix and not prefix.startswith('.'), 'DECLARED_DEV_CANDIDATE_IDENTITY')
    task.need(set(scenes)==set(DEV_OFFSETS),'EXACT_ALL_TWENTY_ORIGINAL_DEV_SCENES')
    rows=[]
    for i,layout in enumerate(DEV_OFFSETS):
        proof=pinned(scenes[layout]);model=Path(scenes[layout]['path']).parent/'SCENE.private.mjb'
        task.need(proof['layout']==layout and proof['parent']==scene_id(layout)
            and proof['split']=='DEV'and proof['compiled']['scene_sha256']==digest(model),
            'ACTUAL_STATIC_DEV_SCENE_NOT_TEST_OR_FOREIGN_LAYOUT')
        for restart in(1,2):
            # Matched across methods and endpoints; no result-dependent draws.
            seeds=list(range(202611090000+1000*i+100*restart,
                             202611090000+1000*i+100*restart+task.QUERIES))
            for method in(('B0','B1')if restart==1 else('B1','B0')):
                rows.append(dict(native_run=f'{prefix}-{layout}-r{restart}-{method.lower()}',
                    method=method,layout=layout,parent=scene_id(layout),restart=restart,
                    split='DEV',purpose=student.PURPOSE,command=task.COMMAND,
                    model_sha256=digest(model),query_seeds=seeds))
    panel=dict(schema=student.SCHEMA,task_learning_role=task.STUDENT_DEV_ROLE,
        protocol=task.protocol(task.STUDENT_DEV_ROLE),source_sha=source_sha,
        B1_global_step=step,training_seed=seed,policies=policies,assignments=rows)
    student.validate(panel);return panel


def closed_base(value):
    spec=pinned(value);root=Path(spec['registered_root'])
    task.need(Path(value['path'])==root/'EXECUTION_SPEC.json','EXACT_CLOSED_BASE_SPEC')
    lease=json.loads((root/'control/LEASE.json').read_bytes())
    worker=json.loads((root/'WORKER_EXIT.private.json').read_bytes())
    task.need(lease.get('status')=='CLOSED'and lease.get('spec_sha256')==value['sha256']
        and lease.get('worker_exit_code')==worker.get('exit_code')==0,
        'ACTUAL_CLOSED0_NATIVE_OR_NEURAL_BASE')
    return spec


def actual_site(campaign):
    task.need(os.getuid()==2009 and os.uname().nodename=='corelab-4080s'
        and not(Path(campaign)/'STOP.json').exists(),'ORIGINAL_CORE_OWNER_WITHOUT_STOP')


def register_panel(request):
    from scripts.pi0292.durable import OwnerLock,write
    from scripts.pi0294.e1.journal import Journal
    from scripts.pi0294.night10.native_relay import snapshot,once
    from .target_gr00t_profile_v1 import PARENT
    from . import task_learning_live_pair_v1 as pair
    q=pinned(request['Core_qualification']);core=pinned(q['Core_spec'])
    n=closed_base(request['native_base']);g=closed_base(request['gpu_base'])
    actual_site(n['campaign_root']);source=Path(core['source_root']);root=Path(request['root'])
    task.need(source.resolve()==source and root.parent.resolve()==root.parent
        and root.parent==Path(core['registered_root']).parent
        and digest(source/MEMBER)==core['source_files'].get(MEMBER)==digest(__file__),
        'FRESH_PANEL_UNDER_SAME_REGISTERED_PROJECT_AND_SEALED_SOURCE')
    step=q['global_step'];package=q['package']
    policies={m:dict(arm=m,parent_manifest_sha256=PARENT,package=None,Core_qualification=None)for m in('B0','B1')}
    policies['B1'].update(package=package,Core_qualification=request['Core_qualification'])
    for m in('B0','B1'):pair.policy_contract(policies[m],g['checkpoint_files'],expected_global_step=step)
    manifest=json.loads((Path(package['path'])/'PACKAGE_MANIFEST.json').read_bytes())
    seed=manifest['recipe']['seed'];prefix=f'tl-student-s{seed}-step{step}-pkg{package["sha256"][:12]}'
    task.need(root.name==f'task-learning-student-dev-seed{seed}-step{step}-pkg{package["sha256"][:12]}',
        'ONE_CANONICAL_PANEL_PER_ACTUAL_CANDIDATE')
    panel=make_panel(core['source_sha'],step,seed,policies,request['scenes'],prefix)
    panel['registration']=dict(schema=SCHEMA,native_base=request['native_base'],gpu_base=request['gpu_base'],
        Core_qualification=request['Core_qualification'],source_root=str(source),source_files=core['source_files'],
        scenes=request['scenes'],scene_allowlist=n['scene_allowlist'],scene_allowlist_sha256=n['scene_allowlist_sha256'],
        memory_execution=core['memory_execution'],memory_execution_reference=core['memory_execution_reference'])
    campaign=Path(n['campaign_root']);key=root.name.upper()+'-PANEL';j=Journal(campaign/'control/journal',campaign.name)
    with closing(OwnerLock(campaign/'OWNER.lock')):
        task.need(not root.exists()and not j.path(key).exists(),'UNKNOWN_DEV_PANEL_NEVER_RECREATE')
        def effect():
            root.mkdir();(root/'control').mkdir();write(root/'PANEL.private.json',panel)
            write(root/'control/LEASE.json',dict(status='SEALED_NONCOMPUTE_PANEL',
                panel_sha256=digest(root/'PANEL.private.json'),planned_conditions=80,created_epoch=time.time()))
            reg,pin=snapshot(n['registry'])
            task.need(not any(v.get('path')==str(root)for v in reg['entries']),'FRESH_DEV_PANEL_REGISTRY')
            reg['entries'].append(dict(path=str(root),role=key,scope=SCHEMA,owner=n['owner'],
                source_sha=core['source_sha'],local_lease=str(root/'control/LEASE.json'),executor_node=n['host']))
            once(Path(n['registry']),reg,key+'-REGISTRY',expected_before_sha256=pin)
            return dict(status='FIXED_FULL_DEV80_PANEL_SEALED_NO_WORKER_STARTED',panel=descriptor(root/'PANEL.private.json'))
        return j.once(key,dict(package=package,source_sha=core['source_sha']),effect)


def group_input(request,panel):
    reg=panel['registration'];span=request['assignment_slice']
    inp=dict(schema=batch.CONTINUATION_SCHEMA,source_sha=panel['source_sha'],
        panel=request['panel']['path'],panel_sha256=request['panel']['sha256'],assignment_slice=span,
        population_predecessor=request.get('population_predecessor'),**{k:reg[k]for k in(
            'source_root','source_files','native_base','gpu_base','scenes','scene_allowlist',
            'scene_allowlist_sha256','memory_execution','memory_execution_reference')})
    if panel.get('task_learning_role')==task.CORE_LOCAL_ROLE:
        from .task_learning_local_policy_v1 import INPUT_SCHEMA
        inp['schema']=INPUT_SCHEMA
    if panel.get('task_learning_role')==task.ENDPOINT_ROLE:
        from .task_learning_endpoint_native_v3 import INPUT_SCHEMA
        inp['schema']=INPUT_SCHEMA
    if panel.get('task_learning_role')==task.MATCHED_ROLE:
        from .task_learning_matched_native_v3_1 import INPUT_SCHEMA
        inp['schema']=INPUT_SCHEMA
    if panel.get('task_learning_role')==task.CONIC_ROLE:
        from .task_learning_conic_native_v3 import INPUT_SCHEMA
        inp['schema']=INPUT_SCHEMA
        inp['conic_solver_dependency']=reg['conic_solver_dependency']
    batch.selection(inp,panel)
    for label in('pilot1','fusion'):
        descriptor_=request[label+'_binding'];binding=pinned(descriptor_)
        lease=json.loads((Path(descriptor_['path']).parent/'LEASE.json').read_bytes())
        task.need(lease.get('status')=='ACTIVE'and time.time()+1800<binding['deadline_epoch'],
            'FRESH_FINITE_DEV_SCOPE_'+label)
        inp[label+'_binding']=descriptor_['path'];inp[label+'_binding_sha256']=descriptor_['sha256']
    wall=request['max_wall_s']
    task.need(type(wall)is int and 1800<=wall<=6*3600,'FINITE_ONE_DEV_GROUP_WINDOW')
    end=min(time.time()+wall,*[pinned(request[k+'_binding'])['deadline_epoch']-120 for k in('pilot1','fusion')])
    inp['deadline_epoch']=end;return inp


def register_group(request):
    from scripts.pi0292.durable import OwnerLock,write
    from scripts.pi0294.e1.journal import Journal
    from scripts.pi0294.night10.native_relay import snapshot,once
    from .pilot1_scope import validate
    panel=pinned(request['panel']);inp=group_input(request,panel);root=Path(request['root'])
    n=closed_base(inp['native_base']);closed_base(inp['gpu_base']);campaign=Path(n['campaign_root'])
    actual_site(campaign)
    task.need(root.parent==Path(request['panel']['path']).parent and root.parent.resolve()==root.parent,
        'GROUP_UNDER_ORIGINAL_SEALED_PANEL')
    task.need(root.name==f'{root.parent.name}-group-{inp["assignment_slice"][0]:03d}-{inp["assignment_slice"][1]:03d}',
        'CANONICAL_UNEXECUTED_DEV_GROUP_SLICE')
    key=root.name.upper()+'-REGISTER';j=Journal(campaign/'control/journal',campaign.name)
    with closing(OwnerLock(campaign/'OWNER.lock')):
        task.need(not root.exists()and not j.path(key).exists(),'UNKNOWN_DEV_GROUP_NEVER_RECREATE')
        def effect():
            root.mkdir()
            for name in('control','runs','logs','tmp','cache'):(root/name).mkdir()
            write(root/'INPUT.private.json',inp)
            write(root/'control/LEASE.json',dict(status='ACTIVE',run_uuid=root.name,source_sha=inp['source_sha'],
                input_sha256=digest(root/'INPUT.private.json'),deadline_epoch=inp['deadline_epoch']))
            reg,pin=snapshot(n['registry'])
            task.need(not any(v.get('path')==str(root)for v in reg['entries']),'FRESH_DEV_GROUP_REGISTRY')
            reg['entries'].append(dict(path=str(root),role=root.name.upper(),scope=n['scope'],owner=n['owner'],
                source_sha=inp['source_sha'],pilot1_binding=inp['pilot1_binding'],
                local_lease=str(root/'control/LEASE.json'),executor_node=n['host']))
            once(Path(n['registry']),reg,key+'-REGISTRY',expected_before_sha256=pin)
            validate(dict(n,registered_root=str(root),source_sha=inp['source_sha'],deadline_epoch=inp['deadline_epoch'],
                pilot1_binding=inp['pilot1_binding'],pilot1_binding_sha256=inp['pilot1_binding_sha256']))
            return dict(status='DEV_GROUP_REGISTERED_NOT_DISPATCHED',root=str(root),input=descriptor(root/'INPUT.private.json'),
                assignment_slice=inp['assignment_slice'],deadline_epoch=inp['deadline_epoch'])
        return j.once(key,dict(panel=request['panel'],assignment_slice=inp['assignment_slice']),effect)


def dispatch_group(input_path):
    from scripts.pi0292.durable import write
    from scripts.pi0294.e1.journal import Journal
    inp=json.loads(Path(input_path).read_bytes());root=Path(input_path).parent;panel=pinned(dict(path=inp['panel'],sha256=inp['panel_sha256']))
    batch.selection(inp,panel);n=pinned(inp['native_base']);g=pinned(inp['gpu_base']);actual_site(n['campaign_root'])
    lease=json.loads((root/'control/LEASE.json').read_bytes());j=Journal(Path(n['campaign_root'])/'control/journal',Path(n['campaign_root']).name)
    key=root.name.upper()+'-LAUNCH'
    task.need(lease.get('status')=='ACTIVE'and lease.get('input_sha256')==digest(input_path)
        and not(root/'SUPERVISOR.private.json').exists()and not j.path(key).exists(),
        'REGISTERED_UNSTARTED_DEV_GROUP_WITHOUT_UNKNOWN_DISPATCH')
    q=subprocess.run(['systemctl','--user','show','wm02-'+root.name,'--property=LoadState,MainPID'],
        capture_output=True,text=True,timeout=10,check=True)
    task.need('LoadState=not-found'in q.stdout and 'MainPID=0'in q.stdout,'NO_EXISTING_DEV_GROUP_UNIT')
    remaining=int(inp['deadline_epoch']-time.time());task.need(remaining>1800,'TIME_FOR_FINITE_DEV_GROUP')
    actual=batch.resources(g);write(root/'ACTUAL_CAPACITY_BEFORE_DISPATCH.private.json',actual)
    argv=['systemd-run','--user','--unit=wm02-'+root.name,'--property=Type=exec','--property=RemainAfterExit=yes',
        '--property=Restart=no','--property=RuntimeMaxSec='+str(remaining+90),'--property=TimeoutStopSec=90',
        '--property=AllowedCPUs=0-3,8-11','--property=CPUQuota=800%','--property=MemoryMax=40G',
        '--property=MemorySwapMax=0','--property=StandardOutput=append:'+str(root/'logs/out.log'),
        '--property=StandardError=append:'+str(root/'logs/err.log'),'--working-directory='+inp['source_root'],
        '/usr/bin/env','-u','LD_PRELOAD','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1',
        'PYTHONPATH='+inp['source_root']+'/src:'+inp['source_root'],n['python'],'-B','-m',
        ('scripts.wm02.task_learning_core_matched_panel_v3_1'if inp.get('schema')=='CORE_MATCHED128_SMOOTH_GROUP_INPUT_V3'else
         'scripts.wm02.task_learning_core_endpoint_panel_v3'if inp.get('schema')=='CORE_C500_ENDPOINT_BASIS_GROUP_INPUT_V3'else
         'scripts.wm02.task_learning_core_conic_panel_v3'if inp.get('schema')=='CORE_C500_EXPLICIT_CONIC_GROUP_INPUT_V3'else
         'scripts.wm02.task_learning_core_local_panel_v1'if inp.get('schema')=='TASK_LEARNING_CORE_LOCAL_GROUP_INPUT_V1'
         else'scripts.wm02.task_learning_student_dev_v1'),'--input',str(input_path)]
    def effect():
        write(root/'LAUNCH_INTENT.private.json',dict(argv=argv,automatic_retry=False))
        q=subprocess.run(argv,capture_output=True,text=True,timeout=20)
        receipt=dict(exit_code=q.returncode,stdout=q.stdout,stderr=q.stderr)
        write(root/'DISPATCH.private.json',receipt)
        task.need(q.returncode==0,'UNKNOWN_DEV_LAUNCH_RECONCILE_NEVER_REPLAY');return receipt
    return j.once(key,dict(input_sha256=digest(input_path)),effect)
