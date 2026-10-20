"""Saved-data closeout only; never reopen collection or promote a failed bank."""
import os
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np

from scripts.pi0292.durable import OwnerLock,read,write
from scripts.pi0294.fd1 import put,sha
from scripts.pi0294.scene_cards import digest
from scripts.pi0294.e1.scope import STATS
from .bank import parent_files
from .export import event_index
from .mixed_batch import qualify_all
from .plan import PLAN_RELEASE
from .scope import approval,sealed_file

SCHEMA='PI0294_DATA_COV1_SAVED_CLOSEOUT_AUDIT_V1'
ROLE='PI0294_DATA_COV1_SAVED_CLOSEOUT_AUDIT'
FAILED='CLOSED_FIRST_FOUR_QUALIFICATION_GATE_NOT_MET'


def authority(adoption,base,stop,limits,*,now):
    approval(adoption)
    if (base.get('status')!=FAILED or stop.get('reason')!='FIRST_FOUR_QUALIFICATION_GATE_NOT_MET'
            or base.get('unknown_effects') or base.get('episode_starts')!=4
            or any(base.get(k)!=0 for k in ('CPU_reserved_s','local_CPU_reserved_s','constructors_reserved',
                'episode_starts_reserved','ticks_reserved'))
            or limits.get('T0_epoch')!=adoption['T0_epoch']
            or not now<limits.get('deadline_epoch',0)<=adoption['T0_epoch']+24*3600
            or type(limits.get('CPU_s')) is not int or not 30<limits['CPU_s']<=300
            or limits.get('local_control_CPU_upper_s')!=300
            or base['CPU_s']+limits['CPU_s']+180>32*3600
            or base['local_CPU_s']+300>4*3600
            or any(limits.get(k)!=0 for k in ('episodes','constructors','physics_ticks','neural_queries','updates','PBS'))
            or limits.get('output_bytes')!=64*1024**2 or len(limits.get('cpus',[]))!=1
            or set(limits['cpus'])&set(adoption['PV1_reserved_CPUs'])):
        raise PermissionError('DATA_COV1_CLOSEOUT_ONLY_SAME_APPROVED_CLOCK_AND_REMAINING_CPU_NO_NEW_SCIENCE')


def validate(root):
    root=Path(root)
    for name in ('RUN_MANIFEST.json','OUTPUT_BINDING.private.json','INPUT_BINDINGS.private.json'):
        sealed_file(root/name)
    m=read(root/'RUN_MANIFEST.json');b=read(root/'OUTPUT_BINDING.private.json');inputs=read(root/'INPUT_BINDINGS.private.json')
    if (m.get('schema')!=SCHEMA or m.get('scope')!=SCHEMA or m.get('owner')!='EXISTING_LOCAL_PC'
            or m.get('run_uuid')!=root.name or m['execution']['profile']!='4080-cpu-data'
            or m['execution']['uid']!=os.getuid() or m['execution']['host']!=os.uname().nodename
            or root.is_symlink() or root.resolve()!=Path(b['run_root']).resolve()
            or root.parent.resolve()!=Path(b['registered_parent']).resolve()
            or b['uid']!=os.getuid() or b['role']!=ROLE
            or sha(root/'OUTPUT_BINDING.private.json')!=m['execution']['output_binding_sha256']
            or sha(root/'INPUT_BINDINGS.private.json')!=m['execution']['input_binding_sha256']
            or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or 'PBS_JOBID' in os.environ):
        raise PermissionError('DATA_COV1_CLOSEOUT_REGISTERED_CPU_HOST_MANIFEST_AND_BINDINGS')
    old=Path(inputs['closed_source_root']);ob=read(old/'OUTPUT_BINDING.private.json')
    if (root.parent.parent.resolve()!=old.resolve() or old.is_symlink()
            or read(ob['lease'])['status']!='CLOSED' or sha(ob['lease'])!=inputs['source_lease_sha256']
            or sha(old/'evidence/DATA_COV1_TERMINAL.json')!=inputs['source_terminal_sha256']
            or sha(old/'RESOURCE_LEDGER.json')!=inputs['source_ledger_sha256']
            or sha(old/'STOP.json')!=inputs['source_stop_sha256']):
        raise PermissionError('DATA_COV1_ORIGINAL_CLOSED_RECEIPTS_AND_STOP_MUST_REMAIN_UNCHANGED')
    a=read(old/'DATA_COV1_ADOPTION_AND_SCOPE.json');base=read(old/'RESOURCE_LEDGER.json')
    if (m['authority']['protocol_release']!=PLAN_RELEASE
            or m['authority']['adoption_sha256']!=sha(old/'DATA_COV1_ADOPTION_AND_SCOPE.json')
            or m['authority']['approval_ref']!=a['approval_ref']):
        raise PermissionError('DATA_COV1_CLOSEOUT_NO_NEW_OR_TRANSFERRED_AUTHORITY')
    authority(a,base,read(old/'STOP.json'),m['limits'],now=time.time())
    sealed_file(old/'plans/statistics.json',STATS)
    sealed_file(old/'CPU_DATA_INPUT_BINDINGS.private.json',inputs['source_CPU_inputs_sha256'])
    prior_inputs=read(old/'CPU_DATA_INPUT_BINDINGS.private.json')
    if any(inputs[k]!=prior_inputs[k] for k in ('official_source','cosmos')):
        raise PermissionError('DATA_COV1_SAME_FROZEN_OFFICIAL_PROCESSOR_AND_LOCAL_TOKENIZER_ASSETS')
    lease=read(b['lease']);sealed_file(b['lease'])
    if (lease['status']!='ACTIVE' or lease['run_uuid']!=root.name or lease['owner']!='EXISTING_LOCAL_PC'
            or lease['deadline_epoch']!=m['limits']['deadline_epoch']):
        raise PermissionError('DATA_COV1_ONLY_NEW_READONLY_CLOSEOUT_LEASE_NOT_OLD_LEASE_REOPEN')
    reg=read(b['registry'])['entries'];entries=[e for e in reg if e.get('role')==ROLE]
    if len(entries)!=1 or Path(entries[0]['path']).resolve()!=root.resolve():
        raise PermissionError('DATA_COV1_ONE_REGISTERED_CLOSEOUT_AUDITOR')
    infra=Path(b['registry']).parent
    if any(sha(infra/n)!=m['execution'][k] for n,k in (('enter.sh','entry_sha256'),('doctor.py','doctor_sha256'))):
        raise ValueError('DATA_COV1_SAME_DEPLOYED_ENTRY')
    code=root/m['code']['capsule_relative'];seal=root/m['code']['source_seal_relative']
    sealed_file(seal,m['code']['source_seal_sha256']);s=read(seal)
    if (s['source_sha']!=m['code']['vla_commit'] or code.is_symlink()
            or not code.resolve().is_relative_to(root.resolve())
            or Path(__file__).resolve()!=(code/'scripts/pi0294/data_cov1/closeout.py').resolve()
            or os.environ.get('DATA_COV1_CLOSEOUT_MANIFEST')!=str(root/'RUN_MANIFEST.json')):
        raise PermissionError('DATA_COV1_ACTUAL_COLD_CLOSEOUT_CAPSULE')
    for n,h in s['files'].items():
        p=code/n
        if Path(n).is_absolute() or not p.resolve().is_relative_to(code.resolve()):
            raise PermissionError('DATA_COV1_CLOSEOUT_SOURCE_PATH_ESCAPE')
        sealed_file(p,h)
    if len(a['frozen_numerical_files'])!=15:
        raise PermissionError('DATA_COV1_ORIGINAL_15_NUMERICAL_FILES')
    for n,h in a['frozen_numerical_files'].items():sealed_file(code/n,h)
    assets=read(old/'plans/ASSET_IDENTITY.private.json')['absolute_files']
    if len(assets)!=26 or any(sha(p)!=h for p,h in assets.items()):
        raise ValueError('DATA_COV1_ORIGINAL_26_NATIVE_ASSETS_UNCHANGED')
    proof=read(root/'plans/SOURCE_PUBLICATION_AND_CI.json')
    if (proof['source_sha']!=m['code']['vla_commit'] or proof['CI_head_sha']!=m['code']['vla_commit']
            or proof['CI_status']!='completed' or proof['CI_conclusion']!='success'
            or not proof['files_exact_readback']):
        raise PermissionError('DATA_COV1_CLOSEOUT_EXACT_PUBLICATION_AND_CI')
    sealed_file(root/'plans/SOURCE_PUBLICATION_AND_CI.json',m['code']['publication_sha256'])
    sealed_file(root/'plans/WORK_PLAN.private.json',m['data']['plan_sha256'])
    plan=read(root/'plans/WORK_PLAN.private.json')
    if (plan['operation']!='SAVED_PARENT_BATCH_AND_EVENT_CLOSEOUT' or len(plan['parents'])!=2
            or len(plan['event_parents'])!=3
            or len({p['id'] for p in plan['event_parents']})!=3
            or not {p['id'] for p in plan['parents']}<={p['id'] for p in plan['event_parents']}):
        raise PermissionError('DATA_COV1_EXACT_TWO_QUALIFIED_PARENTS_NOT_A_BANK_OR_NEW_DATA')
    if not set(m['limits']['cpus'])<=os.sched_getaffinity(0):
        raise PermissionError('DATA_COV1_EXISTING_DISJOINT_CPU')
    return m,inputs,plan,base


def closed_export(old,binding):
    folder=Path(binding['export_root']);child=folder.parent.parent
    if child.parent.resolve()!=(old/'children').resolve() or folder.is_symlink():
        raise PermissionError('DATA_COV1_SAME_CLOSED_CAMPAIGN_EXPORT_ONLY')
    b=read(child/'OUTPUT_BINDING.private.json');l=read(child/'RESOURCE_LEDGER.json')
    if (read(b['lease'])['status']!='CLOSED' or l['status']!='TERMINAL_DATA_COV1_SAVED_EXPORT'
            or l['unknown_effects'] or b['uid']!=os.getuid()):
        raise PermissionError('DATA_COV1_IMMUTABLE_TERMINAL_EXPORT_ONLY')
    sealed_file(folder/'EXPORT_RECEIPT.json',binding['export_receipt_sha256'])
    return read(folder/'EXPORT_RECEIPT.json')


def run(root):
    from scripts.pi0291_arx5_data.native_dataset import cpu_guard
    from scripts.pi0292.train_adapter import build_processor
    began=time.process_time();children=resource.getrusage(resource.RUSAGE_CHILDREN)
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_CPU,(300,300))
    root=Path(root);lock=OwnerLock(root/'CPU_CLOSEOUT_OWNER.lock')
    m,inputs,plan,base=validate(root);old=Path(inputs['closed_source_root'])
    ledger=read(root/'RESOURCE_LEDGER.json')
    if ledger['status']!='RESERVED' or ledger['unknown_effects']:
        lock.close();raise PermissionError('DATA_COV1_NO_CLOSEOUT_RESTART_OR_UNKNOWN_RETRY')
    os.sched_setaffinity(0,set(m['limits']['cpus']));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    resource.setrlimit(resource.RLIMIT_CPU,(m['limits']['CPU_s'],m['limits']['CPU_s']))
    failure=None
    def guard():
        if (time.time()>=m['limits']['deadline_epoch'] or time.process_time()-began>=m['limits']['CPU_s']-20
                or (root/'STOP.json').exists() or sha(old/'STOP.json')!=inputs['source_stop_sha256']):
            raise PermissionError('DATA_COV1_CLOSEOUT_ORIGINAL_CLOCK_CPU_OR_STOP')
        if sum(p.stat().st_size for p in (root/'evidence').glob('*') if p.is_file())>m['limits']['output_bytes']:
            raise PermissionError('DATA_COV1_CLOSEOUT_SMALL_EVIDENCE_ONLY')
    def expire(*args):raise TimeoutError('DATA_COV1_CLOSEOUT_WALL_BOUND')
    try:
        signal.signal(signal.SIGALRM,expire);signal.alarm(max(1,int(m['limits']['deadline_epoch']-time.time())))
        cpu_guard();ledger['status']='RUNNING';write(root/'RESOURCE_LEDGER.json',ledger)
        members=0
        for binding in plan['parents']:
            row=closed_export(old,binding);members+=len(parent_files(binding['export_root'],row,guard))
        processor=build_processor(inputs['official_source'],inputs['cosmos'],read(old/'plans/statistics.json'))
        batch=qualify_all(plan['parents'],processor,inputs['official_source'],inputs['cosmos'],guard)
        put(root/'evidence/ACTUAL_CROSS_PARENT_BATCH.json',batch)
        events={}
        for binding in plan['event_parents']:
            guard();row=closed_export(old,binding);folder=Path(binding['source_episode'])
            cb=read(folder.parent.parent/'OUTPUT_BINDING.private.json')
            if (folder.parent.parent.parent.resolve()!=(old/'children').resolve()
                    or read(cb['lease'])['status']!='CLOSED'
                    or sha(folder/'FILE_MANIFEST.json')!=row['source_raw_manifest_sha256']
                    or sha(folder/'physics.npz')!=read(folder/'SCORE.json')['physics_sha256']):
                raise PermissionError('DATA_COV1_CLOSED_REAL_PHYSICS_EVENTS_ONLY')
            card=read(folder/'CARD.json')
            if digest(card)!=row['source_card_sha256'] or card['id']!=row['id']:
                raise ValueError('DATA_COV1_REAL_SAVED_CARD_NOT_REGENERATED_GEOMETRY')
            with np.load(folder/'physics.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
            e=event_index(data,card);e['source_raw_manifest_sha256']=row['source_raw_manifest_sha256']
            put(root/'evidence'/('EVENTS-'+card['id']+'.json'),e);events[card['id']]=e
        import torch
        if (torch.cuda.is_initialized() or 'mujoco' in sys.modules or any(n.startswith('gr00t')
                and any(k in n for k in ('modeling_','.backbone.','.action_head.')) for n in sys.modules)):
            raise PermissionError('DATA_COV1_NO_PHYSICS_MODULE_CUDA_OR_MODEL_IN_CLOSEOUT')
        result=dict(status='PASS_REQUIRED_SAVED_PARENT_BATCH_AND_EVENT_CLOSEOUT_NOT_STAGE_READY',
            cross_parent_batch=batch,verified_numeric_and_RGB_members=members,
            event_indices=events,source_terminal_sha256=inputs['source_terminal_sha256'],
            original_collection_and_stop_and_outcomes_unchanged=True,first_four_gate_pass=False,
            stage_bank_READY=False,HF_delivery_claim=False,original_T0_epoch=m['limits']['T0_epoch'],
            source_CPU_s=base['CPU_s'],source_local_CPU_s=base['local_CPU_s'],
            new_episodes=0,new_constructors=0,new_physics=0,new_neural_queries=0,new_updates=0,new_PBS=0)
        guard();put(root/'evidence/CLOSEOUT_RESULT.json',result)
        return dict(status=result['status'],parents=2,real_samples=12,event_parents=len(events))
    except BaseException as exc:
        failure=type(exc).__name__+': '+str(exc);raise
    finally:
        signal.alarm(0);after=resource.getrusage(resource.RUSAGE_CHILDREN)
        ledger.update(status='FAILED_PRESERVED' if failure else 'TERMINAL_SAVED_CLOSEOUT_AUDIT',
            CPU_s=time.process_time()-began+after.ru_utime+after.ru_stime-children.ru_utime-children.ru_stime,
            CPU_reserved_s=0,error=failure)
        write(root/'RESOURCE_LEDGER.json',ledger);lock.close()
