"""Run ADAPT zero-update qualification or its bounded phase25 on one PBS A40."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from go2arx5_vla.sla_adapter import adapt5k_contract as contract
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from scripts.pi0292.durable import OwnerLock, write
from scripts.sla.evaluate_adapt5k import restore_proof
from scripts.sla.scale20k_portable_endpoint import local_custody
from scripts.sla.scale20k_vanda_offline import read_pbs_with_auth_retry
from scripts.wm02.multisource_d4_vanda import GIB, actual_gpu, pbs_job, pin


WALL = 10800
SCHEMA = 'SLA_ADAPT5K_VANDA_BOUNDED_PHASE_ACTUAL_V1'


def require_previous_qualification(config):
    prior = read_pin(config['previous_scope'])
    job = prior['PBS_JOBID']

    def finished_job():
        text = subprocess.check_output(['/opt/pbs/bin/qstat','-xf','-F','json',job],
            stderr=subprocess.PIPE,text=True,timeout=20)
        return json.loads(text)['Jobs'][job]

    actual = read_pbs_with_auth_retry(finished_job)
    if (prior['schema'] != SCHEMA or prior['role'] != 'qualify' or prior['exit_code']
            or prior['new_updates'] or prior['qualification_gradient_windows'] != contract.BATCH
            or prior['parent_checkpoint'] != config['parent_checkpoint']
            or prior['data_contract'] != config['data_contract'] or prior['source_sha'] != config['source_sha']
            or actual['job_state'] != 'F' or actual['Exit_status'] != 0
            or actual['Job_Owner'].split('@')[0] != config['user'] or actual['project'] != config['project']):
        raise ValueError('ADAPT_FIRST_25_REQUIRES_REAL_COLD_PHASE_ZERO_RECEIPTS')
    return prior


def pbs_site(config, root):
    job, window = read_pbs_with_auth_retry(pbs_job,config)
    deadline = window['deadline_epoch']
    uuid,name,total,free = actual_gpu()
    fraction = .80
    if window['PBS_wall_s'] > WALL or not 1800 < deadline-time.time() <= WALL or free <= (fraction*total)+2*GIB:
        raise PermissionError('ADAPT5K_REQUIRES_A_REAL_RESERVED_FINITE_A40_WITH_HEADROOM')
    cache = Path(os.environ['TMPDIR']).resolve(strict=True)/'sla-adapt5k-compile'
    cache.mkdir()
    paths = {k:str(Path(config[k]).resolve(strict=True)) for k in ('parent_params','dataset')}
    paths['compilation_cache_dir'] = str(cache)
    observed = root/'PBS_ALLOCATED.private.json'
    write(observed,dict(window,job_id=job,uid=os.getuid(),host=os.uname().nodename,project=config['project'],
        state='R',allocated_GPU_uuids=[uuid],GPU_free_bytes=free,PID=os.getpid(),observed_epoch=time.time()))
    site = root/'SITE.private.json'
    write(site,dict(schema='SLA_SINGLE_GPU_EXISTING_SITE_V2',host=os.uname().nodename,uid=os.getuid(),GPU_uuid=uuid,
        GPU_name=name,allocation_deadline_epoch=window['allocation_deadline_epoch'],
        allocator_limit_bytes=math.ceil(fraction*total),GPU_headroom_bytes=2*GIB,host_headroom_bytes=32*GIB,
        paths=paths,allocation=dict(kind='PBS',job_id=job,project=config['project'],actual_allocation=pin(observed))))
    env = dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,JAX_PLATFORMS='cuda',XLA_FLAGS=config['XLA_FLAGS'],
        XLA_PYTHON_CLIENT_MEM_FRACTION=str(fraction),XLA_PYTHON_CLIENT_PREALLOCATE='false',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    common = dict(plan=contract.PLAN,source_sha=config['source_sha'],site_binding=pin(site),GPU_uuid=uuid,
        JAX_memory_fraction=fraction,deadline_epoch=deadline-60,max_walltime_s=WALL,max_gpu_hours=WALL/3600,
        save_and_exit_on_limit=True,model_service_timeout_s=1800,RPC_timeout_s=1800,save_reserve_s=300,
        STOP_path=str(root/'STOP.json'),hardware_allowed=False,physics=False,
        neural_owner_lock=str(root/'DEVICE_WORKER.lock'),**paths)
    return common,env,job


def run(config):
    root = Path(config['root']).resolve(strict=True)
    source = Path(__file__).resolve().parents[2]
    lineage = Path(config['lineage_root']).resolve(strict=True)
    role = config['role']
    if (root.stat().st_uid != os.getuid() or lineage.stat().st_uid != os.getuid() or root.parent != lineage
            or (root/'RESULT.private.json').exists() or (source/'SOURCE_SHA.private').read_text().strip() != config['source_sha']
            or config['plan'] != contract.PLAN or role not in ('qualify','prefix25')):
        raise PermissionError('ADAPT5K_NEW_FINITE_PBS_ROLE_AND_SINGLE_LINEAGE_REQUIRED')
    result = dict(schema=SCHEMA,role=role,source_sha=config['source_sha'],
        PBS_JOBID=os.environ.get('PBS_JOBID'),data_contract=config['data_contract'],
        parent_checkpoint=config['parent_checkpoint'],new_updates=0,model_generations=0,native_requests=0,
        attempted_roles=[],completed_roles={},status='ADAPT_PBS_PREFLIGHT',exit_code=1)
    lock = OwnerLock(lineage/'LINEAGE_OWNER.lock')
    stop = root/'STOP.json'
    current = None

    def stopping(*_):
        if not stop.exists():
            write(stop,dict(reason='OPERATOR_PBS_STOP'))

    signal.signal(signal.SIGTERM,stopping)
    signal.signal(signal.SIGINT,stopping)
    try:
        if (lineage/'STOP.json').exists() or stop.exists():
            raise PermissionError('ADAPT5K_LINEAGE_OR_ROLE_STOPPED')
        custody_pin = config['MIX_parent_custody']
        custody = read_pin(custody_pin)
        if (custody != local_custody(Path(custody_pin['path']).parent,custody['transfer'])
                or custody['checkpoint'] != config['parent_checkpoint'] or custody['step'] != config['parent_step']
                or custody['training_spec'] != config['parent_training_spec']):
            raise ValueError('ADAPT5K_NEEDS_BYTE_COMPLETE_ORIGINAL_FORMAL_MIX_PARENT')
        assets = read_pin(config['data_contract'])
        custody = read_pin(assets['R200_custody'])
        if custody['train_episodes'] != 200 or custody['train_windows'] != 210398 or custody['SOURCE_TEST_access']:
            raise ValueError('ADAPT5K_REFUSES_PARTIAL_R200_CUSTODY')
        training = read_pin(assets['train_input'])
        evidence = read_pin(assets['eval_input'])
        if (training['data']['robodojo'] != custody['data']
                or training['data']['target'] != evidence['target_D1']
                or training['parent_checkpoint_sha256'] != config['parent_checkpoint']['sha256']
                or evidence['D1_freeze_sha256'] != config['D1_freeze_sha256']):
            raise ValueError('ADAPT5K_SITE_TRAIN_AND_EVAL_ASSETS_REQUIRE_THE_SAME_FROZEN_D1')
        cpu = read_pin(assets['train_proof'])
        if (cpu['core'] != cpu['vanda'] or cpu['vanda']['parent_checkpoint_sha256'] != config['parent_checkpoint']['sha256']
                or read_pin(evidence['panel'])['old_TEST_access'] or sum(len(x) for x in cpu['vanda']['inputs'].values()) != 22):
            raise ValueError('ADAPT5K_SITE_TRAIN_INPUTS_REQUIRE_ACTUAL_CROSS_SITE_READER_PROOF')
        old = None if role == 'qualify' else require_previous_qualification(config)
        common,env,job = pbs_site(config,root)
        convention = dict(common,parent_manifest=config['parent_manifest'],tokenizer=config['tokenizer'])
        train = dict(convention,data=training['data'],train_data_proof=assets['train_proof'],
            target_D1_sha256=config['D1_freeze_sha256'],parent_checkpoint=config['parent_checkpoint'],
            parent_training_spec=config['parent_training_spec'],parent_step=config['parent_step'],
            stage_start=training['stage_start'],stage_stop=training['stage_stop'],target_seed=training['target_seed'],
            stage_bank_sha256=cpu['vanda']['bank_digests'],microbatch=dict.fromkeys(contract.DOMAINS,1))

        def execute(name,module,worker_mode,spec):
            nonlocal current
            if (lineage/'STOP.json').exists() or stop.exists() or common['deadline_epoch']-time.time() < 360:
                raise TimeoutError('ADAPT5K_STOP_BEFORE_NEXT_BOUNDED_ROLE')
            role_dir=root/name
            role_dir.mkdir()
            card=root/(name+'.SPEC.private.json')
            write(card,dict(spec,registered_root=str(role_dir)))
            result['attempted_roles'].append(name)
            current=name
            write(root/'PROGRESS.private.json',dict(current_role=name,complete=result['completed_roles'],epoch=time.time()))
            with (root/(name+'.log')).open('x') as log:
                proc = subprocess.run([sys.executable,'-B','-m',module,worker_mode,str(card),pin(card)['sha256']],
                    cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,
                    timeout=max(1,common['deadline_epoch']-time.time()))
            file=role_dir/'RESULT.private.json'
            if not file.is_file():
                raise RuntimeError('ADAPT5K_EFFECT_UNRESOLVED_NO_REPLAY:'+name)
            actual=read_pin(pin(file))
            result['completed_roles'][name]=pin(file)
            result['new_updates']+=actual['new_updates']
            result['model_generations']+=actual.get('model_generations',actual.get('complete_H40_generations',0))
            current=None
            if proc.returncode or actual['exit_code']:
                result.update(status=actual['status'],failure=actual.get('failure'),exit_code=proc.returncode or actual['exit_code'])
                return None
            return actual

        if old is None:
            terminal='qualify'
            measured=execute(terminal,'scripts.sla.train_adapt5k','qualify',
                dict(train,previous_train_or_qualification=None))
            if measured is None:
                return result['exit_code']
            result['qualification_gradient_windows']=measured['gradient_inputs']
            eval_roles=('formal','formal-restore')
            panel=read_pin(evidence['panel'])
            caps=(288+len(panel['phase_train']),48)
        else:
            terminal='train25'
            measured=execute(terminal,'scripts.sla.train_adapt5k','train',dict(train,phase_stop_limit=25,
                previous_train_or_qualification=old['completed_roles']['qualify'],
                initial_output_and_restore_actual=old['output_restore_proof']))
            if measured is None:
                return result['exit_code']
            verified=execute('state_restore25','scripts.sla.train_adapt5k','restore25',
                dict(train,previous_train_or_qualification=result['completed_roles']['train25']))
            if verified is None:
                return result['exit_code']
            eval_roles=('phase25-before','phase25-restore')
            caps=(24,24)
        output_input=dict(convention,**evidence,sealed_training_actual=result['completed_roles'][terminal],
                          checkpoint=measured['checkpoint'])
        first,second=eval_roles
        ordinary=execute(first,'scripts.sla.evaluate_adapt5k',first,dict(output_input,generation_cap=caps[0]))
        if ordinary is None:
            return result['exit_code']
        restored=execute(second,'scripts.sla.evaluate_adapt5k',second,
            dict(output_input,before=result['completed_roles'][first],generation_cap=caps[1]))
        if restored is None:
            return result['exit_code']
        proof=root/'OUTPUT_RESTORE_ACTUAL.private.json'
        write(proof,restore_proof(ordinary,restored))
        result.update(output_restore_proof=pin(proof),status='ADAPT_PBS_FINITE_ROLE_COMPLETE',exit_code=0)
        return 0
    except (Exception,KeyboardInterrupt) as error:
        result.update(status='ADAPT_PBS_ROLE_RECONCILIATION_REQUIRED',unresolved_role=current,
            failure=f'{type(error).__name__}: {error}',exit_code=1)
        raise
    finally:
        write(root/'RESULT.private.json',result)
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('sha256')
    args = parser.parse_args()
    raise SystemExit(run(read_pin(dict(path=args.config,sha256=args.sha256))))
