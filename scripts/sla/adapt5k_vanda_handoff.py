"""Release the next ADAPT segment or an independent formal Eval on an actual PBS grant."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from go2arx5_vla.sla_adapter import adapt5k_checkpoint as checkpoint, adapt5k_contract as contract
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from scripts.pi0292.durable import OwnerLock, write
from scripts.sla import adapt5k_curriculum
from scripts.sla.adapt5k_vanda import SCHEMA, pbs_history, pbs_site, site_inputs
from scripts.sla.evaluate_adapt5k import restore_proof
from scripts.sla.scale20k_vanda_offline import read_pbs_with_auth_retry
from scripts.sla.train_adapt5k import first_nonzero_gate
from scripts.wm02.multisource_d4_vanda import pin


TRAIN_LIMITS = tuple(range(250, contract.UPDATES + 1, 250))
ORIGINAL_RUNTIME = '1432c8dc88c57adf39e8fa363f5438c5e63aa8fd'
RUNTIME_MEMBERS = ('SOURCE_SHA.private', 'scripts/sla/train_adapt5k.py', 'scripts/sla/evaluate_adapt5k.py',
                   'src/go2arx5_vla/sla_adapter/adapt5k_checkpoint.py',
                   'src/go2arx5_vla/sla_adapter/adapt5k_training.py',
                   'src/go2arx5_vla/sla_adapter/adapt5k_contract.py')


def verify_train_result(training_pin, parent):
    actual = read_pin(training_pin)
    saved = actual['checkpoint']
    root, marker = checkpoint.read_complete(saved, actual['binding'])
    step, start = marker['phase_step'], actual['start_phase']
    stage = actual['binding']['stage_start']
    updates = step - start
    if ((actual['status'],actual['exit_code']) not in
            (('ADAPT_FIXED_SEGMENT_SAVED',0),('ADAPT_BOUNDED_SAVED_PREFIX',75))
            or actual['source_sha'] != ORIGINAL_RUNTIME or actual['binding']['source_revision'] != ORIGINAL_RUNTIME
            or not stage <= start < step <= actual['binding']['stage_stop'] or actual['end_phase'] != step
            or actual['binding']['parent_checkpoint_sha256'] != parent['sha256']
            or actual['new_updates'] != updates or actual['native_requests'] or actual['native_ticks']
            or not 4*updates <= actual['complete_H40_generations'] <= 4*(updates+1)
            or actual['optimization_visits'] != {d:n*updates for d,n in contract.BATCH.items()}
            or root != Path(training_pin['path']).parent/f'phase{step:05d}'):
        raise ValueError('ADAPT5K_HANDOFF_REQUIRES_A_COMPLETE_FORWARD_TRAINING_SEAL')
    ledger = root.parent/'UPDATES.private.jsonl'
    data = ledger.read_bytes()
    lines = data.splitlines()
    if not data.endswith(b'\n') or len(lines) != updates:
        raise ValueError('ADAPT5K_HANDOFF_REQUIRES_COMPLETE_DURABLE_UPDATE_LEDGER')
    rates = contract.phase_rates()
    for expected, line in enumerate(lines, start+1):
        row = json.loads(line)
        clocks = contract.progress(actual['binding']['parent_step'],expected,stage)
        if (any(row[key] != value for key,value in clocks.items()) or row['learning_rate'] != rates[expected-1]
                or not math.isfinite(row['grad_norm'])
                or row['domains']['target']['training_full_generations'] != 4
                or tuple(row['domains']['target']['auxiliary_slots']) != contract.auxiliary_slots(expected-1)):
            raise ValueError(f'ADAPT5K_HANDOFF_UPDATE_CLOCK_OR_AUX_CHANGED:{expected}')
    return actual, dict(step=step, checkpoint=saved, ledger=pin(ledger), progress=marker['progress'], arrays=140)


def cold_predecessor(spec, base, previous):
    job = spec['previous_job']
    actual = read_pbs_with_auth_retry(pbs_history,job)
    required = previous/'RESULT.private.json'
    result = read_pin(pin(required))
    if (actual['job_state'] != 'F' or actual.get('Exit_status') not in (0,75)
            or actual['Job_Owner'].split('@')[0] != base['user'] or actual['project'] != base['project']
            or result['schema'] != SCHEMA or result['PBS_JOBID'] != job or result['exit_code'] != actual['Exit_status']
            or result['data_contract'] != base['data_contract'] or result['parent_checkpoint'] != base['parent_checkpoint']
            or result['native_requests']):
        raise ValueError('ADAPT5K_HANDOFF_REQUIRES_THE_EXACT_COLD_PBS_TRAINING_OWNER')
    phase25 = result['role'] == 'prefix25'
    if phase25:
        if (result['source_sha'] != ORIGINAL_RUNTIME or result['status'] != 'ADAPT_PBS_FINITE_ROLE_COMPLETE'
                or result['new_updates'] != 25 or result['model_generations'] != 148):
            raise ValueError('ADAPT5K_PHASE25_MUST_COMPLETE_STATE_AND_OUTPUT_RESTORE')
        train = result['completed_roles']['train25']
        gates = dict(phase25_restore_actual=result['completed_roles']['state_restore25'],
                     phase25_output_restore_actual=result['output_restore_proof'])
        first_nonzero_gate(dict(previous_train_or_qualification=train,**gates))
    else:
        if result['role'] != 'train' or result['status'] not in ('ADAPT_FIXED_SEGMENT_SAVED','ADAPT_BOUNDED_SAVED_PREFIX'):
            raise ValueError('ADAPT5K_HANDOFF_REQUIRES_A_TRAINING_PREDECESSOR')
        train = result['completed_roles']['train']
        gates = {}
    detail, seal = verify_train_result(train,base['parent_checkpoint'])
    if result['new_updates'] != detail['new_updates'] or (phase25 and seal['step'] != 25):
        raise ValueError('ADAPT5K_HANDOFF_PBS_AND_WORKER_COUNTS_DIFFER')
    return train, detail, seal, gates


def boundary_evaluation_registered(spec, base, lineage):
    registration_pin = spec['independent_eval2000']
    registration_path = Path(registration_pin['path']).resolve(strict=True)
    registration = read_pin(registration_pin)
    job = registration['job'] if 'job' in registration else registration['released']['job']
    card = read_pin(registration['spec'])
    status = read_pbs_with_auth_retry(pbs_history,job)
    if (registration_path not in {
            lineage/'slaadp2kev-01'/'SUBMISSION.private.json',
            lineage/'slaadp2kev-02'/'SUBMISSION.private.json',
            }
            or Path(card['root']).resolve(strict=True) != registration_path.parent
            or Path(registration['spec']['path']).resolve(strict=True) != registration_path.with_name('DEPENDENCY.private.json')
            or card['job_id'] != job or card['role'] != 'eval' or card['step'] != contract.SWITCH_PHASE
            or card['previous_job'] != spec['previous_job'] or job == spec['job_id']
            or status['project'] != base['project'] or status['Job_Owner'].split('@')[0] != base['user']
            or status['Job_Name'] != 'slaadp2kev'
            or status['job_state'] not in ('H','Q','R','E','F')):
        raise ValueError('ADAPT5K_REQUIRES_THE_REGISTERED_INDEPENDENT_2000_EVAL_EDGE')


def run(spec):
    root, previous, lineage, runtime = (Path(spec[k]).resolve(strict=True) for k in ('root','previous','lineage','runtime'))
    source = Path(__file__).resolve().parents[2]
    role, limit = spec['role'], spec['step']
    stop = root/'STOP.json'
    if (spec['job_id'] != os.environ.get('PBS_JOBID') or root == previous or root.parent != lineage
            or previous.parent != lineage or root.stat().st_uid != os.getuid() or lineage.stat().st_uid != os.getuid()
            or (source/'SOURCE_SHA.private').read_text().strip() != spec['source_sha']
            or (root/'RESULT.private.json').exists() or (root/'ACTIVATED.private.json').exists()
            or role not in ('train','eval') or limit not in (TRAIN_LIMITS if role == 'train' else (500,2000,5000))):
        raise PermissionError('ADAPT5K_HANDOFF_REQUIRES_FRESH_BOUNDED_PBS_ROLE')
    stop_files = (stop, lineage/'STOP.json', previous/'STOP.json')
    lock_path = lineage/'LINEAGE_OWNER.lock' if role == 'train' else lineage/f'EVAL_PHASE{limit:05d}.lock'
    lock = OwnerLock(lock_path)
    result = dict(schema=SCHEMA, role=role, PBS_JOBID=spec['job_id'], source_sha=spec['source_sha'],
                  runtime_revision=ORIGINAL_RUNTIME, phase_limit=limit, previous_job=spec['previous_job'],
                  new_updates=0, model_generations=0, native_requests=0, native_ticks=0,
                  optimization_visits=dict.fromkeys(contract.DOMAINS,0), completed_roles={}, exit_code=1)

    def request_stop(*_):
        if not stop.exists():
            write(stop,dict(reason='OPERATOR_PBS_STOP'))

    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGINT,request_stop)
    try:
        if any(p.exists() for p in stop_files):
            raise PermissionError('ADAPT5K_HANDOFF_STOPPED')
        if (tuple(spec['runtime_members']) != RUNTIME_MEMBERS
                or (runtime/'SOURCE_SHA.private').read_text().strip() != ORIGINAL_RUNTIME
                or any(pin(runtime/name)['sha256'] != spec['runtime_members'][name] for name in RUNTIME_MEMBERS)):
            raise ValueError('ADAPT5K_HANDOFF_REQUIRES_THE_ORIGINAL_IMMUTABLE_TRAIN_AND_EVAL_WORKERS')
        base = read_pin(spec['phase_zero_config'])
        result.update(data_contract=base['data_contract'],parent_checkpoint=base['parent_checkpoint'])
        train_pin, detail, seal, gates = cold_predecessor(spec,base,previous)
        if (role == 'train' and not 25 <= seal['step'] < limit
                or role == 'eval' and (seal['step'] != limit or detail['exit_code'] != 0)):
            raise ValueError('ADAPT5K_HANDOFF_REQUIRES_FORWARD_TRAINING_OR_EXACT_FORMAL_EVAL_ENDPOINT')
        config = dict(base,job_id=spec['job_id'],deadline_epoch=spec['deadline_epoch'])
        assets,training,evidence,cpu = site_inputs(config)
        proof = assets['train_proof']
        if limit > contract.SWITCH_PHASE:
            if seal['step'] == contract.SWITCH_PHASE:
                boundary_evaluation_registered(spec,base,lineage)
            decision,training,cpu = adapt5k_curriculum.select(spec,lineage,training,detail,seal)
            selected = read_pin(decision)
            proof = selected['train_proof']
            if seal['step'] > contract.SWITCH_PHASE:
                prior_scope = read_pin(pin(previous/'RESULT.private.json'))
                if (prior_scope.get('stage2_curriculum') != decision
                        or detail['binding']['stage_bank_sha256'] != selected['stage2_bank_sha256']
                        or detail['binding']['stage_start'] != contract.SWITCH_PHASE):
                    raise ValueError('ADAPT5K_HANDOFF_STAGE2_PREDECESSOR_USED_A_DIFFERENT_CURRICULUM')
            result['stage2_curriculum'] = decision
        wall = 21600 if role == 'train' else 5400
        common,env,_ = pbs_site(config,root,wall=wall)
        paths = [str(runtime),str(runtime/'src'),*spec['dependencies']]
        env['PYTHONPATH'] = ':'.join(paths)
        common.update(parent_manifest=base['parent_manifest'],tokenizer=base['tokenizer'])
        write(root/'ACTIVATED.private.json',dict(previous=seal,training_receipt=train_pin,job=spec['job_id'],
                                                deadline_epoch=common['deadline_epoch'],role=role,limit=limit))

        def run_worker(name,module,mode,card):
            if any(p.exists() for p in stop_files) or time.time()+360 >= common['deadline_epoch']:
                raise TimeoutError('ADAPT5K_HANDOFF_STOP_BEFORE_WORKER')
            work = root/name
            work.mkdir()
            spec_path = root/(name+'.SPEC.private.json')
            write(spec_path,dict(common,registered_root=str(work),**card))
            write(root/'PROGRESS.private.json',dict(current_role=name,epoch=time.time(),previous_step=seal['step']))
            result['unresolved_role'] = name
            with (root/(name+'.log')).open('x') as log:
                proc = subprocess.run([sys.executable,'-B','-m',module,mode,str(spec_path),pin(spec_path)['sha256']],
                                      cwd=runtime,env=env,stdout=log,stderr=subprocess.STDOUT,
                                      timeout=max(1,common['deadline_epoch']-time.time()))
            outcome = pin(work/'RESULT.private.json')
            data = read_pin(outcome)
            if proc.returncode != data['exit_code']:
                raise ValueError('ADAPT5K_HANDOFF_PROCESS_AND_DURABLE_EXIT_DIFFER')
            result['completed_roles'][name] = outcome
            result['new_updates'] += data['new_updates']
            result['model_generations'] += data.get('model_generations',data.get('complete_H40_generations',0))
            result['unresolved_role'] = None
            return data

        if role == 'train':
            data = run_worker('train','scripts.sla.train_adapt5k','train',dict(
                data=training['data'],train_data_proof=proof,target_D1_sha256=base['D1_freeze_sha256'],
                parent_checkpoint=base['parent_checkpoint'],parent_training_spec=base['parent_training_spec'],
                parent_step=base['parent_step'],stage_start=training['stage_start'],stage_stop=training['stage_stop'],
                target_seed=training['target_seed'],stage_bank_sha256=cpu['vanda']['bank_digests'],
                microbatch=dict.fromkeys(contract.DOMAINS,1),phase_stop_limit=limit,
                previous_train_or_qualification=train_pin,**gates))
            result.update(status=data['status'],exit_code=data['exit_code'],optimization_visits=data['optimization_visits'])
            if data['exit_code'] in (0,75):
                _, result['sealed'] = verify_train_result(result['completed_roles']['train'],base['parent_checkpoint'])
            else:
                result['failure'] = data.get('failure')
        else:
            evaluation = dict(evidence,sealed_training_actual=train_pin,checkpoint=seal['checkpoint'])
            before = run_worker('formal','scripts.sla.evaluate_adapt5k','formal',
                                dict(evaluation,generation_cap=288+len(read_pin(evidence['panel'])['phase_train'])))
            if before['exit_code']:
                result.update(status=before['status'],failure=before.get('failure'))
                return result['exit_code']
            restored = run_worker('formal-restore','scripts.sla.evaluate_adapt5k','formal-restore',
                                  dict(evaluation,generation_cap=48,before=result['completed_roles']['formal']))
            if restored['exit_code']:
                result.update(status=restored['status'],failure=restored.get('failure'))
                return result['exit_code']
            restored_outputs = root/'OUTPUT_RESTORE_ACTUAL.private.json'
            write(restored_outputs,restore_proof(before,restored))
            result.update(status='ADAPT_FORMAL_EVALUATION_COMPLETE',output_restore_proof=pin(restored_outputs),exit_code=0)
        return result['exit_code']
    except (Exception, KeyboardInterrupt) as error:
        result.update(status='ADAPT_HANDOFF_NEEDS_RECONCILIATION',failure=f'{type(error).__name__}: {error}',exit_code=1)
        raise
    finally:
        write(root/'RESULT.private.json',result)
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec')
    parser.add_argument('sha256')
    args = parser.parse_args()
    raise SystemExit(run(read_pin(dict(path=args.spec,sha256=args.sha256))))
