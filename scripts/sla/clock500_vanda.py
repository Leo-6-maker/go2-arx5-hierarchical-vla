"""Own one PBS-GPU CLOCK training segment or an independent endpoint Eval."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from go2arx5_vla.sla_adapter.clock500 import ARMS, ENDPOINT, PARENT, PLAN
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from scripts.pi0292.durable import OwnerLock, write
from scripts.sla.adapt5k_vanda import pbs_site
from scripts.sla.clock500_activation import activated_comparison
from scripts.sla.clock500_trigger import owned_job, pbs
from scripts.sla.evaluate_adapt5k import restore_proof
from scripts.sla.train_clock500 import SCOPE, comparison_inputs
from scripts.wm02.multisource_d4_vanda import pin


WRAPPER = 'SLA_CLOCK500_OWNED_PBS_BRANCH_V1'


def previous_effect(spec):
    if spec['previous'] is None:
        if spec['role']!='train':
            raise ValueError('CLOCK500_EVAL_REQUIRES_THE_REAL_COLD_500_UPDATE_ENDPOINT')
        return None,None
    registered = read_pin(spec['previous'])
    cold = pbs(registered['job_id'])
    previous = Path(registered['root']).resolve(strict=True)
    if previous.parent!=Path(spec['lineage']).resolve(strict=True):
        raise ValueError('CLOCK500_COLD_PREDECESSOR_OUTSIDE_MATCHED_BRANCH_LINEAGE')
    outer = read_pin(pin(previous/'RESULT.private.json'))
    effect_pin = outer['effects']['train']
    effect = read_pin(effect_pin)
    if (not owned_job(cold,spec,'F') or outer['exit_code'] not in (0,75)
            or cold.get('Exit_status')!=outer['exit_code'] or outer['exit_code']!=effect['exit_code']
            or registered['role']!='train' or registered['source_sha']!=spec['source_sha']
            or registered['job_id']!=outer['job_id'] or effect['job_id']!=outer['job_id']
            or outer['scope']!=WRAPPER or outer['comparison']!=spec['comparison']
            or effect['scope']!=SCOPE or outer['arm']!=spec['arm'] or effect['arm']!=spec['arm']
            or outer['native_requests'] or effect['native_requests'] or effect['native_ticks']
            or Path(effect_pin['path']).parent!=previous/'train'
            or not effect['checkpoint']):
        raise ValueError('CLOCK500_COLD_BRANCH_PBS_HANDOFF_NOT_QUALIFIED')
    if spec['role']=='eval' and (outer['exit_code'] or effect['end_phase']!=ENDPOINT):
        raise ValueError('CLOCK500_INDEPENDENT_EVAL_NEEDS_EXACT_500_UPDATE_ENDPOINT')
    if spec['role']=='train' and not effect['end_phase']<spec['stop_phase']:
        raise ValueError('CLOCK500_TRAIN_SUCCESSOR_MUST_ADVANCE_BEYOND_COLD_PREFIX')
    return effect_pin,effect


def run(spec):
    root = Path(spec['root']).resolve(strict=True)
    lineage = Path(spec['lineage']).resolve(strict=True)
    source = Path(__file__).resolve().parents[2]
    job = os.environ.get('PBS_JOBID')
    if (spec['plan']!=PLAN or spec['job_id']!=job or not job or root.stat().st_uid!=os.getuid()
            or lineage.stat().st_uid!=os.getuid() or root.parent!=lineage
            or spec['source_sha']!=(source/'SOURCE_SHA.private').read_text().strip()
            or (root/'RESULT.private.json').exists() or (root/'ACTIVATED.private.json').exists()
            or spec['arm'] not in ARMS or spec['role'] not in ('train','eval')
            or spec['physics'] is not False or spec['hardware_allowed'] is not False
            or not PARENT<spec['stop_phase']<=ENDPOINT):
        raise PermissionError('CLOCK500_NEW_OWNED_PBS_SCOPE_REQUIRED')
    role = spec['role']
    stop = root/'STOP.json'
    result = dict(scope=WRAPPER,role=role,arm=spec['arm'],job_id=job,source_sha=spec['source_sha'],
                  comparison=spec.get('comparison'),effects={},optimizer_updates=0,model_generations=0,
                  native_requests=0,native_ticks=0,exit_code=1)

    def interrupted(*_):
        if not stop.exists():
            write(stop,dict(reason='OPERATOR_PBS_STOP'))

    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig,interrupted)
    lock = OwnerLock(lineage/('LINEAGE_OWNER.lock' if role=='train' else 'EVAL_PHASE02500.lock'))
    try:
        if stop.exists() or Path(spec['campaign_stop']).exists():
            raise PermissionError('CLOCK500_SITE_SCOPE_STOPPED')
        if 'activation' in spec:
            spec = dict(spec,comparison=activated_comparison(spec))
            result['comparison'] = spec['comparison']
        pair,data,anchor = comparison_inputs(spec)
        if (pair['branch_roots'][spec['arm']]!=str(lineage) or spec['deadline_epoch']>pair['deadline_epoch']
                or pair['campaign_stop']!=spec['campaign_stop']):
            raise ValueError('CLOCK500_MUST_USE_ITS_OWN_REGISTERED_ARM_LINEAGE')
        before,effect = previous_effect(spec)
        config = read_pin(pair['phase_zero_config'])
        common,env,_ = pbs_site(dict(config,job_id=job,deadline_epoch=spec['deadline_epoch']),root,
                                wall=21600 if role=='train' else 5400)
        if common['GPU_uuid']!=pair['runtime']['GPU_uuid'] or spec['project']!=pair['project']:
            raise PermissionError('CLOCK500_PAIRED_RUNTIME_SITE_CHANGED')
        phase = effect['end_phase'] if effect else PARENT
        claim = lineage/(f'APPLY_AFTER{phase:05d}.private.json' if role=='train' else 'EVAL_PHASE02500.private.json')
        if claim.exists():
            raise PermissionError('CLOCK500_FIXED_BRANCH_INTERVAL_ALREADY_CLAIMED')
        write(claim,dict(job=job,source_sha=spec['source_sha'],root=str(root),comparison=spec['comparison'],
                         previous=before,phase=phase))
        common.update(source_sha=spec['source_sha'],arm=spec['arm'],comparison=spec['comparison'],
                      plan=PLAN,job_id=job,project=pair['project'],user=pair['user'],campaign_stop=spec['campaign_stop'])
        write(root/'ACTIVATED.private.json',dict(parent=anchor,job=job,source_sha=spec['source_sha'],
                                                arm=spec['arm'],deadline_epoch=common['deadline_epoch']))
        dependency = os.pathsep.join([str(source),str(source/'src'),*spec['dependencies']])
        env['PYTHONPATH'] = dependency

        def execute(name, module, arguments, parameters):
            left = common['deadline_epoch']-time.time()
            if left<=360 or stop.exists() or Path(spec['campaign_stop']).exists():
                raise TimeoutError('CLOCK500_STOP_OR_DEADLINE_BEFORE_NEXT_ROLE')
            location = root/name
            location.mkdir()
            card = root/(name+'.SPEC.private.json')
            write(card,dict(common,registered_root=str(location),**parameters))
            result['unresolved_role'] = name
            with (root/(name+'.log')).open('x') as log:
                proc = subprocess.run([sys.executable,'-B','-m',module,*arguments,str(card),pin(card)['sha256']],
                    timeout=left,env=env,cwd=source,stdout=log,stderr=subprocess.STDOUT)
            descriptor = pin(location/'RESULT.private.json')
            response = read_pin(descriptor)
            if proc.returncode!=response['exit_code']:
                raise ValueError('CLOCK500_PROCESS_EXIT_DIFFERS_FROM_DURABLE_EFFECT')
            result['effects'][name] = descriptor
            result['unresolved_role'] = None
            result['optimizer_updates'] += response.get('optimizer_updates',0)
            result['model_generations'] += response.get('model_generations',response.get('complete_H40',0))
            return response

        if role=='train':
            done = execute('train','scripts.sla.train_clock500',[],dict(previous=before,stop_phase=spec['stop_phase']))
            result.update(status=done['status'],exit_code=done['exit_code'],checkpoint=done['checkpoint'])
        else:
            original = read_pin(read_pin(config['data_contract'])['eval_input'])
            eval_card = dict(original,tokenizer=data['tokenizer'],parent_manifest=data['parent_manifest'],
                             checkpoint=effect['checkpoint'],sealed_training_actual=before)
            first = execute('ordinary','scripts.sla.evaluate_adapt5k',['formal','--clock-branch'],
                            dict(eval_card,generation_cap=376))
            if first['exit_code']:
                result.update(status=first['status'],exit_code=first['exit_code'])
                return result['exit_code']
            restored = execute('independent-restore','scripts.sla.evaluate_adapt5k',['formal-restore','--clock-branch'],
                               dict(eval_card,generation_cap=48,before=result['effects']['ordinary']))
            if restored['exit_code']:
                result.update(status=restored['status'],exit_code=restored['exit_code'])
                return result['exit_code']
            proof = restore_proof(first,restored)
            if first['model_generations']!=376 or proof['phase_train_predictions']!=88 or result['model_generations']!=424:
                raise ValueError('CLOCK500_EXPECTS_FIXED_288_PLUS_88_PLUS_48_GENERATIONS_PER_ENDPOINT')
            path = root/'OUTPUT_RESTORE_ACTUAL.private.json'
            write(path,dict(proof,schema='SLA_CLOCK500_BRANCH_OUTPUT_RESTORE_PROOF_V1',arm=spec['arm'],
                            comparison=spec['comparison']))
            result.update(status='CLOCK500_FIXED_OFFLINE_AND_INDEPENDENT_RESTORE_COMPLETE',exit_code=0,
                          output_restore=pin(path))
    except (Exception,KeyboardInterrupt) as error:
        result.update(status='CLOCK500_PBS_TECHNICAL_TERMINAL_RECONCILE',first_failure=f'{type(error).__name__}: {error}')
    finally:
        try:
            write(root/'RESULT.private.json',result)
        finally:
            lock.close()
    return result['exit_code']


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec')
    parser.add_argument('sha256')
    args = parser.parse_args()
    raise SystemExit(run(read_pin(dict(path=args.spec,sha256=args.sha256))))
