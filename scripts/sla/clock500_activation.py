"""Activate the preregistered CLOCK PBS graph only from exact positive evidence."""
import argparse
from contextlib import closing
import os
from pathlib import Path
import time

from go2arx5_vla.sla_adapter.adapt5k_contract import PLAN as ADAPT_PLAN
from go2arx5_vla.sla_adapter.clock500 import ARMS, PLAN
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from scripts.pi0292.durable import OwnerLock, write
from scripts.sla.adapt5k_curriculum import DECISION
from scripts.sla.clock500_trigger import SCOPE as TRIGGER_SCOPE, owned_job, pbs
from scripts.sla.freeze_clock500_native_population import validate as validate_population
from scripts.wm02.multisource_d4_vanda import pin


SCOPE = 'SLA_CLOCK500_CONDITIONAL_PBS_ACTIVATION_V1'
ACTIVE = 'CLOCK500_MATCHED_TRAIN_AND_EVAL_ACTIVATED'


def trigger_status(trigger, registered):
    if (trigger['scope']!=TRIGGER_SCOPE or trigger['job_id']!=registered['job_id']
            or trigger['source_sha']!=registered['source_sha']
            or any(trigger[k] for k in ('optimizer_updates','model_generations','native_requests','native_ticks'))):
        raise ValueError('CLOCK500_ACTIVATION_TRIGGER_IDENTITY_CHANGED')
    if trigger['scientific_trigger'] is not True:
        return trigger['status']
    if (trigger['joint_numerator']!=0 or trigger['eligible_windows']!=24
            or trigger['semantic_audit']['labels_verified']!=88
            or trigger['semantic_audit']['due_experts_verified']!=24
            or trigger['semantic_audit']['original_H40_and_seam_verified']!=88):
        raise ValueError('CLOCK500_ACTIVATION_REQUIRES_ZERO_OF_24_AND_THE_ORIGINAL_LABEL_AUDIT')
    if trigger['status']=='TRIGGERED_AWAITING_EXECUTION_READINESS':
        return ACTIVE
    if trigger['status']=='BLOCKED_ARTIFACT' and trigger.get('first_failure')=='MISSING_COLD_D2_CURRICULUM_DECISION':
        return ACTIVE
    return trigger['status']


def activate(spec, effect):
    registered = read_pin(spec['trigger_spec'])
    job = pbs(registered['job_id'])
    if job.get('Exit_status')!=0 or not owned_job(job,spec,'F'):
        raise ValueError('CLOCK500_ACTIVATION_NEEDS_COLD_CPU_TRIGGER')
    input_path = Path(registered['root'])/'RESULT.private.json'
    trigger_pin = pin(input_path)
    trigger = read_pin(trigger_pin)
    status = trigger_status(trigger,registered)
    effect.update(trigger=trigger_pin,scientific_trigger=trigger['scientific_trigger'])
    if status!=ACTIVE:
        effect['status'] = status
        return 3

    root = Path(spec['root'])
    stop = (root/'STOP.json',Path(spec['campaign_stop']))
    wait_until = min(time.time()+1200,spec['deadline_epoch']-3600)
    decision_path = Path(registered['curriculum_decision'])
    while not decision_path.is_file():
        if any(x.exists() for x in stop) or time.time()>=wait_until:
            effect['status'] = 'CLOCK500_AWAITS_COLD_D2_DECISION'
            return 3
        owner = pbs(spec['D2_selector_job'])
        if (owner['Job_Owner'].split('@')[0]!=spec['user'] or owner['project']!=spec['project']
                or owner['job_state'] not in ('H','Q','R','E','W')):
            # Recheck after observing the selector terminal to close the race.
            if decision_path.is_file():
                break
            effect['status'] = 'CLOCK500_D2_SELECTOR_CLOSED_WITHOUT_DECISION'
            return 3
        time.sleep(15)
    selection = pin(decision_path)
    chosen = read_pin(selection)
    if (chosen['schema']!=DECISION or chosen['plan']!=ADAPT_PLAN or chosen['mode']!='D2'
            or chosen['phase2000_checkpoint']!=trigger['checkpoint'] or chosen['ready']!=registered['D2_ready']
            or chosen['used_eval_score'] or chosen['stage2_bank_sha256']!=spec['expected_bank_sha256']
            or ('curriculum' in trigger and trigger['curriculum']!=selection)):
        effect['status'] = 'CLOCK500_COLD_D2_DECISION_INELIGIBLE'
        return 3
    if any(x.exists() for x in stop) or time.time()+3600>=spec['deadline_epoch']:
        effect['status'] = 'CLOCK500_ACTIVATION_EXPIRED_OR_STOPPED'
        return 3
    frozen = read_pin(spec['native_population'])
    validate_population(frozen)
    origin = read_pin(spec['population_qualification'])
    if (origin['status']!='CLOCK500_COLD_PAIRED_POPULATION_FROZEN' or origin['future_native_cap']!=72
            or origin['population']['sha256']!=spec['native_population']['sha256']
            or origin['compiled']['protection_fingerprints']!=[spec['original_robot_fingerprint']]):
        raise ValueError('CLOCK500_ACTIVATION_NATIVE_POPULATION_BINDING_CHANGED')
    keys = ('plan','source_sha','branch_roots','phase_zero_config','native_population','population_qualification',
            'original_robot_fingerprint','runtime','project','user','deadline_epoch','campaign_stop')
    file = root/'COMPARISON.private.json'
    write(file,dict({k:spec[k] for k in keys},arms=list(ARMS),updates_per_arm=500,
                    trigger=trigger_pin,trigger_spec=spec['trigger_spec'],D2_decision=selection))
    effect.update(status=ACTIVE,comparison=pin(file))
    return 0


def run(spec, spec_pin):
    root = Path(spec['root']).resolve(strict=True)
    source = Path(__file__).resolve().parents[2]
    job = os.environ.get('PBS_JOBID')
    if (spec['scope']!=SCOPE or spec['plan']!=PLAN or spec['job_id']!=job or root.stat().st_uid!=os.getuid()
            or spec['source_sha']!=(source/'SOURCE_SHA.private').read_text().strip()):
        raise PermissionError('CLOCK500_ACTIVATION_REQUIRES_ITS_ACTUAL_CPU_ALLOCATION')
    with closing(OwnerLock(root/'OWNER.lock')):
        if (root/'RESULT.private.json').exists() or (root/'COMPARISON.private.json').exists():
            raise FileExistsError('CLOCK500_ACTIVATION_ALREADY_HAS_A_DURABLE_EFFECT')
        effect = dict(scope=SCOPE,job_id=job,spec=spec_pin,source_sha=spec['source_sha'],
                      scientific_trigger=None,optimizer_updates=0,model_generations=0,native_requests=0,native_ticks=0)
        try:
            allocation = pbs(job)
            if not owned_job(allocation,spec,'R') or int(allocation['Resource_List'].get('ngpus',0)):
                raise PermissionError('CLOCK500_ACTIVATION_REQUIRES_ITS_ACTUAL_CPU_ALLOCATION')
            if time.time()+3600>=spec['deadline_epoch'] or (root/'STOP.json').exists() or Path(spec['campaign_stop']).exists():
                effect['status'] = 'CLOCK500_ACTIVATION_EXPIRED_OR_STOPPED'
                code = 3
            else:
                code = activate(spec,effect)
        except (Exception,KeyboardInterrupt) as error:
            effect.update(status='CLOCK500_ACTIVATION_REQUIRES_RECONCILIATION',first_failure=f'{type(error).__name__}: {error}')
            code = 1
        write(root/'RESULT.private.json',dict(effect,exit_code=code))
    return code


def activated_comparison(spec):
    """Resolve the previously registered CPU owner and its one immutable effect."""
    registered = spec['activation']
    job = pbs(registered['job_id'])
    root = Path(registered['root']).resolve(strict=True)
    report = read_pin(pin(root/'RESULT.private.json'))
    gate = read_pin(report['spec'])
    if (not owned_job(job,spec,'F') or job.get('Exit_status')!=0
            or report['scope']!=SCOPE or report['status']!=ACTIVE or report['exit_code']!=0
            or report['job_id']!=registered['job_id'] or report['source_sha']!=spec['source_sha']
            or Path(report['spec']['path'])!=root/'EXECUTION.private.json' or gate['root']!=str(root)
            or gate['scope']!=SCOPE or gate['plan']!=PLAN or gate['source_sha']!=spec['source_sha']
            or gate['job_id']!=registered['job_id'] or gate['branch_roots'][spec['arm']]!=spec['lineage']
            or Path(report['comparison']['path'])!=root/'COMPARISON.private.json'):
        raise PermissionError('CLOCK500_GPU_SCOPE_HAS_NO_MATCHING_COLD_ACTIVATION')
    return report['comparison']


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec')
    parser.add_argument('sha256')
    args = parser.parse_args()
    registered = dict(path=args.spec,sha256=args.sha256)
    raise SystemExit(run(read_pin(registered),registered))
