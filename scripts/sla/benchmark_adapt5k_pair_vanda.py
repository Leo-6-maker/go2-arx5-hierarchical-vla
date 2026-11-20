"""Qualify the registered two-A40 ADAPT comparison inside its own PBS job."""
import argparse
from contextlib import closing
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from go2arx5_vla.sla_adapter import adapt5k_pair_site
from go2arx5_vla.sla_adapter.adapt5k_contract import PLAN
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from scripts.pi0292.durable import OwnerLock, write
from scripts.sla.adapt5k_vanda import pbs_history
from scripts.sla.benchmark_adapt5k import (
    BATCH_SCOPE, TWO_GPU_VARIANT, exposure_fingerprint, paired_reference, require_batched_output_panel,
)
from scripts.sla.benchmark_adapt5k_vanda import batch_output_roles
from scripts.sla.evaluate_adapt5k_performance import compare_outputs
from scripts.sla.scale20k_vanda_offline import read_pbs_with_auth_retry
from scripts.wm02.multisource_d4_vanda import allocation_window, pin


WALL = 10800
SITE_SCHEMA = 'SLA_ADAPT_TWO_LOCAL_A40_PBS_V1'


def site_inputs(spec, config, reference, root, prior):
    job_id = spec['job_id']
    job = read_pbs_with_auth_retry(pbs_history, job_id)
    window = allocation_window(job, campaign_deadline=spec['deadline_epoch'])
    deadline = window['deadline_epoch']
    host = os.uname().nodename
    if (job_id != os.environ.get('PBS_JOBID') or job['job_state'] != 'R'
            or job['project'] != config['project'] or job['Job_Owner'].split('@')[0] != config['user']
            or config['uid'] != os.getuid() or host != reference['device']['host']
            or int(job['Resource_List']['ngpus']) != 2 or window['PBS_wall_s'] > WALL
            or deadline - time.time() <= 1800):
        raise PermissionError('ADAPT_PAIR_OWN_TWO_A40_PBS_ALLOCATION_AND_REFERENCE_HOST_REQUIRED')
    visible = adapt5k_pair_site.visible_cuda_uuids()
    primary = reference['device']['GPU_uuid']
    if primary not in visible or len(set(visible)) != 2:
        raise PermissionError('ADAPT_PAIR_ACTUAL_PBS_MASK_MISSING_REFERENCE_CARD')
    ordered = [primary, next(uuid for uuid in visible if uuid != primary)]
    compilation = Path(os.environ['TMPDIR']).resolve(strict=True)/'sla-adapt5k-pair-compile'
    compilation.mkdir()
    binding = root/'PAIR_SITE.private.json'
    write(binding, dict(schema=SITE_SCHEMA, job_id=job_id, host=host, uid=os.getuid(),
        project=config['project'], user=config['user'], deadline_epoch=deadline,
        GPU_uuids=ordered, reference_GPU_uuid=primary, parent_checkpoint_sha256=prior['checkpoint']['sha256']))
    pair = dict(project=config['project'], deadline_epoch=deadline, compilation_cache_dir=str(compilation),
                expected_pair_parent_sha256=prior['checkpoint']['sha256'], pair_site_binding=pin(binding))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(ordered), JAX_PLATFORMS='cuda',
        XLA_FLAGS=config['XLA_FLAGS'], XLA_PYTHON_CLIENT_MEM_FRACTION='0.8',
        XLA_PYTHON_CLIENT_PREALLOCATE='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    # CUDA enumeration in this process is already fixed by the PBS mask.
    # Validate the reordered subset in a fresh child before any JAX work.
    validate = ('import json,sys; from go2arx5_vla.sla_adapter.adapt5k_pair_site import validate; '
                'print(json.dumps(validate(json.load(sys.stdin))))')
    measured = json.loads(subprocess.check_output([sys.executable,'-B','-c',validate],
        input=json.dumps(pair),text=True,env=env,timeout=50))
    observed = root/'PBS_ALLOCATED.private.json'
    write(observed, dict(window, state='R', job_id=job_id, uid=os.getuid(), host=host, PID=os.getpid(),
          project=config['project'], allocated_GPU_uuids=ordered, topology=measured['topology'],
          devices=measured['devices'], observed_epoch=time.time()))
    paths = {key:str(Path(config[key]).resolve(strict=True)) for key in ('parent_params','dataset')}
    paths['compilation_cache_dir'] = str(compilation)
    one_gpu = root/'PRIMARY_EVAL_SITE.private.json'
    write(one_gpu, dict(schema='SLA_SINGLE_GPU_EXISTING_SITE_V2', host=host, uid=os.getuid(),
        GPU_uuid=primary, GPU_name=measured['GPU_name'], allocation_deadline_epoch=window['allocation_deadline_epoch'],
        allocator_limit_bytes=math.ceil(.8*measured['devices'][0]['GPU_bytes']), GPU_headroom_bytes=2*adapt5k_pair_site.GIB,
        host_headroom_bytes=64*adapt5k_pair_site.GIB, paths=paths, allocation=dict(kind='PBS', job_id=job_id,
             project=config['project'], actual_allocation=pin(observed))))
    common = dict(plan=PLAN, source_sha=spec['source_sha'], project=config['project'], **paths,
        deadline_epoch=deadline, GPU_uuid=primary, site_binding=pin(one_gpu), JAX_memory_fraction=.8,
        max_walltime_s=WALL, max_gpu_hours=2*WALL/3600, save_and_exit_on_limit=True,
        model_service_timeout_s=1800, RPC_timeout_s=1800, STOP_path=str(root/'STOP.json'),
        neural_owner_lock=str(root/'DEVICE_WORKER.lock'), hardware_allowed=False, physics=False)
    return common, env, pair, pin(observed)


def run(spec):
    root = Path(spec['root']).resolve(strict=True)
    source = Path(__file__).resolve().parents[2]
    plan, prior, config, previous = map(read_pin, (spec['engineering_contract'], spec['sealed_predecessor'],
                                                    spec['phase_zero_config'], spec['cold_PBS_scope']))
    if (root.stat().st_uid != os.getuid() or spec['plan'] != PLAN or spec['scope'] != BATCH_SCOPE
            or spec['job_id'] != os.environ.get('PBS_JOBID') or spec['physics'] is not False
            or spec['hardware_allowed'] is not False or spec['source_sha'] != (source/'SOURCE_SHA.private').read_text().strip()
            or plan['update_limit'] != {TWO_GPU_VARIANT:16} or plan['scope'] != BATCH_SCOPE
            or plan['parent_checkpoint_sha256'] != prior['checkpoint']['sha256']
            or previous['role'] != 'train' or previous['exit_code'] or previous['completed_roles']['train'] != spec['sealed_predecessor']
            or previous['data_contract'] != config['data_contract'] or previous['parent_checkpoint'] != config['parent_checkpoint']
            or (root/'RESULT.private.json').exists() or (root/'ACTIVATED.private.json').exists()):
        raise PermissionError('ADAPT_PAIR_FRESH_ENGINEERING_SCOPE_WITH_EXACT_ANCHOR_REQUIRED')
    ready = root/'ACTIVATED.private.json'
    stop = root/'STOP.json'
    actual = dict(scope=BATCH_SCOPE, role=TWO_GPU_VARIANT, status='ADAPT_PAIR_REQUIRES_RECONCILIATION',
        job_id=spec['job_id'], source_sha=spec['source_sha'], engineering_only=True, eligible_as_formal_parent=False,
        optimizer_updates=0, scientific_updates=0, native_requests=0, native_ticks=0,
        offline_roles={}, offline_generated=0, offline_attempted=0, exit_code=1)

    def stopping(*_):
        if not stop.exists():
            write(stop, dict(reason='OPERATOR_PBS_STOP'))

    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig,stopping)

    start = time.monotonic()
    with closing(OwnerLock(root/'OWNER.lock')):
        try:
            if stop.exists():
                raise PermissionError('ADAPT_PAIR_STOPPED_BEFORE_PREFLIGHT')
            parent_job = read_pbs_with_auth_retry(pbs_history, previous['PBS_JOBID'])
            if (parent_job['job_state'] != 'F' or parent_job['Exit_status'] != 0
                    or parent_job['project'] != config['project'] or parent_job['Job_Owner'].split('@')[0] != config['user']):
                raise PermissionError('ADAPT_PAIR_ORIGINAL_PHASE_1000_OWNER_NOT_COLD')
            panel = require_batched_output_panel(plan)
            reference_pin,reference = paired_reference(dict(spec,project=config['project']),prior,panel)
            cold = read_pin(spec['paired_single_scope'])
            previous_roles = cold['offline_roles']['compiled_mb2']
            original = read_pin(previous_roles['ordinary'])
            restored = read_pin(previous_roles['restore'])
            baseline = read_pin(cold['offline_roles']['compiled_mb1']['ordinary'])
            if (original['engineering_checkpoint'] != reference['engineering_checkpoint'] or original['panel'] != panel
                    or not compare_outputs(original,original,restored)['physical_target_compatible']):
                raise ValueError('ADAPT_PAIR_NEEDS_SAME_ROUND_SINGLE_A40_48_AND_EXACT_RESTORE')
            common,env,pair, observed = site_inputs(spec,config,reference,root,prior)
            write(ready,dict(engineering_only=True, allocation=observed, reference=reference_pin, checkpoint=prior['checkpoint']))
            actual.update(allocation=observed, reference=reference_pin, unresolved_role=TWO_GPU_VARIANT)
            work = root/TWO_GPU_VARIANT
            work.mkdir()
            card = root/'PAIR_BENCHMARK.private.json'
            write(card,dict(common | pair, scope=BATCH_SCOPE, variant=TWO_GPU_VARIANT,
                registered_root=str(work), paired_single_scope=spec['paired_single_scope'],
                engineering_contract=spec['engineering_contract'],training_card=spec['training_card'],
                sealed_predecessor=spec['sealed_predecessor']))
            terminal = work/'RESULT.private.json'
            try:
                with (root/'training.log').open('x') as log:
                    outcome = subprocess.run([sys.executable,'-B','-m','scripts.sla.benchmark_adapt5k',
                        str(card),pin(card)['sha256']], timeout=common['deadline_epoch']-time.time(),
                        env=env, cwd=source, stdout=log, stderr=subprocess.STDOUT)
            finally:
                if terminal.is_file():
                    actual['training'] = pin(terminal)
                    actual['optimizer_updates'] = read_pin(actual['training'])['actual_optimizer_updates']
                elif (ledger:=work/'UPDATES.private.jsonl').is_file():
                    rows=[json.loads(row) for row in ledger.read_bytes().splitlines(keepends=True) if row.endswith(b'\n')]
                    actual['optimizer_updates'] = len(rows)
                    actual['unsealed_ledger_prefix'] = dict(ledger=pin(ledger),updates=len(rows),completed_H40=4*len(rows),
                                                          exposure_sha256=exposure_fingerprint(rows,fm_microbatch=2))
            if not terminal.is_file():
                raise RuntimeError(f'ADAPT_PAIR_WORKER_EXIT_{outcome.returncode}_WITHOUT_TERMINAL')
            effect = read_pin(actual['training'])
            if outcome.returncode != effect['exit_code'] or effect['exit_code'] or actual['optimizer_updates'] != 16:
                raise RuntimeError(f'ADAPT_PAIR_TRAINING_EFFECT_FAILED:{effect.get("failure")}')
            one_gpu = dict(env,CUDA_VISIBLE_DEVICES=common['GPU_uuid'])
            roles = batch_output_roles(root,source,common,one_gpu,spec,panel,actual,TWO_GPU_VARIANT,actual['training'])
            ordinary, recovered = map(read_pin,(roles['ordinary'],roles['restore']))
            outputs = compare_outputs(baseline,ordinary,recovered)
            matched = compare_outputs(original,ordinary,recovered)
            scaling = dict(effect['strong_scaling'],output_qualification_pending=False)
            adopted = scaling['timed_thresholds_met'] and outputs['physical_target_compatible']
            actual.update(status='ADAPT_PAIR_ENGINEERING_COMPLETE',exit_code=0,unresolved_role=None,
                strong_scaling=scaling,outputs=outputs,outputs_vs_single_mb2=matched,compatible_and_faster=adopted)
        except (Exception,KeyboardInterrupt) as error:
            actual['first_failure'] = f'{type(error).__name__}: {error}'
        finally:
            write(root/'RESULT.private.json',dict(actual,wall_s=time.monotonic()-start))
    return actual['exit_code']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec')
    parser.add_argument('sha256')
    args = parser.parse_args()
    raise SystemExit(run(read_pin(dict(path=args.spec,sha256=args.sha256))))
