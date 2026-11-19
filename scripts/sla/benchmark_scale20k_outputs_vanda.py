"""Use one actual PBS device for two fresh, independent replay processes."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from go2arx5_vla.sla_adapter.adapt5k_contract import PLAN
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from scripts.pi0292.durable import OwnerLock, write
from scripts.sla.adapt5k_vanda import pbs_history, pbs_site
from scripts.sla.benchmark_scale20k_outputs import PANEL, VARIANTS
from scripts.sla.scale20k_vanda_offline import read_pbs_with_auth_retry
from scripts.wm02.multisource_d4_vanda import pin


def run(config):
    root = Path(config['root']).resolve(strict=True)
    source = Path(__file__).resolve().parents[2]
    stop = root/'STOP.json'
    if (config['plan'] != PLAN or root.stat().st_uid != os.getuid() or stop.exists()
            or (root/'RESULT.private.json').exists() or (source/'SOURCE_SHA.private').read_text().strip() != config['source_sha']):
        raise PermissionError('FRESH_IMMUTABLE_BOUNDED_ENGINEERING_REPLAY_REQUIRED')
    outputs = {}
    receipt = dict(scope=PANEL,PBS_JOBID=os.environ.get('PBS_JOBID'),source_sha=config['source_sha'],
        model_generations=0,new_optimizer_updates=0,native_requests=0,role_results=outputs,exit_code=1)
    lock = OwnerLock(root/'PBS_OWNER.lock')
    try:
        old = read_pin(config['benchmark_pbs_result'])
        authority = read_pbs_with_auth_retry(pbs_history,old['PBS_JOBID'])
        archived = read_pin(old['candidate'])
        if (old['exit_code'] or authority['job_state'] != 'F' or authority['Exit_status']
                or authority['project'] != config['project'] or authority['Job_Owner'].split('@')[0] != config['user']
                or archived['variant'] != config['variant'] or config['variant'] not in VARIANTS
                or old['candidate'] != config['benchmark_terminal']):
            raise ValueError('EXACT_PBS_COLD_ENGINEERING_PARENT_REQUIRED')
        common,env,job = pbs_site(config,root)
        spec = dict(common,**{k:config[k] for k in ('original_panel','site_assets','selection','benchmark_terminal',
                                                  'parent_training_spec')},model_generation_cap=48)
        for role in ('ordinary','restore'):
            if stop.exists() or common['deadline_epoch']-time.time() < 180:
                raise TimeoutError('A40_ENGINEERING_REPLAY_FINITE_SITE_WINDOW')
            child = root/role
            child.mkdir()
            card = root/(role+'.SPEC.private.json')
            if role == 'restore':
                spec['ordinary_result'] = outputs['ordinary']
            write(card,dict(spec,registered_root=str(child)))
            write(root/'PROGRESS.private.json',dict(role=role,epoch=time.time(),job=job))
            with (root/(role+'.log')).open('x') as fd:
                p = subprocess.run([sys.executable,'-B','-m','scripts.sla.benchmark_scale20k_outputs',
                    role,str(card),pin(card)['sha256']],cwd=source,env=env,stdout=fd,stderr=subprocess.STDOUT,
                    timeout=max(1,common['deadline_epoch']-time.time()))
            tail = child/'RESULT.private.json'
            if not tail.is_file():
                raise RuntimeError('ENGINEERING_OUTPUT_EFFECT_NEEDS_RECONCILIATION:'+role)
            outputs[role]=pin(tail)
            actual = read_pin(outputs[role])
            receipt['model_generations'] += actual['model_generations']
            if p.returncode or actual['exit_code']:
                receipt.update(status=actual['status'],failure=actual.get('failure'))
                return 1
        receipt.update(status='ENGINEERING_96_OUTPUTS_AND_INDEPENDENT_RESTORE_COMPLETE',exit_code=0)
        return 0
    except Exception as error:
        receipt.update(status='ENGINEERING_REPLAY_TECHNICAL_TERMINAL',failure=f'{type(error).__name__}: {error}')
        raise
    finally:
        lock.close()
        write(root/'RESULT.private.json',receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('sha256')
    args = parser.parse_args()
    raise SystemExit(run(read_pin(dict(path=args.config,sha256=args.sha256))))
