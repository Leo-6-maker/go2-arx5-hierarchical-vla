"""Evaluate a backed-up H200 endpoint on the original three-domain panel, in two cold GPU processes."""
import argparse
from collections import Counter
from contextlib import closing
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from scripts.pi0292.durable import OwnerLock, read
from scripts.sla import hopper_checkpoint as checkpoint
from scripts.sla.hopper_inputs import inputs, PLAN, PARENT
from scripts.sla.hopper_panels import FixedPanels, FORMAL_PANEL
from scripts.sla.hopper_parent_custody import digest, write_once
from scripts.sla.hopper_site import DEADLINE, pinned, site_epoch, wall_seconds

COUNTS = {'formal': dict(target=96, robodojo=96, libero=96, target_phase=88),
          'formal-restore': dict(target=16, robodojo=16, libero=16)}


def load_endpoint(config, receipt):
    import jax
    from go2arx5_vla.sla_adapter import scale20k_model
    from go2arx5_vla.sla_adapter.adapt5k_training import OPTIMIZER
    from go2arx5_vla.sla_adapter.scale20k_contract import canonical_hash, make_adamw
    from go2arx5_vla.sla_adapter.multitask_data_v6 import C20_STATISTICS

    root = Path(receipt['custody'])
    original = pinned(config['original_parent'])
    original_binding = original['binding']
    header = pinned(dict(path=str(root/'COMPLETE.json'), sha256=receipt['checkpoint']))
    bound = header['binding']
    if (config['original_parent']['sha256'] != PARENT or receipt['plan'] != PLAN or receipt['parent'] != PARENT
            or bound['source'] != config['training_source'] or bound['role'] != receipt['role']
            or bound['plan'] != PLAN or bound['parent'] != PARENT or bound['initial_banks'] != checkpoint.BANKS
            or header['progress'] != checkpoint.progress(bound, receipt['phase'])
            or receipt['arrays'] != len(header['arrays']) or len(header['arrays']) != 140):
        raise ValueError('Endpoint and complete archived training state disagree')
    spec = inputs(Path(config['train_assets']))
    if (spec['parent_manifest']['sha256'] != original_binding['parent_manifest_sha256']
            or digest(Path(spec['tokenizer']['path'])) != original_binding['tokenizer_sha256']
            or spec['data']['target']['statistics']['sha256'] != original_binding['target_state_sha256']
            or pinned(spec['data']['target']['statistics'])['frozen_action_statistics_sha256'] != C20_STATISTICS
            or canonical_hash(OPTIMIZER) != original_binding['optimizer_sha256']):
        raise ValueError('Frozen parent, tokenizer, statistics or optimizer identity differs')
    graphs, params, frozen, model = scale20k_model.load(spec['parent_manifest'], spec['parent_params'])
    if (canonical_hash(model['inventory']) != original_binding['partition_sha256']
            or model['frozen_fingerprint'] != bound['frozen_model']):
        raise ValueError('Frozen model or parameter partition differs')
    init, _ = make_adamw(**OPTIMIZER)
    template = dict(parameters=jax.eval_shape(lambda x:x,params), optimizer=jax.eval_shape(init,params),
                    rng=jax.ShapeDtypeStruct((2,),np.uint32))
    del params
    state, _ = checkpoint.resume(dict(path=str(root),sha256=receipt['checkpoint']), template,
                                 phase=receipt['phase'], bound=bound, predecessor=header['predecessor'])
    return graphs, state, frozen


def device(run, config):
    import jax
    import json
    job = read(run/'JOB.json')['job']
    if job != os.environ.get('PBS_JOBID') or config['plan'] != PLAN:
        raise PermissionError('Missing the scheduled formal-evaluation allocation')
    q = json.loads(subprocess.check_output(['/opt/pbs/bin/qstat','-xf','-F','json',job],
                                            text=True,timeout=40))['Jobs'][job]
    host=os.uname().nodename.split('.')[0]
    selection=q['Resource_List']
    if (q['job_state']!='R' or q['project']!=config['project'] or q['Job_Owner'].split('@')[0]!=os.environ['USER']
            or q['exec_host'].split('/')[0].split('.')[0].lower()!=host.lower() or int(selection['ngpus'])!=1
            or wall_seconds(selection['walltime'])!=config['max_job_seconds']
            or jax.default_backend()!='gpu' or len(jax.local_devices())!=1
            or 'A40' not in jax.local_devices()[0].device_kind):
        raise PermissionError('Formal evaluation requires a bound, single VANDA A40 job')
    visible=os.environ['CUDA_VISIBLE_DEVICES']
    uuid=subprocess.check_output(['nvidia-smi','-i',visible,'--query-gpu=uuid','--format=csv,noheader'],
                                 text=True,timeout=20).strip()
    if not uuid.startswith('GPU-') or ',' in visible:
        raise ValueError('Unexpected visible device')
    return dict(job=job,host=host,uuid=uuid,deadline=min(DEADLINE,config['grant_end'],
                site_epoch(q['stime'])+config['max_job_seconds']))


def evaluate(run, mode):
    from go2arx5_vla.sla_adapter.scale20k_evaluation import bytes_sha

    spec=read(run/'SPEC.json')
    cfg=pinned(spec['config'])
    endpoint=pinned(spec['endpoint'])
    if endpoint['purpose']!='formal' or endpoint['eval_due'] is not True:
        raise ValueError('No formal endpoint request')
    site=device(run,cfg)
    out=run/mode
    out.mkdir()
    previous={}
    attempted=Counter()
    generated=Counter()
    exact=0
    rows=[]
    def guard():
        if time.time()>=site['deadline']-90 or (run/'STOP').exists() or (Path(cfg['control'])/'STOP').exists():
            raise TimeoutError('End of registered evaluation window')
    def before(kind, ordinal):
        guard()
        if attempted[kind]>=COUNTS[mode][kind]:
            raise ValueError('Formal generation quota exhausted')
        write_once(out/f'{kind}-{ordinal:03}.intent.json',dict(kind=kind,ordinal=ordinal,phase=endpoint['phase']))
        attempted[kind]+=1
    def after(kind, ordinal):
        generated[kind]+=1
        write_once(out/f'{kind}-{ordinal:03}.generated.json',dict(kind=kind,ordinal=ordinal))
    def archive(kind,ordinal,row,normalized,decoded,metrics):
        nonlocal exact
        file=out/f'{kind}-{ordinal:03}.npz'
        with file.open('xb') as f:
            np.savez(f,normalized=normalized,decoded=decoded)
            f.flush()
            os.fsync(f.fileno())
        record=dict(row,kind=kind,ordinal=ordinal,normalized_sha256=bytes_sha(normalized),
                    decoded_sha256=bytes_sha(decoded),file=file.name,file_sha256=digest(file),metrics=metrics)
        if mode=='formal-restore':
            prior=previous[(kind,ordinal)]
            if (any(record[k]!=prior[k] for k in (*row,'normalized_sha256','decoded_sha256'))
                    or digest(run/'formal'/prior['file'])!=prior['file_sha256']):
                raise ValueError(f'Independent cold output mismatch at {kind}/{ordinal}')
            exact+=1
        rows.append(record)
    guard()
    began=time.monotonic()
    try:
        if mode=='formal-restore':
            original=read(run/'formal/RESULT.json')
            if (original['site']!=site or original['endpoint']!=spec['endpoint'] or original['status']!='SEALED'
                    or original['generated']!=COUNTS['formal'] or len(original['rows'])!=376):
                raise ValueError('Missing a complete ordinary pass on this device')
            previous={(r['kind'],r['ordinal']):r for r in original['rows']}
        graphs,state,frozen=load_endpoint(cfg,endpoint)
        panels=FixedPanels(Path(cfg['train_assets']),Path(cfg['eval_assets']))
        if {k:len(v) for k,v in panels.rows(mode).items()}!=COUNTS[mode]:
            raise ValueError('Unexpected frozen evaluation population')
        panels.evaluate(mode,graphs,state,frozen,before_generation=before,after_generation=after,archive=archive)
        if attempted!=COUNTS[mode] or generated!=attempted or (mode=='formal-restore' and exact!=48):
            raise ValueError('Incomplete panel or independent restore')
        status,error='SEALED',None
    except Exception as exc:
        status,error='FAILED',f'{type(exc).__name__}: {exc}'
    result=dict(plan=PLAN,endpoint=spec['endpoint'],mode=mode,phase=endpoint['phase'],checkpoint=endpoint['checkpoint'],
                panel=FORMAL_PANEL,site=site,status=status,error=error,attempted=dict(attempted),generated=dict(generated),
                exact=exact,rows=rows,wall_s=time.monotonic()-began,updates=0,optimizer_visits=0,native=0)
    write_once(out/'RESULT.json',result)
    return 0 if status=='SEALED' else 1


def batch(run):
    with closing(OwnerLock(run/'owner.lock')):
        for mode in COUNTS:
            child=subprocess.run([sys.executable,'-B','-m','scripts.sla.hopper_formal_outputs',str(run),mode])
            if child.returncode:
                write_once(run/'RESULT.json',dict(status='OFFLINE_EVAL_FAILED',first_failure_mode=mode,
                           child_status=child.returncode,job=os.environ.get('PBS_JOBID')))
                return child.returncode
        row=read(run/'formal-restore/RESULT.json')
        write_once(run/'RESULT.json',dict(status='FORMAL_424_SEALED',job=row['site']['job'],checkpoint=row['checkpoint'],
                   ordinary=288,phase_TRAIN=88,independent=48,exact=row['exact'],optimizer_updates=0,native=0))
        return 0


if __name__=='__main__':
    c=argparse.ArgumentParser()
    c.add_argument('run',type=Path)
    c.add_argument('mode',choices=['batch',*COUNTS])
    a=c.parse_args()
    os.umask(0o077)
    raise SystemExit(batch(a.run) if a.mode=='batch' else evaluate(a.run,a.mode))
