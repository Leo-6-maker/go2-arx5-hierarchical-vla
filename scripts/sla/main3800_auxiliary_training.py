"""Run one finite independently registered auxiliary-selection arm from MAIN3800."""
from argparse import ArgumentParser
from contextlib import closing
import os
from pathlib import Path
import signal
import time

import numpy as np

from scripts.pi0292.durable import OwnerLock, write
from scripts.pi0294.e1.journal import Journal
from scripts.sla import hopper_checkpoint as checkpoint
from scripts.sla.hopper_inputs import BATCH, PLAN
from scripts.sla.hopper_parent_custody import digest, write_once
from scripts.sla.hopper_site import live_h200, pinned
from scripts.sla.hopper_training import initial_state
from scripts.sla.main3800_visit_audit import TRAIN052


ARMS = ('A','B')
START, STOP = 3800,4300
PARENT = '502d890d3ff314ac42f4c79c360942b9b626bf1fcc4b6fc488fbab15ba508361'
SCHEMA = 'SLA_MAIN3800_MATCHED_AUXILIARY_500_V1'


def plan_rows(spec):
    receipt = pinned(spec['selection_plan'])
    design = receipt['possible_auxiliary_design']
    if receipt['adopted']!=TRAIN052 or design['updates_per_arm']!=500 or design['aux_per_arm']!=2000:
        raise ValueError('Auxiliary roster is not the fixed 500-step target FM population')
    rows = design['selections']
    if [r['update'] for r in rows] != list(range(START+1,STOP+1)):
        raise ValueError('Matched roster lost the original absolute cursors')
    for r in rows:
        shift = 4*((2000+r['update']-1)%16)
        if (r['A']!=list(range(shift,shift+4)) or len(r['B'])!=4 or len(set(r['B']))!=4
                or r['B'][2:]!=r['A'][2:] or any(not 0<=i<BATCH['target'] for i in r['B'])):
            raise ValueError('The matched intervention may replace only two auxiliary slots')
    return rows


def update(engine, guard, row, arm, job):
    from go2arx5_vla.sla_adapter.scale20k_contract import bank_sha256

    if row['update']!=engine.step+1 or bank_sha256(engine.indices(engine.step,'target')[0])!=row['target_slice_sha256']:
        raise ValueError('The original target FM visit bank changed')
    before = dict(engine.gradient_windows)
    attempts, complete = engine.attempted_aux,engine.completed_aux
    stats = engine.update(guard,auxiliary_slots=row['B'] if arm=='B' else None)
    visits = {d:n-before[d] for d,n in engine.gradient_windows.items()}
    if (engine.step!=row['update'] or visits!=BATCH or engine.attempted_aux-attempts!=4
            or engine.completed_aux-complete!=4 or tuple(stats['domains']['target']['auxiliary_slots'])!=tuple(row[arm])):
        raise ValueError('Actual update or auxiliary slots differ from the matched budget')
    return dict(stats,visits=visits,arm=arm,job=job)


def save(path, engine, bound, arm, spec, predecessor):
    import jax
    from go2arx5_vla.sla_adapter import scale20k_checkpoint as original

    state = original.training_tree(engine.parameters,engine.states,engine.rng)
    clocks = checkpoint.progress(bound,engine.step)
    checkpoint.check_state(state,clocks,bound)
    result = original.seal_arrays(path,state,dict(schema=SCHEMA,arm=arm,inherited_binding=bound,
                  progress=clocks,predecessor=predecessor,comparison_parent=PARENT,roster=spec['selection_plan']))
    if engine.step==STOP:
        raw = path/'COMPLETE.json'
        marker = pinned(dict(path=str(raw),sha256=result['sha256']))
        leaves = jax.tree_util.tree_flatten_with_path(state)[0]
        for (key,live),(_,disk) in zip(leaves,original.verified_arrays(path,marker,leaves),strict=True):
            if not np.array_equal(np.asarray(live).ravel().view(np.uint8),disk.ravel().view(np.uint8)):
                raise ValueError(f'Endpoint array failed exact restore readback: {key}')
    return result


def worker(root, mount_sha):
    mount = pinned(dict(path=str(root/'MOUNT.private.json'),sha256=mount_sha))
    spec = pinned(mount['auxiliary_experiment'])
    parent = spec['original_main_segment']
    gate = pinned(spec['native_gate'])
    arm = spec['arm']
    if (spec['schema']!=SCHEMA or arm not in ARMS or spec['start']!=START or spec['stop']!=STOP
            or spec['updates_per_arm']!=500 or spec['physics'] or spec['hardware_connections_allowed']
            or parent['previous']['sha256']!=PARENT or parent['role']!='MAIN' or parent['role_start']!=0
            or parent['plan']!=PLAN or gate['native_requests']!=6 or gate['selected_intervention']!='AUXILIARY_SLOT_SELECTION'
            or gate['parent']!=PARENT or gate['checkpoint_phase']!=START or gate['scientific_failure_retries']):
        raise PermissionError('The native result and bounded independent training role are required')
    rows = plan_rows(spec)
    if arm=='B':
        control = Path(mount['comparison_root'])/'A'/'RESULT.private.json'
        a = pinned(dict(path=str(control),sha256=digest(control)))
        if (a['status']!='COMPLETE' or a['arm']!='A' or a['site']['job']!=mount['control_job']
                or a['updates']!=500 or a['exact_arrays']!=140
                or digest(Path(a['checkpoint']['path'])/'COMPLETE.json')!=a['checkpoint']['sha256']):
            raise PermissionError('The matched control must first seal its qualified endpoint')
    site = live_h200(mount,'training')
    stopped = False

    def stop_signal(*_):
        nonlocal stopped
        stopped = True

    def guard():
        if stopped or (root/'STOP.json').exists() or time.time()>=site['deadline']-900:
            raise TimeoutError('Finite auxiliary comparison stopped before the actual deadline')

    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,stop_signal)
    with (closing(OwnerLock(Path(mount['comparison_root'])/'OWNER.lock')),
          closing(OwnerLock(Path(mount['lineage'])/'MAIN/OWNER.lock'))):
        guard()
        write_once(root/'START.private.json',dict(site=site,arm=arm,parent=PARENT,source=mount['source_revision']))
        import jax

        jax.config.update('jax_compilation_cache_dir',mount['compilation_cache_dir'])
        engine, binding = initial_state(Path(mount['custody']),parent,START,STOP)
        journal = Journal(root/'updates',SCHEMA+'-'+arm)
        checkpoints = root/'checkpoints'
        checkpoints.mkdir()
        previous = dict(sha256=PARENT)
        retained = START
        status = 'COMPLETE'
        try:
            for row in rows:
                phase = row['update']
                try:
                    guard()
                except TimeoutError:
                    status = 'FINITE_STOP'
                    break
                key = f'phase{phase:05d}'
                if journal.path(key).exists():
                    raise RuntimeError('Existing effect at this phase requires explicit reconciliation')
                actual = journal.once(key,dict(arm=arm,row=row,parent=PARENT,spec=mount['auxiliary_experiment']),
                                      lambda: update(engine,guard,row,arm,mount['job']))
                write(root/'PROGRESS.private.json',dict(phase=phase,visit_cursors=actual['phase_cursors'],epoch=time.time()))
                if phase in (4050,STOP):
                    previous = save(checkpoints/key,engine,binding,arm,spec,previous['sha256'])
                    retained = phase
            if engine.step>retained:
                previous = save(checkpoints/f'phase{engine.step:05d}',engine,binding,arm,spec,previous['sha256'])
                retained = engine.step
        finally:
            engine.close()
        write_once(root/'RESULT.private.json',dict(status=status,arm=arm,site=site,updates=engine.step-START,
                  visits=engine.gradient_windows,aux_attempts=engine.attempted_aux,aux_complete=engine.completed_aux,
                  checkpoint=previous,checkpoint_phase=retained,exact_arrays=140 if retained==STOP else None,native_requests=0))
        if status!='COMPLETE':
            raise TimeoutError('The bounded arm stopped before producing a complete matched endpoint')


if __name__=='__main__':
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('root',type=Path)
    cli.add_argument('mount_sha')
    args = cli.parse_args()
    os.umask(0o077)
    try:
        worker(args.root,args.mount_sha)
    except Exception as exc:
        write_once(args.root/'REVIEW_REQUIRED.json',dict(reason=f'{type(exc).__name__}: {exc}',
                   journal_records=len(list(args.root.glob('updates/phase*.json'))),epoch=time.time()))
        raise
