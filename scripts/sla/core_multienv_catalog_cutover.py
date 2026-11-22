"""Drain the disk-paused original owner and seal its final TRAIN batch on Core."""
import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.pi0292.durable import OwnerLock, read
from scripts.pi0294.e1.journal import Journal
from scripts.sla import core_multienv_train as contract
from scripts.sla.core_multienv_catalog import admission_rows, producer_binding, run as catalog_run
from scripts.sla.core_multienv_payload import capacity
from scripts.sla.core_multienv_wave_suffix import readback
from scripts.sla.hopper_parent_custody import digest, write_once
from scripts.sla.hopper_site import pinned


def pin(path):
    return dict(path=str(path),sha256=digest(path))


def live(root):
    unit = 'wm02-'+root.name+'.service'
    text = subprocess.check_output(['systemctl','--user','show',unit,'-p','MainPID','-p',
                                    'ActiveState','-p','InvocationID'],text=True,timeout=5)
    return dict(row.split('=',1) for row in text.splitlines())


def validate_initial(config):
    # Resolve the legacy object writer before asking the producer to drain.
    from scripts.sla.core_hopper_snapshot import object_file

    if not callable(object_file):
        raise ImportError('The retained catalog object writer is unavailable')
    core = pinned(config['catalog'])
    producer, cards, _ = producer_binding(core)
    spec = pinned(core['producer'])
    original_catalog = Path(core['root'])
    pilot = Path(pinned(config['pilot'])['registered_root'])
    roots = {'wave':producer,'catalog':original_catalog,'pilot':pilot}
    if any(live(path)!=config['old_owners'][key] or (path/'STOP.json').exists()
           for key,path in roots.items()):
        raise PermissionError('An original service changed since the approved cold handoff preview')
    status = read(producer/'PROGRESS.private.json')
    wave_cap = read(producer/'CAPACITY.private.json')
    head = read(original_catalog/'HEARTBEAT.json')
    repair = read(pilot/'PROGRESS.private.json')
    closure = pinned(config['previous_rehearsal'])
    ledger = readback(producer)
    rows, paid = admission_rows(producer,cards)
    count = head['latest']['count']-contract.PREFIX_COUNT
    remaining = rows[count:]
    first = [r['condition'] for r in ledger['pending'][:2]]
    if (status['stage']!='WAIT_CAPACITY_OR_STUDENT_HANDOFF' or status['current'] is not None
            or time.time()-status['epoch']>45 or wave_cap['free_disk']>=wave_cap['floor']+wave_cap['reserve']
            or ledger['original_records_sha256']!=config['records_sha256'] or first!=config['first_conditions']
            or closure['status']!='COLD_ORIGINAL_NEXT2_CASE_CONTRACT_VALIDATED'
            or [r['condition'] for r in closure['cases']]!=first
            or head['latest']['sha256']!=config['latest_seal'] or count<=0 or not 1<=len(remaining)<=4
            or paid!=head['paid_TRAIN'] or len(rows)!=head['new_admitted']
            or repair['stage']!='WAIT_ORIGINAL_SLOT_HANDOFF'
            or Journal(pilot/'control'/'journal',pilot.name).path(repair['condition']).is_file()
            or (pilot/'cases'/repair['condition']).exists() or (pilot/'RESULT.private.json').exists()
            or not any(r['condition']==repair['condition'] and r['slot']==26 for r in ledger['paused'])):
        raise PermissionError('Expected the original idle disk fence, native ledger and unpaid repair slot')
    required = sum((origin/ref['file']).stat().st_size for ep,_,origin in remaining
                   for ref in [ep['labels'],*ep['frame_shards']])
    volumes = capacity(producer,config['payload_storage'],spec['multienv'])
    local = volumes['local']
    if not volumes['passed'] or local['free_disk']<local['floor']+local['reserve']+required:
        raise PermissionError('Cannot drain both original owners without room for the final TRAIN batch')
    return roots, core, spec, dict(conditions=first,original_paid=paid,original_admitted=len(rows),
                                  remaining_admitted=len(remaining),remaining_bytes=required,volumes=volumes)


def finished(root, name):
    file = root/name
    state = live(root)
    return file.is_file() and state['MainPID']=='0' and state['ActiveState']=='inactive'


def run(root, config, probe):
    if (os.uname().nodename!='corelab-4080s' or os.getuid()!=2009 or config['root']!=str(root)
            or config['plan']!=contract.PLAN or time.time()+120>=config['deadline']
            or (not probe and not os.environ.get('INVOCATION_ID'))):
        raise PermissionError('Expected the finite Core cutover service')
    original = pinned(pinned(config['catalog'])['producer'])
    with closing(OwnerLock(original['mobile_world_lock'])):
        roots, core, _, proof = validate_initial(config)
        if probe:
            return dict(status='ORIGINAL_CUTOVER_PREFLIGHT',**proof)
        if (root/'STOP').exists():
            raise InterruptedError('Cutover stop before draining the original role')
        write_once(root/'INTENT.private.json',dict(proof,original=core['producer'],pilot=config['pilot'],
                   native_requests=0,native_ticks=0,epoch=time.time()))
        write_once(roots['wave']/'STOP.json',dict(reason='PR425_WAVE_A_DISK_COLD_CATALOG_CUTOVER',
                   coordinator=str(root),old_records=config['records_sha256'],epoch=time.time()))
        ends = {'wave':'RESULT.private.json','catalog':'TERMINAL.json','pilot':'RESULT.private.json'}
        until = min(config['deadline']-25,time.time()+120)
        while not all(finished(roots[k],file) for k,file in ends.items()):
            if time.time()>=until:
                raise TimeoutError('Read the original termination receipts before any new submission')
            time.sleep(2)
        retained = {k:pin(roots[k]/file) for k,file in ends.items()}
        pilot = pinned(retained['pilot'])
        old = pinned(retained['wave'])
        if (pilot['native_state']!='NOT_RUN' or pilot['native_requests']!=0 or pilot['exit_code']
                or old['paid_requests']!=proof['original_paid'] or old['unreconciled'] or old['exit_code']):
            raise RuntimeError('Original paid work or the unpaid repair slot requires manual reconciliation')
        write_once(root/'ORIGINAL_CLOSED.private.json',retained)
    if (root/'STOP').exists():
        raise InterruptedError('Cutover stop after the pinned predecessor terminals')
    source = Path(__file__).resolve().parents[2]
    merged = dict(core,root=str(root),source_root=str(source),source_members=config['source_members'],
                  close_original=dict(terminal=retained['wave'],catalog=config['catalog'],
                     catalog_terminal=retained['catalog'],payload_storage=config['payload_storage']))
    write_once(root/'CATALOG_CONFIG.private.json',merged)
    catalog_run(root,merged,False)
    return read(root/'TERMINAL.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('sha')
    parser.add_argument('--probe',action='store_true')
    a=parser.parse_args()
    os.umask(0o077)
    print(json.dumps(run(a.root,pinned(dict(path=str(a.root/'CONFIG.json'),sha256=a.sha)),a.probe)))
