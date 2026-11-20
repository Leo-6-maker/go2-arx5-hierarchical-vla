"""Reconcile and stop only explicitly registered Core owners at the Goal V2 boundary."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


PLAN = '1b2a25b7c1720051a2627d30831fb18130462374'
DAY = 86400


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def read_pin(pin):
    path = Path(pin['path'])
    if not path.is_file() or sha(path) != pin['sha256']:
        raise ValueError('GOAL_FINALIZER_PIN_CHANGED:'+path.name)
    return read(path)


def write_once(path, body):
    with path.open('x') as output:
        json.dump(body,output,indent=2,allow_nan=False)
        output.flush()
        os.fsync(output.fileno())


def unit_state(name):
    if not re.fullmatch(r'(?:wm02-|sla-observer-)[a-zA-Z0-9-]+\.service',name):
        raise ValueError('GOAL_FINALIZER_UNRECOGNIZED_CORE_OWNER')
    value = subprocess.run(['systemctl','--user','show',name,'-p','Id','-p','LoadState',
                            '-p','MainPID','-p','ControlGroup','-p','InvocationID',
                            '-p','ActiveState','-p','ExecStart'],capture_output=True,text=True,timeout=12)
    if value.returncode:
        raise RuntimeError('GOAL_FINALIZER_CANNOT_READ_SYSTEMD_OWNER')
    fields = dict(line.split('=',1) for line in value.stdout.splitlines() if '=' in line)
    if fields.get('Id') != name or not fields.get('MainPID','').isdigit():
        raise ValueError('GOAL_FINALIZER_INCOMPLETE_SYSTEMD_OWNER')
    return fields


def scoped_path(base, entry, uid):
    root = Path(entry['root']).resolve()
    if not root.is_relative_to(base) or root == base or (root.exists() and root.stat().st_uid != uid):
        raise PermissionError('GOAL_FINALIZER_ROOT_OUTSIDE_PROJECT')
    path = root/entry['spec_file']
    if path.parent != root or path.is_symlink():
        raise PermissionError('GOAL_FINALIZER_REPORT_CARD_REQUIRED')
    return root,path


def inspect(entry, config, *, stop=False):
    uid = config['uid']
    root,card = scoped_path(Path(config['project_root']).resolve(strict=True),entry,uid)
    unit = unit_state(entry['unit'])
    live = int(unit['MainPID']) > 0
    status = 'ACTIVE' if live else 'COLD'
    observed = dict(role=entry['role'],unit=entry['unit'],pid=int(unit['MainPID']),
                    cgroup=unit.get('ControlGroup'),state=unit.get('ActiveState'),
                    result=None,stop_written=False)
    if not card.is_file():
        if live or unit.get('ControlGroup'):
            raise PermissionError('GOAL_FINALIZER_ACTIVE_OWNER_CARD_MISSING')
        return dict(observed,status='UNSTARTED_NO_CARD')
    digest = sha(card)
    if entry.get('spec_sha256') and digest != entry['spec_sha256']:
        raise ValueError('GOAL_FINALIZER_OWNER_CARD_CHANGED')
    spec = read(card)
    own_root = spec.get('registered_root',spec.get('root'))
    if own_root != str(root):
        raise PermissionError('GOAL_FINALIZER_CARD_ROOT_MISMATCH')
    if entry.get('expected_source') and spec.get('source_sha') != entry['expected_source']:
        raise PermissionError('GOAL_FINALIZER_DEFERRED_SOURCE_CHANGED')
    if live and (str(card) not in unit.get('ExecStart','')
                 or not unit['ControlGroup'] or unit['ActiveState'] != 'active'):
        raise PermissionError('GOAL_FINALIZER_SYSTEMD_OWNER_CHANGED')
    observed['spec_sha256'] = digest
    result_file = root/'RESULT.private.json'
    if result_file.is_file():
        result = read(result_file)
        observed['result'] = dict(sha256=sha(result_file),status=result.get('status',result.get('schema')),
            exit_code=result.get('exit_code'),first_failure=result.get('first_failure',result.get('failure')))
    lease = root/'control/LEASE.json'
    if lease.is_file():
        data = read(lease)
        if data.get('spec_sha256',digest) != digest:
            raise PermissionError('GOAL_FINALIZER_LEASE_CARD_CHANGED')
        observed['lease'] = data.get('status')
    stop_path = root/'STOP.json'
    if stop and (live or observed['result'] is None) and not stop_path.exists():
        write_once(stop_path,dict(reason='SLA_GOAL_V2_CAMPAIGN_BOUNDARY',campaign_sha256=config['campaign']['sha256'],
                                  epoch=time.time()))
        observed['stop_written'] = True
    return dict(observed,status=status)


def run(config, *, audit):
    campaign = read_pin(config['campaign'])
    root = Path(config['root']).resolve(strict=True)
    if (config['uid'] != os.getuid() or config['host'] != os.uname().nodename or root.stat().st_uid != os.getuid()
            or campaign['plan_sha'] != PLAN or campaign['deadline_epoch'] != campaign['t0_epoch']+14*DAY
            or Path(campaign['stop']).parent != root or Path(config['campaign']['path']).parent != root):
        raise PermissionError('GOAL_FINALIZER_EXACT_EXISTING_CAMPAIGN_REQUIRED')
    result = root/'CORE_FINALIZER_ACTUAL.private.json'
    if not audit and result.exists():
        return read(result)
    epoch = time.time()
    stop = Path(campaign['stop'])
    expired = epoch >= campaign['deadline_epoch']
    if not audit and not expired and not stop.is_file():
        raise PermissionError('GOAL_FINALIZER_REQUIRES_OPERATOR_STOP_OR_ACTUAL_DEADLINE')
    if not audit and not stop.exists():
        try:
            write_once(stop,dict(reason='GOAL_V2_DEADLINE',epoch=epoch,campaign_sha256=config['campaign']['sha256']))
        except FileExistsError:
            pass
    entries = config['owners']
    if len({x['unit'] for x in entries}) != len(entries) or len({x['root'] for x in entries}) != len(entries):
        raise ValueError('GOAL_FINALIZER_DUPLICATE_OWNER')
    states=[]
    for entry in entries:
        try:
            states.append(inspect(entry,config,stop=not audit))
        except (OSError,ValueError,RuntimeError) as error:
            states.append(dict(unit=entry['unit'],role=entry['role'],status='REQUIRES_RECONCILIATION',
                               error=f'{type(error).__name__}: {error}'))
    actual = dict(schema='SLA_MULTISOURCE_GOAL_V2_CORE_FINALIZER_V1',observed_epoch=epoch,audit_only=audit,
                  campaign_sha256=config['campaign']['sha256'],deadline_epoch=campaign['deadline_epoch'],
                  reason='DEADLINE' if expired else 'OPERATOR_STOP' if stop.exists() else 'READ_ONLY',
                  site='Corelab',owners=states,optimizer_updates=0,model_generations=0,native_requests=0,native_ticks=0)
    if not audit:
        # Site shutdown and the full scientific completion tally have separate evidence.
        actual['requires_post_stop_reconciliation'] = any(row['status'] in ('ACTIVE','REQUIRES_RECONCILIATION') for row in states)
        write_once(result,actual)
    return actual


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec')
    parser.add_argument('sha256')
    parser.add_argument('--audit',action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    print(json.dumps(run(read_pin(dict(path=args.spec,sha256=args.sha256)),audit=args.audit),allow_nan=False))
