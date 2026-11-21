"""Freeze the start banks on a VANDA CPU allocation, then deliver them to Hopper."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

from scripts.sla.hopper_inputs import BASE, D2_SPEC, PLAN, SPEC_NAME, check_samples, freeze_banks, inputs
from scripts.sla.hopper_parent_custody import HOST, SSH, digest, remote, write_once


def run(root):
    config = json.loads((root/'MOUNT.private.json').read_bytes())
    if (os.environ.get('PBS_JOBID') != config['job'] or config['plan']!=PLAN
            or time.time() >= config['deadline_epoch'] or (root/'STOP').exists()):
        raise RuntimeError('Bank preparation requires its own live CPU allocation')
    if digest(root/'source.tar.gz') != config['bundle_sha256']:
        raise ValueError('Source bundle changed')
    source=root/'source'
    for name, row in json.loads((source/'SOURCE_BINDING.private.json').read_bytes())['modules'].items():
        if digest(source/name) != row['sha256']:
            raise ValueError('Extracted bank preparation source changed')
    if digest(BASE/SPEC_NAME) != D2_SPEC:
        raise ValueError('Initial training spec changed')
    spec = inputs(BASE)
    datasets,proof = check_samples(spec)
    out=root/'banks'
    out.mkdir()
    previous=BASE/'sla-adapt5k-d1-single-lineage-vanda-20261120-01/slaadp2000-02/train'
    ledger=out/'PARENT_UPDATES.private.jsonl'
    shutil.copyfile(previous/'UPDATES.private.jsonl',ledger)
    seed=json.loads((BASE/SPEC_NAME).read_bytes())['target_seed']
    result=freeze_banks(datasets,previous/'phase02000',ledger,seed,out)
    result.update(checked_train_samples=proof,site='VANDA',job=config['job'],source=config['source_revision'],
                  updates=0,optimization_visits=0,generations=0,native=0)
    write_once(out/'BANKS.private.json',result)
    files=sorted(out.iterdir())
    seal=''.join(digest(p)+'  '+p.name+'\n' for p in files)
    (out/'SHA256SUMS').write_text(seal)
    dest=Path(config['receiver'])
    incoming=dest.with_name(dest.name+'.incoming')
    remote('bash','-c','set -eu; umask 077; test ! -e "$2"; mkdir -- "$1"','bash',str(incoming),str(dest))
    subprocess.run(['rsync','-rt','--chmod=Fu=rw,Fgo=,Du=rwx,Dgo=','-e',shlex.join(SSH),
                    str(out)+'/',HOST+':'+str(incoming)+'/'],check=True)
    check='set -eu; cd "$1"; test "$(sha256sum SHA256SUMS | cut -d" " -f1)" = "$3"; sha256sum -c SHA256SUMS >/dev/null; sync -f .; cd ..; test ! -e "$2"; mv -T "$1" "$2"; sync -f .; sha256sum "$2/BANKS.private.json"'
    actual=remote('bash','-c',check,'bash',str(incoming),str(dest),hashlib.sha256(seal.encode()).hexdigest())
    if actual.split()[0]!=digest(out/'BANKS.private.json'):
        raise ValueError('Hopper bank seal readback differs')
    write_once(root/'RESULT.private.json',dict(remote_banks_sha256=actual.split()[0],**result))


if __name__=='__main__':
    os.umask(0o077)
    root=Path.cwd()
    with (root/'OWNER.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            run(root)
        except Exception as error:
            write_once(root/'FAILURE.private.json',dict(job=os.environ.get('PBS_JOBID'),error=str(error)))
            raise
