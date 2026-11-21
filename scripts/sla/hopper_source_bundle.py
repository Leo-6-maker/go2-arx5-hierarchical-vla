"""Build an inspectable closure from pinned git source for the new server capsules."""
import argparse
import ast
import hashlib
from importlib.util import resolve_name
import io
import json
from pathlib import Path
import subprocess
import tarfile

BASE = '1432c8dc88c57adf39e8fa363f5438c5e63aa8fd'
PERFORMANCE = '2caa24b2447e41f6a64323fadc1a428c72d583db'
KERNEL = 'src/go2arx5_vla/sla_adapter/scale20k_training.py'
SERIAL = 'src/go2arx5_vla/sla_adapter/hopper_serial_reference.py'


def git(*args):
    return subprocess.check_output(['git',*args])


def files(rev):
    return {p for p in git('ls-tree','-r','--name-only',rev,'src','scripts').decode().splitlines()
            if p.endswith('.py')}


def build(revision, entries, output):
    if git('rev-parse',revision).decode().strip() != revision:
        raise ValueError('Use a full immutable source revision')
    origins = dict.fromkeys(files(BASE),BASE)
    origins[KERNEL] = PERFORMANCE
    origins[SERIAL] = BASE
    new = {p for p in files(revision) if p.startswith('scripts/sla/hopper_')}
    new.add('src/go2arx5_vla/sla_adapter/adapt5k_contract.py')
    origins.update(dict.fromkeys(new,revision))
    mapping = {}
    for name in origins:
        dotted=name.removeprefix('src/').removesuffix('.py').replace('/','.')
        mapping[dotted.removesuffix('.__init__')]=name
    pending = list(entries)
    if 'scripts/sla/hopper_bank_custody.py' in entries:
        pending.append('scripts/sla/hopper_bank_receipt.py')
    sources = {}
    while pending:
        name=pending.pop()
        if name in sources:
            continue
        body=git('show',origins[name]+':'+(KERNEL if name==SERIAL else name))
        sources[name]=body
        dotted=name.removeprefix('src/').removesuffix('.py').replace('/','.')
        package=dotted.rpartition('.')[0]
        modules=set()
        for node in ast.walk(ast.parse(body,filename=name)):
            if isinstance(node,ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node,ast.ImportFrom):
                prefix=resolve_name('.'*node.level+(node.module or ''),package) if node.level else node.module
                modules.add(prefix)
                modules.update(prefix+'.'+alias.name for alias in node.names)
        for mod in modules | {dotted}:
            while mod:
                if mod in mapping and mapping[mod] not in sources:
                    pending.append(mapping[mod])
                mod=mod.rpartition('.')[0]
    proof = {name:dict(sha256=hashlib.sha256(body).hexdigest(),revision=origins[name]) for name,body in sources.items()}
    seal=json.dumps(dict(new_revision=revision,modules=proof),sort_keys=True,indent=2).encode()
    with tarfile.open(output,'x:gz') as tar:
        for path,body in sorted({**sources,'SOURCE_BINDING.private.json':seal}.items()):
            info=tarfile.TarInfo(path)
            info.size=len(body)
            info.mode=0o600
            tar.addfile(info,io.BytesIO(body))
    return len(sources)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('revision')
    parser.add_argument('output',type=Path)
    parser.add_argument('entries',nargs='+')
    a=parser.parse_args()
    print(build(a.revision,a.entries,a.output))
