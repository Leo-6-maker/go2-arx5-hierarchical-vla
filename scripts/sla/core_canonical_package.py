"""Export the original Go2 parameters from a full checkpoint without generating predictions."""
import argparse
from contextlib import closing
import os
from pathlib import Path
import tempfile

from scripts.pi0292.durable import OwnerLock
from scripts.sla.hopper_endpoint_backup import SCHEMA as STATE_SCHEMA, checkpoint
from scripts.sla.hopper_parent_custody import PARENT, PLAN, digest, sync_dir, write_once
from scripts.sla.hopper_site import pinned

PROTOCOL = 'SLA_CORE_CANONICAL_EVAL_V1'
CANONICAL_PLAN = '4fe2c8cf1b69c168b2e3079e061bb7c702089cd3'
PANEL = '3136404c06f890a9dd830b2cf5f40e487ef436ede16e805e14ffc0cb06b5eefd'
SOURCE = 'b9581196d0acd02fb748222988e5a5543ce1f232'
PHASES = (0, 2000, 5000, 10000, 20000, 4000, 7000, 12000, 17000)
TARGET_PREFIXES = ("['parameters']['shared']", "['parameters']['target']")


def rows(header):
    if header['schema'] != STATE_SCHEMA:
        raise ValueError('Export requires a full 140-array sealed training state')
    return go2_rows(header['arrays'])


def go2_rows(arrays):
    if len(arrays) != 140:
        raise ValueError('Export requires exactly 140 complete-state arrays')
    if [r['file'] for r in arrays] != [f'{i:06d}.npy' for i in range(len(arrays))]:
        raise ValueError('Unexpected complete-state array file order')
    chosen = [r for r in arrays if r['key'].startswith(TARGET_PREFIXES)]
    if (len(chosen) != 29 or len({r['key'] for r in chosen}) != 29
            or any(r['dtype'] != 'float32' for r in chosen)):
        raise ValueError('The complete state does not map to the 29 original Go2 arrays')
    return [dict(r, file='parameters/'+r['file']) for r in chosen]


def validate(root, sha):
    m = pinned({'path':str(root/'CHECKPOINT.json'), 'sha256':sha})
    header = pinned({'path':str(root/'COMPLETE.json'), 'sha256':m['checkpoint_sha256']})
    b = header['binding']
    if (m['schema']!=PROTOCOL or m['canonical_plan']!=CANONICAL_PLAN or m['plan']!=PLAN
            or m['parent']!=PARENT or m['panel']!=PANEL or m['reference']
            or m['physics'] or m['hardware_allowed'] or m['native_admission']
            or (m['phase'], m['role'], m['frozen_model'], m['training_source']) !=
               (header['progress']['phase_step'], b['role'], b['frozen_model'], b['source'])
            or m['training_source']!=SOURCE or m['phase'] not in PHASES
            or m['arrays']!=rows(header)):
        raise ValueError('Canonical package and complete-state mapping differ')
    names = {Path(r['file']).name for r in m['arrays']}
    if (set(p.name for p in (root/'parameters').iterdir())!=names
            or {p.name for p in root.iterdir()}!={'parameters','CHECKPOINT.json','COMPLETE.json'}):
        raise ValueError('Unexpected canonical package members')
    if any((root/r['file']).is_symlink() or digest(root/r['file'])!=r['sha256'] for r in m['arrays']):
        raise ValueError('Canonical parameter file SHA mismatch')
    return m


def export(line, parent, dest, role, phase, event_sha, action_sha):
    match = checkpoint(line, role, phase, event_sha)
    event = match['event']
    if not event['deployment_due'] or event['binding']['source']!=SOURCE:
        raise PermissionError('The sealed endpoint is outside the original deployment population')
    cp = Path(match['folder'])
    header_sha = event['checkpoint']['sha256']
    header = pinned({'path':str(cp/'COMPLETE.json'),'sha256':header_sha})
    bound = header['binding']
    ancestor = pinned({'path':str(parent),'sha256':PARENT})['binding']
    phase_zero = pinned({
        'path':str(line/'MAIN/checkpoints/phase00000/COMPLETE.json'),
        'sha256':'d7949c75ecc33253bea605a3aed738df4ecdc473d8737848a1bca787ab253c18',
    })['binding']
    if (bound['frozen_model']!=phase_zero['frozen_model']
            or action_sha!=ancestor['frozen_C20_action_statistics_sha256']):
        raise ValueError('The sealed endpoint changed the frozen model or action statistics')
    expected = dict(schema=PROTOCOL,canonical_plan=CANONICAL_PLAN,plan=PLAN,parent=PARENT,
        role=role,phase=phase,checkpoint_sha256=header_sha,training_source=SOURCE,panel=PANEL,
        frozen_model=bound['frozen_model'],foundation_sha256=ancestor['parent_manifest_sha256'],
        partition_sha256=ancestor['partition_sha256'],state_sha256=ancestor['target_state_sha256'],
        tokenizer_sha256=ancestor['tokenizer_sha256'],action_sha256=action_sha,event_sha256=event_sha,
        reference=[],arrays=rows(header),physics=False,hardware_allowed=False,native_admission=False)
    dest.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    with closing(OwnerLock(dest.parent/'export.lock')):
        if not dest.exists():
            # Both locations are in private custody on the same filesystem.
            stage = Path(tempfile.mkdtemp(dir=dest.parent, prefix='.export-'))
            (stage/'parameters').mkdir()
            os.link(cp/'COMPLETE.json',stage/'COMPLETE.json')
            for r in expected['arrays']:
                os.link(cp/Path(r['file']).name,stage/r['file'])
            write_once(stage/'CHECKPOINT.json',expected)
            sha = digest(stage/'CHECKPOINT.json')
            validate(stage,sha)
            sync_dir(stage/'parameters')
            sync_dir(stage)
            stage.rename(dest)
            sync_dir(dest.parent)
        sha = digest(dest/'CHECKPOINT.json')
        if pinned({'path':str(dest/'CHECKPOINT.json'),'sha256':sha})!=expected:
            raise ValueError('Existing canonical package conflicts with the sealed export')
        return dict(phase=phase,checkpoint=header_sha,event=event_sha,package=sha,
                    complete_state_arrays=140,exported_go2_arrays=29,model_generations=0,native_requests=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('line',type=Path)
    p.add_argument('parent',type=Path)
    p.add_argument('dest',type=Path)
    p.add_argument('role')
    p.add_argument('phase',type=int)
    p.add_argument('event_sha')
    p.add_argument('action_sha')
    a=p.parse_args()
    import json
    os.umask(0o077)
    print(json.dumps(export(a.line,a.parent,a.dest,a.role,a.phase,a.event_sha,a.action_sha)))
