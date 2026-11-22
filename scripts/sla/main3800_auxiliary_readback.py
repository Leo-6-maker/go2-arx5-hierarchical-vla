"""Reconcile both completed Hopper arms and export private Go2-only evaluation packages."""
from argparse import ArgumentParser
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import tempfile

from scripts.pi0292.durable import OwnerLock
from scripts.pi0294.e1.journal import digest as record_sha
from scripts.sla.core_canonical_package import PANEL, go2_rows
from scripts.sla.hopper_parent_custody import PARENT as ADAPT2K, digest, sync_dir, write_once
from scripts.sla.hopper_site import pinned


ARMS = ('A', 'B')
SCHEMA = 'SLA_MAIN3800_MATCHED_AUXILIARY_500_V1'
EXPORT = 'SLA_GO2_PRIVATE_AUXILIARY_ENDPOINT_V1'
PHASES = range(3801, 4301)
PER_UPDATE = dict(target=64, robodojo=32, libero=32)
TRAIN052 = 'e0c5c481a95972e4ba5c9b712e1c3a1240d462ad28222a2c0e041e307fb42278'
MAIN3800 = '502d890d3ff314ac42f4c79c360942b9b626bf1fcc4b6fc488fbab15ba508361'


def pin(path):
    return dict(path=str(path), sha256=digest(path))


def local(path):
    return pinned(pin(path))


def journal_row(file, arm, job, row, experiment):
    evidence = local(file)
    payload = dict(arm=arm, row=row, parent=MAIN3800, spec=experiment)
    phase = row['update']
    if (evidence['run_uuid'] != f'{SCHEMA}-{arm}' or evidence['status'] != 'COMPLETE'
            or evidence['message_id'] != f'phase{phase:05d}' or evidence['message_id'] != file.stem
            or evidence['payload_sha256'] != record_sha(payload)
            or evidence['result_sha256'] != record_sha(evidence['result'])):
        raise ValueError(f'Unreconciled durable update in {arm} at {phase}')
    result = evidence['result']
    target = result['domains']['target']
    if (result['arm'] != arm or result['job'] != job or result['phase_step'] != phase
            or result['ancestor_updates'] != 2500 or result['visits'] != PER_UPDATE
            or result['phase_cursors'] != {key: phase*n for key,n in PER_UPDATE.items()}
            or result['full_H40'] != 4 or target['training_full_generations'] != 4
            or target['auxiliary_slots'] != row[arm] or target['visit_slice_sha256'] != row['target_slice_sha256']):
        raise ValueError(f'Actual update, visit cursor or auxiliary slots changed in {arm} at {phase}')
    return (result['learning_rate'], tuple(result['domains'][d]['visit_slice_sha256'] for d in PER_UPDATE)), evidence


def complete(root, arm, config, design, scheduler):
    folder = root/arm
    mount = local(folder/'MOUNT.private.json')
    experiment = mount['auxiliary_experiment']
    spec = pinned(experiment)
    result = local(folder/'RESULT.private.json')
    job = mount['job']
    visits = {key: n*len(PHASES) for key,n in PER_UPDATE.items()}
    if (spec['arm'] != arm or spec['selection_plan'] != config['selection_plan']
            or spec['original_main_segment'] != config['original_main_segment']
            or scheduler['job'] != job or scheduler['state'] != 'F' or scheduler['exit'] != 0
            or result['site']['job'] != job or result['status'] != 'COMPLETE' or result['arm'] != arm
            or result['updates'] != len(PHASES) or result['visits'] != visits or result['exact_arrays'] != 140
            or result['checkpoint_phase'] != PHASES[-1] or result['native_requests']
            or result['aux_complete'] != 2000 or result['aux_attempts'] != 2000):
        raise ValueError(f'Terminal evidence is insufficient for an endpoint in {arm}')
    checkpoints = folder/'checkpoints'
    endpoint = checkpoints/'phase04300'
    header = pinned(dict(path=str(endpoint/'COMPLETE.json'), sha256=result['checkpoint']['sha256']))
    bound = header['inherited_binding']
    midpoint_file = checkpoints/'phase04050'/'COMPLETE.json'
    midpoint = pinned(dict(path=str(midpoint_file),sha256=header['predecessor']))
    if (result['checkpoint']['path'] != str(endpoint) or header['schema'] != SCHEMA or header['arm'] != arm
            or header['comparison_parent'] != MAIN3800 or header['roster'] != config['selection_plan']
            or header['progress']['phase_step'] != PHASES[-1] or bound['target_manifest'] != TRAIN052
            or midpoint['schema'] != SCHEMA or midpoint['arm'] != arm or midpoint['progress']['phase_step'] != 4050
            or midpoint['comparison_parent'] != MAIN3800 or midpoint['predecessor'] != MAIN3800
            or midpoint['roster'] != header['roster'] or midpoint['inherited_binding'] != bound):
        raise ValueError(f'Identity or checkpoint lineage changed in {arm}')
    files = {'COMPLETE.json': result['checkpoint']['sha256']}
    members = go2_rows(header['arrays'])
    files.update({r['file']:r['sha256'] for r in header['arrays']})
    if {p.name for p in endpoint.iterdir()} != files.keys():
        raise ValueError(f'Complete-state member list changed in {arm}')
    for name,sha in files.items():
        member = endpoint/name
        if member.is_symlink() or digest(member) != sha:
            raise ValueError(f'Complete-state bytes changed in {arm}: {name}')
    updates = folder/'updates'
    if {p.stem for p in updates.glob('phase*.json')} != {f'phase{n:05d}' for n in PHASES}:
        raise ValueError(f'Durable update population changed in {arm}')
    matched = []
    journal = hashlib.sha256()
    for row in design['selections']:
        key,evidence = journal_row(updates/f"phase{row['update']:05d}.json", arm, job, row, experiment)
        journal.update(bytes.fromhex(evidence['result_sha256']))
        matched.append(key)
    summary = dict(arm=arm, job=job, training_state_sha256=result['checkpoint']['sha256'], updates=result['updates'],
                   visits=visits, auxiliary_H40=2000, actual_event_slots=design['counts'][arm],
                   result_sha256=digest(folder/'RESULT.private.json'), ordered_journal_sha256=journal.hexdigest(),
                   complete_arrays=140, scheduler=scheduler)
    return endpoint, members, matched, summary, bound


def verify_package(folder, manifest_sha):
    manifest = pinned(dict(path=str(folder/'CHECKPOINT.json'), sha256=manifest_sha))
    rows = manifest['arrays']
    if (manifest['schema'] != EXPORT or manifest['arm'] not in ARMS or manifest['phase'] != PHASES[-1]
            or manifest['panel'] != PANEL or manifest['comparison_parent'] != MAIN3800 or len(rows) != 29
            or any(manifest[k] is not False for k in ('physics','native_admission','hardware_allowed'))
            or set(p.name for p in folder.iterdir()) != {'CHECKPOINT.json','parameters'}
            or (folder/'parameters').is_symlink()):
        raise ValueError('Private auxiliary inference package changed')
    names = set()
    filenames = {f'{n:06d}' for n in range(140)}
    for row in rows:
        file = Path(row['file'])
        if (len(file.parts) != 2 or file.parts[0] != 'parameters' or file.suffix != '.npy'
                or file.stem not in filenames
                or not row['key'].startswith(("['parameters']['shared']", "['parameters']['target']"))
                or row['dtype'] != 'float32' or (folder/file).is_symlink()
                or digest(folder/file) != row['sha256']):
            raise ValueError('Unexpected private Go2 parameter')
        names.add(file.name)
    if (len(names) != 29 or len({row['key'] for row in rows}) != 29
            or {p.name for p in (folder/'parameters').iterdir()} != names):
        raise ValueError('Unexpected number of private Go2 parameters')
    return manifest


def export(root, config, destination, scheduler):
    selection = pinned(config['selection_plan'])
    design = selection['possible_auxiliary_design']
    if (config['schema'] != SCHEMA or selection['adopted'] != TRAIN052
            or config['original_main_segment']['previous']['sha256'] != MAIN3800
            or [r['update'] for r in design['selections']] != list(PHASES)
            or set(scheduler) != set(ARMS)):
        raise ValueError('Expected the original matched intervention and its two PBS jobs')
    parent = pinned(dict(path=str(Path(config['custody'])/'parent.verified'/'COMPLETE.json'), sha256=ADAPT2K))['binding']
    original = pinned(dict(path=str(Path(config['original_main_segment']['previous']['path'])/'COMPLETE.json'), sha256=MAIN3800))
    a,b = (complete(root,arm,config,design,scheduler[arm]) for arm in ARMS)
    if a[2]!=b[2] or a[4]!=b[4] or a[4]!=original['binding']:
        raise ValueError('Arms differ outside the two allowed auxiliary slots')
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    with closing(OwnerLock(destination/'export.lock')):
        summaries = {}
        for (checkpoint,rows,_,result,bound) in (a,b):
            arm = result['arm']
            package = destination/arm
            manifest = dict(schema=EXPORT, arm=arm, phase=PHASES[-1], panel=PANEL,
                checkpoint_sha256=result['training_state_sha256'], comparison_parent=MAIN3800,
                frozen_model=bound['frozen_model'], foundation_sha256=parent['parent_manifest_sha256'],
                partition_sha256=parent['partition_sha256'], state_sha256=parent['target_state_sha256'],
                tokenizer_sha256=parent['tokenizer_sha256'], action_sha256=parent['frozen_C20_action_statistics_sha256'],
                training_result_sha256=result['result_sha256'], arrays=rows,
                physics=False, native_admission=False, hardware_allowed=False)
            if not package.exists():
                stage = Path(tempfile.mkdtemp(dir=destination, prefix=f'.{arm}-export-'))
                (stage/'parameters').mkdir()
                for row in rows:
                    os.link(checkpoint/Path(row['file']).name,stage/row['file'])
                write_once(stage/'CHECKPOINT.json',manifest)
                verify_package(stage,digest(stage/'CHECKPOINT.json'))
                sync_dir(stage/'parameters')
                sync_dir(stage)
                stage.rename(package)
                sync_dir(destination)
            manifest_sha = digest(package/'CHECKPOINT.json')
            if verify_package(package,manifest_sha) != manifest:
                raise ValueError(f'Existing private package conflicts with the sealed endpoint in {arm}')
            summaries[arm] = dict(result, manifest_sha256=manifest_sha, exported_arrays=29)
        receipt = dict(status='MATCHED_PAIR_RECONCILED', arms=summaries, matched_FM_updates=500,
                       comparison_parent=MAIN3800, new_generations=0, new_native_requests=0, new_updates=0)
        write_once(destination/'EXPORT.json', receipt)
        return receipt


if __name__ == '__main__':
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('comparison', type=Path)
    cli.add_argument('config_sha')
    cli.add_argument('scheduler', type=Path)
    cli.add_argument('scheduler_sha')
    cli.add_argument('destination', type=Path)
    args = cli.parse_args()
    os.umask(0o077)
    cfg = pinned(dict(path=str(args.comparison/'CONFIG.json'), sha256=args.config_sha))
    terminal = pinned(dict(path=str(args.scheduler), sha256=args.scheduler_sha))
    print(json.dumps(export(args.comparison,cfg,args.destination,terminal),sort_keys=True))
