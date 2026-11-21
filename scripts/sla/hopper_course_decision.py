"""Freeze the latest eligible cumulative TRAIN snapshot at one sealed main-arm boundary."""
import argparse
from contextlib import closing
from pathlib import Path

from scripts.pi0292.durable import OwnerLock, read
from scripts.sla.hopper_checkpoint import BOUNDARIES, SCHEMA
from scripts.sla.hopper_inputs import D2_SPEC, PLAN, PARENT
from scripts.sla.hopper_parent_custody import digest, write_once
from scripts.sla.hopper_site import pinned
from scripts.sla.hopper_snapshot_relay import verified


def pin(path):
    return dict(path=str(path.resolve(strict=True)), sha256=digest(path))


def catalog(ready):
    """Only read small seals at the decision instant; large payload verification is on CPU."""
    visible = []
    for folder in sorted(ready.glob('TRAIN-[0-9][0-9][0-9]')):
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError('Unexpected READY entry')
        visible.append(dict(count=int(folder.name[6:]), seal_sha256=digest(folder/'SEALED.private.json')))
    return dict(root=str(ready.resolve()), visible=visible)


def decide(lineage, phase, event_pin, ready, population):
    if phase not in BOUNDARIES:
        raise ValueError('Unregistered data-switch boundary')
    event = pinned(event_pin)
    if (event['plan'] != PLAN or event['role'] != 'MAIN' or event['phase'] != phase
            or event_pin['path'] != str((lineage/'MAIN'/'sealed'/f'phase{phase:05d}.json').resolve())
            or event['binding']['role'] != 'MAIN' or event['binding']['parent'] != PARENT):
        raise ValueError('Expected the sealed main-arm boundary and original lineage')
    common = event['checkpoint']
    header = pinned(dict(path=str(Path(common['path'])/'COMPLETE.json'), sha256=common['sha256']))
    if (header['schema'] != SCHEMA or header['binding'] != event['binding']
            or header['progress'] != event['cursors'] or header['progress']['phase_step'] != phase
            or header['predecessor'] != event['predecessor']):
        raise ValueError('Common parent training state differs from the boundary event')
    root = lineage/'curriculum'
    root.mkdir(exist_ok=True)
    prior_course = event['course']
    target = event['binding']['target_manifest']
    with closing(OwnerLock(root/'owner.lock')):
        decision = root/f'phase{phase:05d}.json'
        if decision.exists():
            registered = read(decision)
            if registered['event'] != event_pin:
                raise ValueError('Changed checkpoint at a decided boundary')
            return registered
        pos = BOUNDARIES.index(phase)
        previous = read(root/f'phase{BOUNDARIES[pos-1]:05d}.json') if pos else None
        if (prior_course != (previous['next_course'] if previous else None)
                or target != (pinned(prior_course)['manifest_sha256'] if prior_course else D2_SPEC)):
            raise ValueError('The completed phase used a different target course')
        trigger = previous['first_control_trigger'] if previous else None
        membership = pinned(prior_course)['episodes'] if prior_course else []
        snapshot_list = event['train_ready_at_boundary']
        if snapshot_list['root'] != str(ready.resolve()):
            raise ValueError('Bounded TRAIN receipt directory changed')
        eligible = [item for item in snapshot_list['visible'] if item['count'] > len(membership)]
        course = prior_course
        if eligible:
            newest = max(eligible, key=lambda item: item['count'])
            chosen = ready/f"TRAIN-{newest['count']:03d}"
            seal_sha = newest['seal_sha256']
            seal, _, _ = verified(chosen, population, seal_sha)
            if seal['count'] != int(chosen.name[6:]) or seal['episode_sha256'][:len(membership)] != membership:
                raise ValueError('New READY membership conflicts with the previous cumulative course')
            file = root/f'course{phase:05d}.json'
            count = 3 + seal['count']
            write_once(file, dict(plan=PLAN, status='ADOPTED_CORE_TRAIN', adopted_at=phase, folder=str(chosen),
                       seal_sha256=seal_sha, manifest_sha256=seal['manifest_sha256'], episodes=seal['episode_sha256'],
                       bank_root=str(root/f'bank{phase:05d}'), nonlegacy=count,
                       new_slots=min(48, 4*count), old_slots=64-min(48, 4*count)))
            course = pin(file)
            if trigger is None:
                pair = root/'FIRST_CONTROL_TRIGGER.json'
                write_once(pair, dict(plan=PLAN, boundary=phase, common_parent=common, control_target=target,
                           control_course=prior_course, main_course=course, matched_endpoint=phase+2000))
                trigger = pin(pair)
        result = dict(plan=PLAN, phase=phase, event=event_pin, common_parent=common, previous_course=prior_course,
                      next_course=course, first_control_trigger=trigger, sealed_epoch_ns=event['sealed_epoch_ns'],
                      status='ADOPTED_CORE_TRAIN' if eligible else 'NO_NEW_DATA_CONTINUE')
        write_once(decision, result)
        return result


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('lineage', type=Path)
    cli.add_argument('phase', type=int, choices=BOUNDARIES)
    cli.add_argument('boundary_event_sha256')
    cli.add_argument('ready', type=Path)
    cli.add_argument('population', type=Path)
    args = cli.parse_args()
    boundary = args.lineage/'MAIN'/'sealed'/f'phase{args.phase:05d}.json'
    print(decide(args.lineage, args.phase, dict(path=str(boundary.resolve()), sha256=args.boundary_event_sha256),
                 args.ready, args.population)['status'])
