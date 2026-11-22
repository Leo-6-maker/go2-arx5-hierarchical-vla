"""Read back the real paired Corelab event panel by split, task phase and window."""
from argparse import ArgumentParser
import json
from pathlib import Path

from scripts.sla.hopper_site import pinned
from scripts.sla.main3800_cross_layout_proof import matched_rows, pin
from scripts.sla.main3800_event_eval import endpoint_packages


def window(outcome):
    return {
        'full_H40': outcome['H40'], 'finger_gate': outcome['G7_speed'], 'joint_pass': outcome['legal'],
        'event_due': outcome['due'], 'timely_and_legal': outcome['timely_and_legal'],
        'expert_crossing': outcome['expert_crossing'], 'model_crossing': outcome['model_crossing'],
        'first_H40_exceedance': outcome['first_H40_exceedance'],
        'seam_mm_s': outcome['seam_mm_s'], 'finger_margin_mm_s': 40 - outcome['maximum_finger_mm_s'],
        'expert_up_P10_mm': outcome['expert_up_P10_mm'], 'model_up_P10_mm': outcome['model_up_P10_mm'],
        'aligned_P10_mm': outcome['aligned_P10_mm'],
    }


def aggregate(rows):
    due = [r for r in rows if r['due']]
    return dict(windows=len(rows), full_H40=sum(r['H40'] for r in rows),
                finger_gate=sum(r['G7_speed'] for r in rows), joint_pass=sum(r['legal'] for r in rows),
                due=len(due), timely_and_legal=sum(r['timely_and_legal'] for r in due))


def compare(before, after, population):
    rows = []
    for left, right, (split, diagnostic) in zip(before, after, population, strict=True):
        if (left['split'],left['tag'],left['due'],left['expert_crossing']) != (
                right['split'],right['tag'],right['due'],right['expert_crossing']):
            raise ValueError('Paired output changed the expert event or diagnostic split')
        rows.append(dict(ordinal=left['ordinal'], split=split, tag=diagnostic['tag'],
                         baseline=window(left), candidate=window(right)))
    splits = {}
    for split in ('TRAIN','VAL'):
        paired = [(a,b) for a,b in zip(before,after,strict=True) if a['split'] == split]
        a,b = ([p[i] for p in paired] for i in (0,1))
        changes = dict()
        for field in ('legal','timely_and_legal'):
            changes[field] = dict(improved=[r['ordinal'] for l,r in paired if l[field] is False and r[field] is True],
                                  regressed=[r['ordinal'] for l,r in paired if l[field] is True and r[field] is False])
        tags = sorted({r['tag'] for r in a})
        splits[split] = dict(baseline=aggregate(a), candidate=aggregate(b), transitions=changes,
            per_tag={tag:dict(baseline=aggregate([r for r in a if r['tag']==tag]),
                              candidate=aggregate([r for r in b if r['tag']==tag])) for tag in tags})
    return dict(splits=splits, paired_windows=rows)


def readback(config, terminal):
    cfg = pinned(config)
    panel = pinned(cfg['panel'])
    result = pinned(terminal)
    root = Path(config['path']).parent
    if (Path(terminal['path']) != root/'TERMINAL.json' or result['status'] != 'COMPLETE'
            or result['updates'] or result['visits'] or result['native_requests']
            or result['attempts'] != result['generations'] or result['generations'] != 40*len(result['rows'])
            or {k:len(v) for k,v in panel['roster'].items()} != {'TRAIN':24,'VAL':16}):
        raise ValueError('The original Core diagnostic must be complete before paired reporting')
    packages = endpoint_packages(pinned(cfg['native_config']), cfg.get('auxiliary_export'))
    if {arm for arm,_,_ in packages} != set(result['rows']):
        raise ValueError('An expected Core endpoint is missing or has changed')
    for arm,_,manifest in packages:
        matched_rows(root,result,panel,manifest['checkpoint_sha256'],arm)
    population = [(split,r) for split,rows in panel['roster'].items() for r in rows]
    arms = tuple(arm for arm,_,_ in packages)
    comparisons = (('MAIN-phase00000','MAIN-phase03800'),('MAIN-phase02000','MAIN-phase03800')) if len(arms)==3 else (arms,)
    return dict(terminal_sha256=terminal['sha256'], panel_sha256=cfg['panel']['sha256'],
                completed_generations=result['generations'], new_generations=0, new_native_requests=0, new_updates=0,
                comparisons={a+'__'+b:compare(result['rows'][a],result['rows'][b],population) for a,b in comparisons})


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    args = parser.parse_args()
    print(json.dumps(readback(pin(args.run/'CONFIG.json'),pin(args.run/'TERMINAL.json')),sort_keys=True,allow_nan=False))
