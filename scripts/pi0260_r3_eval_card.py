"""Freeze PI0260-R3's authorized 1760 offline quality tuples without model calls."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'reports/PI0260/R3'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sources = {name: ROOT / name for name in ('motion2_eval_cases.json', 'O64_eval_cases.json', 'noise_manifest.json', 'S3_RUN_CARD.json', 'S3_ENGINEERING_AMENDMENT_2.json', 'S3_ENGINEERING_AMENDMENT_3.json', 'H1_R3_ADMITTED.json')}
    data = {name: json.loads(path.read_text()) for name, path in sources.items()}
    motion = data['motion2_eval_cases.json']['cases']
    original = data['O64_eval_cases.json']['rows']
    noises = data['noise_manifest.json']['rows']
    assert len(motion) == 156 and len(original) == 64
    assert [(x['name'], x['seed']) for x in noises] == [('B0', 26031), ('B1', 26032)]
    assert data['H1_R3_ADMITTED.json']['status'] == 'H1_R3_ADMITTED_WITH_MEASURED_TEMPORAL_AMBIGUITY'
    rows = [dict(arm=arm, step=step, cohort=cohort, case_index=i, noise=noise)
            for arm in ('Base', 'G') for step in (0, 6000)
            for cohort, cases in (('Motion2', motion), ('O64', original))
            for i in range(len(cases)) for noise in ('B0', 'B1')]
    assert len(rows) == len({tuple(r.values()) for r in rows}) == 1760
    card = dict(schema='PI0260_R3_FIXED_1760_QUALITY_CARD_V1', authority_commit='fd8e69b59a16aa90b1761d23b3a8f5a78cee8763',
                selected_checkpoint='NONE', deployment_selected_checkpoint='NONE',
                source_sha256={name: sha(path) for name, path in sources.items()},
                planned_logical_tuples=1760, query_attempt_cap=1824, inference_head_cap=7296,
                rows=rows)
    target = ROOT / 'S4_FIXED_EVAL_CARD.json'
    rendered = json.dumps(card, indent=2, sort_keys=True) + '\n'
    if target.exists():
        assert target.read_text() == rendered
    else:
        target.write_text(rendered)
    print(json.dumps(dict(status='R3_FIXED_1760_FROZEN', sha256=sha(target), rows=len(rows))))


if __name__ == '__main__':
    main()
