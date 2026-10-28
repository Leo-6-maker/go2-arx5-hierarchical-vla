"""Freeze PI0261's 2040 fixed offline rows before any new quality query."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'reports'
R3 = ROOT / 'PI0260/R3'
OUT = ROOT / 'PI0261'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def freeze():
    plan = read(OUT / 'PAIRING_PLAN.json')
    pair_receipt = read(OUT / 'PROCESSED_PAIR_BANK_AUDIT.json')
    old = read(R3 / 'S4_FIXED_EVAL_CARD.json')
    assert pair_receipt['status'] == 'PASS_ALL_60_PROCESSED_PAIRS'
    assert len(plan['common_start_bank']) == 60 and len(old['rows']) == 1760
    old_rows = [row for row in old['rows'] if row['arm'] == 'Base' and row['step'] == 6000]
    assert len(old_rows) == 440
    rows = []
    for point in ('Donor', 'C2000', 'P2000'):
        rows.extend(dict(point=point, cohort=row['cohort'], case_index=row['case_index'],
                         noise=row['noise'], source='R3_ORIGINAL') for row in old_rows)
        for index, pair in enumerate(plan['common_start_bank']):
            for command in (0, 1):
                for noise in ('B0', 'B1'):
                    rows.append(dict(point=point, cohort='TRAIN_ENTRY', pair_index=index,
                                     task=pair['task'], family=pair['family'], command=command,
                                     noise=noise, source='PI0261_ORIGINAL_INDEX0'))
    assert len(rows) == 2040
    assert len({json.dumps(row, sort_keys=True) for row in rows}) == 2040
    card = dict(schema='PI0261_FIXED_2040_EVAL_CARD_V1',
                authority_commit='6f2353a718cc877dddb3124029de76007eaa7df3',
                donor_revision='52abc8d151b67bb667e43fcf874dfe2cfa253364',
                original_r3_card_sha256=sha(R3 / 'S4_FIXED_EVAL_CARD.json'),
                motion_cases_sha256=sha(R3 / 'motion2_eval_cases.json'),
                o64_cases_sha256=sha(R3 / 'O64_eval_cases.json'),
                noise_manifest_sha256=sha(R3 / 'noise_manifest.json'),
                pairing_plan_sha256=sha(OUT / 'PAIRING_PLAN.json'),
                processed_pair_audit_sha256=sha(OUT / 'PROCESSED_PAIR_BANK_AUDIT.json'),
                logical_rows=2040, planned_new_queries_cap=1600,
                new_attempt_cap=1728, inference_head_cap=6912,
                selected_checkpoint='NONE', deployment_selected_checkpoint='NONE', rows=rows)
    target = OUT / 'FIXED_EVAL_CARD.json'
    rendered = json.dumps(card, indent=2, sort_keys=True) + '\n'
    if target.exists():
        assert target.read_text() == rendered
    else:
        target.write_text(rendered)
    print(json.dumps(dict(status='PI0261_FIXED_EVAL_FROZEN', logical_rows=len(rows), sha256=sha(target))))


def diagnose():
    plan = read(OUT / 'PAIRING_PLAN.json')
    replay = read(R3 / 'Base_6000.semantic_replay.json')
    groups = defaultdict(list)
    for row in replay['rows']:
        groups[(row['task'], row['kind'], row['noise'])].append(row)
    table = []
    for (task, kind, noise), values in sorted(groups.items()):
        assert len(values) == 12
        commands = {command: [row for row in values if row['command'] == command] for command in (0, 1)}
        assert all(len(v) == 6 for v in commands.values())
        families = defaultdict(list)
        for row in values:
            families[row['family']].append(row)
        assert all(len(v) == 2 for v in families.values())
        table.append(dict(task=task, kind=kind, noise=noise,
                          correct=sum(row['correct'] for row in values),
                          command_correct={str(c): sum(row['correct'] for row in commands[c]) for c in (0, 1)},
                          ties=sum(row['margin_m'] == 0 for row in values),
                          negative=sum(row['margin_m'] < 0 for row in values),
                          fully_correct_pairs=sum(all(row['correct'] for row in pair) for pair in families.values()),
                          median_margin_m=sorted(row['margin_m'] for row in values)[5:7]))
    result = dict(schema='PI0261_S0_SAVED_EVIDENCE_DIAGNOSIS_V1',
                  status='S0_FINITE_AUDIT_COMPLETE_NO_NEW_MODEL_QUERY',
                  semantic_rule='original own/opposite H40 XYZ RMSE; strict margin > 0; tie wrong',
                  scope='KIN-VAL development validation t0 entry, not full contour or physics',
                  original_pairs=12, original_semantic_rows=96, groups=table,
                  old_exposure=plan['old_exposure_diagnosis'],
                  processed_bank_pairs=read(OUT / 'PROCESSED_PAIR_BANK_AUDIT.json')['pairs'],
                  original_condition_audit_sha256=sha(R3 / 'condition_audit.json'),
                  original_mixed_semantics=read(R3 / 'condition_audit.json')['partitions']['TRAIN']['mixed_semantics'],
                  model_queries=0, optimizer_updates=0,
                  selected_checkpoint='NONE', deployment_selected_checkpoint='NONE')
    target = OUT / 'S0_DIAGNOSIS.json'
    rendered = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if target.exists():
        assert target.read_text() == rendered
    else:
        target.write_text(rendered)
    print(json.dumps(dict(status=result['status'], groups=len(table), sha256=sha(target))))


if __name__ == '__main__':
    diagnose()
    freeze()
