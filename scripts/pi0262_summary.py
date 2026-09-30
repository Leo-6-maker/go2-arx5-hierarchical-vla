"""Descriptive family/wording/bias summaries from independently replayed rows."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def main(root):
    def rows(name):
        with (root / name).open(newline='') as stream:
            return list(csv.DictReader(stream))
    card = json.loads((root / 'CROSSED_EVAL_CARD.json').read_text())
    lookup = {r['row_id']: r for r in card['rows']}
    results = rows('CROSSED_RESULTS.csv')
    assert len(results) == 1920
    exposure, color_rates = defaultdict(list), defaultdict(list)
    for row in results:
        if row['split'] != 'TRAIN' or row['template'] not in ('T0', 'T1'):
            continue
        seen = lookup[row['row_id']]['original_template'] == row['template']
        exposure[row['point'], row['noise'], seen].append(row['correct'] == 'True')
    for row in results:
        color_rates[row['point'], row['split'], row['template'], row['noise'], row['color'], row['command']].append(row['correct'] == 'True')
    macro = defaultdict(list)
    for k, v in color_rates.items():
        macro[k[:4]].append(sum(v)/len(v))
    assert all(len(v) == 6 for v in macro.values())
    exposure_rows = [dict(point=k[0], noise=k[1], originally_exposed_combination=k[2],
        denominator=len(v), correct=sum(v)) for k, v in sorted(exposure.items())]
    decomposition = rows('TARGET_CHOICE_DECOMPOSITION.csv')
    categories = defaultdict(Counter)
    for row in decomposition:
        categories[row['point'], row['split'], row['template'], row['noise']][row['category']] += 1
    flips = rows('WORDING_FLIPS.csv')
    transitions = defaultdict(Counter)
    for row in flips:
        transitions[row['point'], row['split'], row['noise'], row['from_template'], row['to_template']][row['transition']] += 1
    crosses = rows('NOISE_BOUNDARY_CROSSINGS.csv')
    boundaries = defaultdict(list)
    for row in crosses:
        boundaries[row['point'], row['split'], row['template']].append(row)
    result = dict(status='PASS_DESCRIPTIVE_SUMMARY_NO_INDEPENDENT_ROW_SIGNIFICANCE',
        original_vs_recombined_T=exposure_rows,
        common_support_macro=[dict(point=k[0], split=k[1], template=k[2], noise=k[3],
            six_color_command_strata_macro_accuracy=sum(v)/len(v)) for k, v in sorted(macro.items())],
        choice_categories=[dict(point=k[0], split=k[1], template=k[2], noise=k[3], **v) for k, v in sorted(categories.items())],
        wording_transitions=[dict(point=k[0], split=k[1], noise=k[2], from_template=k[3], to_template=k[4], **v) for k, v in sorted(transitions.items())],
        noise_boundary=[dict(point=k[0], split=k[1], template=k[2], families=len(v),
            c0_crossings=sum(r['c0_boundary_crossing'] == 'True' for r in v),
            c1_crossings=sum(r['c1_boundary_crossing'] == 'True' for r in v),
            pair_crossings=sum(r['pair_correctness_crossing'] == 'True' for r in v)) for k, v in sorted(boundaries.items())],
        model_queries=0, optimizer_updates=0)
    path = root / 'DIAGNOSTIC_SUMMARY.json'
    rendered = json.dumps(result, sort_keys=True, indent=2) + '\n'
    if path.exists():
        assert path.read_text() == rendered
    else:
        path.write_text(rendered)
    print(json.dumps(dict(exposure=exposure_rows,
        P_choice=[v for v in result['choice_categories'] if v['point'] == 'P2000'],
        P_flips=[v for v in result['wording_transitions'] if v['point'] == 'P2000'],
        P_macro=[v for v in result['common_support_macro'] if v['point'] == 'P2000'],
        P_boundaries=[v for v in result['noise_boundary'] if v['point'] == 'P2000']), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
