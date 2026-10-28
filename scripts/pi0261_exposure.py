"""Count original R3 training visits and actual TRAIN instruction exposure."""
import argparse
import json
from collections import Counter
from pathlib import Path


def main(dataset, visits_path, output):
    visits = json.loads(visits_path.read_text())['visits']
    assert len(visits) == 6000
    sequences = {}
    counts = Counter()
    template_counts = Counter()
    for visit in visits:
        name = visit['sequence']
        if name not in sequences:
            rec = json.loads((dataset / (name + '.json')).read_text())
            assert rec['sequence']['split'] == 'TRAIN'
            sequences[name] = rec['sequence']['instruction']
        counts[visit['task'], visit['command'], visit['family'], visit['quartile']] += 1
        template_counts[visit['task'], visit['command'], sequences[name]] += 1
    assert len(sequences) == 120
    result = dict(schema='PI0261_R3_ACTUAL_EXPOSURE_V1', status='PASS_R3_SOURCE_BOUND_EXPOSURE_COUNT',
                  visits=6000, sequences=120, counts=[dict(task=t, command=c, family=f, quartile=q, visits=n)
                                                for (t, c, f, q), n in sorted(counts.items())],
                  instruction_templates=[dict(task=t, command=c, text=text, visits=n)
                                         for (t, c, text), n in sorted(template_counts.items())],
                  model_queries=0, optimizer_updates=0)
    rendered = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if output.exists():
        assert output.read_text() == rendered
    else:
        output.write_text(rendered)
    print(json.dumps(dict(status=result['status'], visits=6000, templates=len(template_counts),
                          family_phase_groups=len(counts))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--old-visits', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.dataset, args.old_visits, args.output)
