"""Bind closed donor/control outputs and unchanged cards without model queries."""
import argparse
from pathlib import Path
from admission import BASE, ROOT, OLD, RAW, RELEASE, read, sha, write


def main():
    assert read(ROOT/'evidence/A0_PREDECESSOR_CLOSEOUT.json')['status'] == 'PASS_CONTINUE_PI0270_A1'
    points = ['ROT2048', 'PREFIX4096', 'AGG4096']
    source = {name: read(OLD/'design'/name) for name in
              ('FIXED_OFFLINE_CARD.json', 'EVALUATION_CARD.json', 'COMPONENT_CARD.json')}
    offline, virtual, component = source.values()
    assert [len(offline[k]) for k in ('fit_rows', 'factor_rows', 'legacy_rows')] == [224, 304, 440]
    assert len(component['rows']) == 32
    scene = read(BASE/'PI0267/design/SCENE_CARD.json')
    families = [f['family_id'] for f in scene['families'] if f['split'] == 'NEW_SIM_EVAL']
    assert len(families) == 8
    files = {}; records = []

    def bind(path, expected=None):
        path = Path(path)
        assert path.is_file() and not path.is_symlink(), path
        if str(path) not in files:
            files[str(path)] = dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))
        row = files[str(path)]
        if expected is not None: assert row['sha256'] == expected, path
        return row

    for name, card in source.items():
        bind(OLD/'design'/name)
        for noise in card['noise']: bind(noise['file'], noise['sha256'])
    for row in offline['fit_rows'] + offline['factor_rows']:
        bind(row['input'], row['input_sha256'])
    for point in points[:2]:
        prior = BASE/'PI0268/R1' if point == 'ROT2048' else OLD
        raw = RAW.parent/'PI0268/R1' if point == 'ROT2048' else RAW
        binding = bind(prior/'model_bindings'/(point+'.json'))
        offline_sha = bind(prior/'design/FIXED_OFFLINE_CARD.json')['sha256']
        virtual_sha = bind(prior/'design/EVALUATION_CARD.json')['sha256']
        for domain, rows in [('fit', offline['fit_rows']), ('factor', offline['factor_rows']), ('legacy', offline['legacy_rows'])]:
            for row in rows:
                path = raw/'offline'/point/domain/row['uid']; record = read(path/'record.json')
                assert record['point'] == point and record['row'] == row and record['card_sha256'] == offline_sha
                assert record['status'] in ('VALID', 'PREQUERY_EXCLUDED_KEEP_DENOMINATOR')
                if record['status'] != 'VALID': assert domain == 'factor' and not row['admitted']
                payload = bind(record['tensor'], record['tensor_sha256']) if record.get('tensor') else None
                records.append(dict(lane='offline', point=point, domain=domain, uid=row['uid'],
                                    record=bind(path/'record.json'), payload=payload,
                                    validity=record['status'], binding_sha256=binding['sha256']))
        for family in families:
            for task in ('M1', 'M3'):
                for command in (0, 1):
                    for backend in ('E0', 'E1'):
                        key = f'{family}-{task}-c{command}-{backend}'
                        path = raw/'virtual'/point/key; record = read(path/'result.json')
                        assert record['point'] == point and record['validity'] == 'VALID'
                        assert record['evaluation_card_sha256'] == virtual_sha
                        assert (record['family'], record['task'], record['command'], record['backend']) == (family, task, command, backend)
                        records.append(dict(lane='virtual', point=point, case=key,
                                            record=bind(path/'result.json'), payload=bind(path/'trajectory.npz', record['arrays_sha256']),
                                            validity=record['validity'], binding_sha256=binding['sha256']))
        # Both completed component sets live in PI0269; retain prequery exclusions.
        for row in component['rows']:
            path = RAW/'component'/point/row['case']; record = read(path/'result.json')
            assert record['point'] == point and record['component_start'] == row
            assert record['validity'] in ('VALID', 'PREQUERY_EXCLUDED')
            assert record['evaluation_card_sha256'] == bind(OLD/'design/EVALUATION_CARD.json')['sha256']
            payload = bind(path/'trajectory.npz', record['arrays_sha256']) if record['validity'] == 'VALID' else None
            records.append(dict(lane='component', point=point, case=row['case'], record=bind(path/'result.json'),
                                payload=payload, validity=record['validity'], binding_sha256=binding['sha256']))
    counts = {lane: sum(r['lane'] == lane for r in records) for lane in ('offline', 'virtual', 'component')}
    assert counts == dict(offline=1936, virtual=128, component=64) and len(records) == 2128
    for name, card in source.items():
        field = 'points' if name == 'COMPONENT_CARD.json' else 'model_points'
        card.update({field: points, 'schema': 'PI0270_REUSED_' + name.removesuffix('.json'),
                     'pi0270_release': RELEASE, 'pi0269_source_card_sha256': bind(OLD/'design'/name)['sha256']})
        if name == 'FIXED_OFFLINE_CARD.json':
            card.update(logical_rows=2904, donor_reused_rows=968, control_reused_rows=968,
                        new_AGG_rows=968, old_FIT_support='AGG consumption must be checked against actual training; not all old FIT is seen')
        elif name == 'EVALUATION_CARD.json':
            card.update(primary_episodes=192, donor_reused_episodes=64, control_reused_episodes=64, new_planned_episodes=64,
                        caps=dict(inference_attempts=30720, inference_heads=122880, new_virtual_starts=304,
                                  total_new_frame_pairs=28672, planned_main_queries=6400, planned_component_queries=640,
                                  planned_offline_queries=968, planned_newfit_queries=384, planned_collection_queries=19200))
        else:
            card.update(proposals=96, reused_proposals=64, new_AGG_proposals=32,
                        support='TRAIN-supported component diagnostic; PREFIX saved states may be AGG training support, not independent transfer')
        write(ROOT/'design'/name, card)
    for p in (BASE/'PI0267/design/SCENE_CARD.json', ROOT/'evidence/A0_PREDECESSOR_CLOSEOUT.json'):
        bind(p)
    result = dict(status='PASS_EXACT_CLOSED_1936_OFFLINE_128_MAIN_64_COMPONENT_REUSE', release=RELEASE,
                  counts=counts, records=records, files=list(files.values()), script_sha256=sha(__file__),
                  prior_closed_replay_reused=True, new_CPU_trajectory_replay=False,
                  model_queries=0, gradient_queries=0, optimizer_updates=0, renderer_calls=0,
                  new_evaluation_launch_allowed=False, AGG_training_and_quality_seal_still_required=True,
                  selected_checkpoint='NONE', deployment_selected_checkpoint='NONE')
    write(ROOT/'evidence/EVALUATION_EXACT_REUSE.json', result)
    print(result['status'], len(files), flush=True)


if __name__ == '__main__':
    main()
