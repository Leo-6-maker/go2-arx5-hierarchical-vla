"""Summarize all fixed saved results, including overlapping reliability failures."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

BASE = Path('/home/liuyu-nfs/Projects/go2-arx5')
ROOT = BASE / 'PI0269'
RAW = Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0269')
POINTS = ('ROT2048', 'UNIFORM4096', 'PREFIX4096')


def median(values):
    return statistics.median(values) if values else None


def reliability_summary(rows):
    receipts = [json.loads(r['retention']) for r in rows]
    delays = [r['first_pose_exit_after_success_s'] - r['original_completion_s']
              for r in receipts if r['original_success'] and r['first_pose_exit_after_success_s'] is not None]
    return dict(episodes=len(rows), original_success=sum(r['original_success'] for r in receipts),
                reliable=sum(r['reliable'] for r in receipts),
                counts={key:sum(bool(r[key]) for r in receipts) for key in
                        ('original_completed_by35', 'full40s', 'final5s_complete_sampling',
                         'final5s_every_tick_in_pose', 'full_episode_safe')},
                successful_then_pose_exit=len(delays), median_completion_to_first_exit_s=median(delays),
                final5s_position_bound_failed=sum(r['final5s_max_xyz_m'] is not None and r['final5s_max_xyz_m'] > .035 for r in receipts),
                final5s_rotation_bound_failed=sum(r['final5s_max_so3_rad'] is not None and r['final5s_max_so3_rad'] > .10 for r in receipts),
                failure_counts_overlap=True,
                median_final_achieved_target_xyz_m=median([float(r['actual_final_xyz_m']) for r in rows]),
                median_final_reference_target_xyz_m=median([float(r['installed_final_xyz_m']) for r in rows]))


def main():
    out = RAW / 'delivery'
    verdict = json.loads((out / 'PRIMARY_VERDICT.json').read_text())
    assert verdict['status'] == 'PRIMARY_EVALUATION_COMPLETE'
    tracking = json.loads((out / 'SAME_TICK_ANALYSIS_RECEIPT.json').read_text())
    assert tracking['main_episodes'] == 192 and tracking['component_proposals'] == 96
    with (out / 'VIRTUAL192.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    with (out / 'COMPONENT96.csv').open() as stream:
        components = list(csv.DictReader(stream))
    assert len(rows) == 192 and len(components) == 96
    groups = []
    for point in POINTS:
        own = [r for r in rows if r['point'] == point]
        assert len(own) == 64
        for task in ('M1', 'M3'):
            for backend in ('E0', 'E1'):
                group = [r for r in own if (r['task'], r['backend']) == (task, backend)]
                assert len(group) == 16
                groups.append(dict(point=point, task=task, backend=backend, **reliability_summary(group)))
    component_groups = []
    for point in POINTS:
        for start in ('M1_TERMINAL', 'M3_ENTRY', 'M3_MIDARC', 'M3_TERMINAL'):
            group = [r for r in components if (r['point'], r['start_type']) == (point, start)]
            assert len(group) == 8
            valid = [r for r in group if r['validity'] == 'VALID']
            assert all(r['validity'] in ('VALID', 'PREQUERY_EXCLUDED') for r in group)
            component_groups.append(dict(point=point, start_type=start, proposals=8, valid=len(valid), excluded=8-len(valid),
                full8s=sum(r['full8s'] == 'True' for r in valid),
                median_goal_pose_time_s=median([float(r['time_in_goal_pose_s']) for r in valid]),
                median_final_xyz_m=median([float(r['final_goal_xyz_m']) for r in valid]),
                median_final_so3_rad=median([float(r['final_goal_so3_rad']) for r in valid]),
                angular_bins_not_original_qualification=True, qualification_authority=False))
    paired = []
    families = sorted({r['family'] for r in rows})
    assert len(families) == 8
    for before, after in (('ROT2048', 'UNIFORM4096'), ('ROT2048', 'PREFIX4096'), ('UNIFORM4096', 'PREFIX4096')):
        for family in families:
            a = [r for r in rows if (r['point'], r['family']) == (before, family)]
            b = [r for r in rows if (r['point'], r['family']) == (after, family)]
            assert len(a) == len(b) == 8
            paired.append(dict(before=before, after=after, family=family, conditions=8,
                original_success_delta=sum(r['success']=='True' for r in b)-sum(r['success']=='True' for r in a),
                reliable_delta=sum(r['reliable']=='True' for r in b)-sum(r['reliable']=='True' for r in a)))
    sources = {name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in
               ('PRIMARY_VERDICT.json', 'VIRTUAL192.csv', 'COMPONENT96.csv', 'SAME_TICK_ANALYSIS_RECEIPT.json')}
    result = dict(status='PASS_COMPLETE_FIXED_SCOPE_DESCRIPTIVE_ANALYSIS', main_groups=groups,
        component_groups=component_groups, family_paired_changes=paired, source_sha256=sources,
        source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interpretation='Descriptive saved-result counts; overlapping failures are not exclusive causes. Eight developed families, no tick-level significance or independent-transfer claim.',
        model_queries=0, teacher_calls=0, render_calls=0, physics_steps=0,
        selected_checkpoint='NONE', deployment_selected_checkpoint='NONE')
    (out/'FINAL_DESCRIPTIVE_ANALYSIS.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(result['status'], len(groups), len(component_groups), len(paired), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-check', action='store_true')
    if parser.parse_args().self_check:
        r = dict(original_success=True, reliable=False, original_completed_by35=True, full40s=True,
                 final5s_complete_sampling=True, final5s_every_tick_in_pose=False, full_episode_safe=True,
                 original_completion_s=2., first_pose_exit_after_success_s=3.,
                 final5s_max_xyz_m=.04, final5s_max_so3_rad=.11)
        value = reliability_summary([dict(retention=json.dumps(r), actual_final_xyz_m='.04', installed_final_xyz_m='.039')])
        assert value['successful_then_pose_exit']==1 and value['median_completion_to_first_exit_s']==1.
        assert value['final5s_position_bound_failed']==value['final5s_rotation_bound_failed']==1
        assert value['reliable']==0 and value['counts']['full_episode_safe']==1
        print('PASS_CPU_SAVED_RESULT_OVERLAPPING_FAILURE_SUMMARY_ZERO_QUERY')
    else:
        main()
