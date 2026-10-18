import unittest

from scripts.pi0294.pv1.decision import teacher_decision
from scripts.pi0294.pv1.resources import admit_resources, project_eligible


def fixture():
    rows, pairs, geometry = [], [], []
    for n in range(4):
        group = 'G' + str(n)
        pairs.append(dict(scene_group=group, direction='+Y' if n < 2 else '-Y',
            initial_native_and_scene_XML_equal=True, initial_two_camera_pixels_equal=True))
        for margin in [.004, .012]:
            identity = group + str(margin)
            rows.append(dict(id=identity, scene_group=group, nominal_clearance_m=margin,
                audit_status='PASS_INDEPENDENT_SAVED_BYTE_AUDIT', audit_checks=dict(raw=True),
                independent_full32_predicates_qualified=True, actual_native_failures=0,
                actual_robot_contact_ticks=100, minimum_vertical_gap_during_actual_robot_contact_m=[margin-.003]*2,
                original_score=dict(persistent_success=True, history=dict(no_lift=True))))
            geometry.append(dict(id=identity, vertical_gap_is_exact_separation_for_inside_above_boxes=True,
                all_pad_box_vertices_inside_finite_table_XY=True, maximum_vertical_gap_reconstruction_error_m=0))
    return (dict(status='PASS_ALL_EIGHT_SAVED_AUDITS_AND_FOUR_EXACT_INITIAL_PAIRS', rows=rows, pairs=pairs),
        dict(status='PASS_SAVED_GEOMETRY_RECONSTRUCTION_NO_NEW_PHYSICS', rows=geometry, integrations=0))


class PV1Admission(unittest.TestCase):
    def test_contact_gain_and_full_success_admit_pair(self):
        e, g = fixture()
        r = teacher_decision(e, g)
        self.assertEqual(r['route'], 'CLEARANCE_PAIR')
        self.assertEqual(r['contact_gain_qualified_pairs'], 4)
        self.assertFalse(r['neural_training_cause_proven'])

    def test_successful_original_does_not_automatically_forbid_pair(self):
        e, g = fixture()
        self.assertEqual(teacher_decision(e, g)['original_successes_out_of_4'], 4)

    def test_height_without_contact_is_not_qualified_improvement(self):
        e, g = fixture()
        for row in e['rows']:
            if row['nominal_clearance_m'] == .012:
                row['actual_robot_contact_ticks'] = 0
        self.assertEqual(teacher_decision(e, g)['route'], 'COVERAGE_FIRST')

    def test_finite_table_and_no_lift_required(self):
        for corruption in ['footprint', 'no_lift', 'task_fail']:
            e, g = fixture()
            if corruption == 'footprint':
                for row in g['rows']:
                    row['all_pad_box_vertices_inside_finite_table_XY'] = False
            elif corruption == 'no_lift':
                e['rows'][1]['original_score']['history']['no_lift'] = False
            else:
                e['rows'][1]['original_score']['persistent_success'] = False
            self.assertEqual(teacher_decision(e, g)['route'], 'COVERAGE_FIRST')

    def test_incomplete_or_native_fault_is_rejected(self):
        for field in ['pair', 'native', 'audit']:
            e, g = fixture()
            if field == 'pair': e['pairs'][0]['initial_two_camera_pixels_equal'] = False
            if field == 'native': e['rows'][0]['actual_native_failures'] = 1
            if field == 'audit': e['rows'][0]['audit_checks']['raw'] = False
            with self.assertRaises(PermissionError): teacher_decision(e, g)

    def test_multiple_projects_do_not_expand_concurrency_or_budget(self):
        with self.assertRaises(PermissionError):
            admit_resources({}, {}, {}, active_GPUs=1, requested_GPUs=1, job_wall_s=3600)
        with self.assertRaises(PermissionError):
            admit_resources(dict(H200_s=76*3600), {}, dict(H200_s=1))
        self.assertIn('PASS', admit_resources(dict(H200_s=76*3600), {}, dict(H200_s=72000), final=True)['status'])

    def test_pending_work_is_reserved_and_site_smaller_wins(self):
        with self.assertRaises(PermissionError):
            admit_resources({}, dict(episode_starts=900), dict(episode_starts=1))
        with self.assertRaises(PermissionError):
            admit_resources({}, {}, {}, requested_GPUs=1, job_wall_s=7200, site_wall_limit_s=3600)
        with self.assertRaises(PermissionError):
            admit_resources({}, {}, dict(physics_ticks=float('nan')))
        with self.assertRaises(PermissionError):
            admit_resources({}, {}, {}, requested_GPUs=1, job_wall_s=3600)
        self.assertIn('PASS',admit_resources({}, {}, dict(H200_s=3600,PBS_submissions=1),
            requested_GPUs=1,job_wall_s=3600)['status'])

    def test_active_flag_does_not_override_expiry(self):
        p=dict(active=True,member=True,start_date='2025-10-01',end_date='2026-09-30',balance_su=100)
        self.assertFalse(project_eligible(p,'2026-10-19'))
        p['end_date']='2027-03-12'
        self.assertTrue(project_eligible(p,'2026-10-19'))
        p['member']=False
        self.assertFalse(project_eligible(p,'2026-10-19'))
