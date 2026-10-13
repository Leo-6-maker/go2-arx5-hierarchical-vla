import copy
import unittest

from .smoke_retry_plan import GIB, PARALLEL_AUTHORITY, prepare


class TestSmokeRetryPlan(unittest.TestCase):
    def inputs(self):
        return dict(authorized_project='CFP04-CF-007',
            project=dict(name='CFP04-CF-007', active=True, users=['test-user'],
                cluster=['test-cluster'], start_date='2026-03-13', end_date='2027-03-12'),
            user='test-user', cluster='test-cluster', today='2026-10-13', balance_su=100,
            ledger=dict(model_loads=0, model_smoke_updates=0, main_updates=0,
                PBS_submissions=[dict(request_id='S3-smoke-01', status='SUBMIT_FAILED_PRESERVED',
                    job_id=None, GPU_reserved_s=0, CPU_reserved_s=0)],
                GPU_allocated_s=0, CPU_s=100, CPU_upper_s=60, stages=[]),
            own_jobs={}, quota=dict(user='test-user', source='NATIVE_HPC_PERSONAL_QUOTA_RAW_BYTES',
                used_bytes=940*GIB, hard_bytes=1024*GIB, report_epoch=1791875900),
            filesystem_free_bytes=1000*GIB, owned_bytes=16*GIB, now=1791876000,
            adoption=dict(release='261bc5b37f20aaff1ad7d5742252d3dc1a91fa65',
                T0_epoch=1791860304, new_work_cutoff_epoch=1791943104,
                deadline_epoch=1791946704, reset_on_resume=False))

    def test_capacity_hold_preserves_failure_and_all_budgets(self):
        values = self.inputs()
        original = copy.deepcopy(values)
        result = prepare(**values)
        self.assertEqual(values, original)
        self.assertEqual(result['blockers'], ['PERSONAL_QUOTA_FREE_MARGIN'])
        self.assertEqual(result['additional_headroom_needed_bytes'], 16*GIB)
        self.assertEqual(result['request_id'], 'S3-smoke-02')
        self.assertEqual(result['PBS_calls_remaining'], 2)
        self.assertFalse(result['authorizes_execution'])

    def test_capacity_pass_is_still_not_execution_authority(self):
        values = self.inputs()
        values['quota']['used_bytes'] = 850*GIB
        result = prepare(**values)
        self.assertEqual(result['blockers'], [])
        self.assertFalse(result['authorizes_execution'])
        self.assertTrue(result['recheck_at_submission_and_allocation'])

    def test_active_flag_alone_does_not_validate_project(self):
        for field, value in [('name', 'other'), ('users', []), ('cluster', []),
                             ('end_date', '2026-09-30'), ('start_date', '2027-01-01'), ('active', False)]:
            with self.subTest(field=field):
                values = self.inputs()
                values['project'][field] = value
                with self.assertRaises(PermissionError):
                    prepare(**values)

    def test_reset_or_duplicate_submission_is_refused(self):
        for change in ('clock', 'failure', 'loads', 'own_jobs', 'reservation', 'pending'):
            with self.subTest(change=change):
                values = self.inputs()
                if change == 'clock': values['adoption']['reset_on_resume'] = True
                if change == 'failure': values['ledger']['PBS_submissions'] = []
                if change == 'loads': values['ledger']['model_loads'] = 1
                if change == 'own_jobs': values['own_jobs'] = {'test-job': {}}
                if change == 'reservation': values['ledger']['PBS_submissions'][0]['GPU_reserved_s'] = 1800
                if change == 'pending': values['ledger']['pending_updates'] = 1
                with self.assertRaises(PermissionError):
                    prepare(**values)

    def test_stale_wrong_user_and_nonfinite_quota_refused(self):
        for field, value in [('report_epoch', 1791870000), ('report_epoch', 1791877000),
                             ('user', 'other'), ('hard_bytes', float('nan')),
                             ('used_bytes', -1), ('source', 'SHARED_DF')]:
            with self.subTest(field=field, value=value):
                values = self.inputs()
                values['quota'][field] = value
                with self.assertRaises((PermissionError, ValueError)):
                    prepare(**values)

    def test_explicit_parallel_revision_does_not_expand_pi0292_budgets(self):
        values=self.inputs();values['quota']['used_bytes']=850*GIB
        values['own_jobs']={'123.server':dict(Job_Owner='test-user@host',job_state='R',Job_Name='pi05_relative',project='CFP04-CF-009',Resource_List=dict(ngpus=1))}
        values.update(parallel_authority=PARALLEL_AUTHORITY,parallel_peer_ids=['123.server'])
        result=prepare(**values)
        self.assertEqual(result['PBS_calls_remaining'],2)
        self.assertEqual(result['next_smoke_walltime_s'],1800)
        self.assertFalse(result['authorizes_execution'])
        self.assertEqual(result['queue_only_after_any_dependencies'],[])
        values['parallel_peer_ids']=[]
        with self.assertRaises(PermissionError):prepare(**values)

    def test_explicit_dependency_queue_never_authorizes_overlap(self):
        values=self.inputs();values['quota']['used_bytes']=850*GIB
        values['own_jobs']={'123.server':dict(Job_Owner='test-user@host',job_state='R',project='other-project',Resource_List=dict(ngpus=1))}
        values['deferred_after_any']=['123.server']
        values['dependency_queue_authority']='2026-10-13_DIRECT_USER_ALLOW_PBS_DEPENDENCY_QUEUE_ONLY'
        result=prepare(**values)
        self.assertEqual(result['queue_only_after_any_dependencies'],['123.server'])
        self.assertFalse(result['authorizes_execution'])
        self.assertTrue(result['actual_allocation_still_requires_no_other_own_job'])
        for mutation in ('no-authority','wrong-id','foreign-owner','not-running','current-project','two-gpu'):
            with self.subTest(mutation=mutation):
                altered=copy.deepcopy(values)
                if mutation=='no-authority':altered['dependency_queue_authority']=None
                if mutation=='wrong-id':altered['deferred_after_any']=['999.server']
                if mutation=='foreign-owner':altered['own_jobs']['123.server']['Job_Owner']='other@host'
                if mutation=='not-running':altered['own_jobs']['123.server']['job_state']='Q'
                if mutation=='current-project':altered['own_jobs']['123.server']['project']=altered['authorized_project']
                if mutation=='two-gpu':altered['own_jobs']['123.server']['Resource_List']['ngpus']=2
                with self.assertRaises(PermissionError):prepare(**altered)


if __name__ == '__main__':
    unittest.main()
