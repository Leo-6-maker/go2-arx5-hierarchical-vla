import copy
import unittest
import gzip
import json
import tempfile
import time
from pathlib import Path
import numpy as np
from tests.world_model.test_pilot2_contact_model import episode
from go2arx5_vla.world_model.pilot2_supported import stage_for
from go2arx5_vla.world_model.pilot2_contact_model import planned_openings
from go2arx5_vla.world_model.pilot2_supported_decision import dataset, candidate_plans


def supported(split, parent, program):
    ep = episode(split, parent, parent+'-'+program); e = ep['episode']
    e.update(command_program=program, close_start_atom=56); plan = planned_openings(e)
    for i, row in enumerate(ep['samples']):
        stage = stage_for(i, e); row['stage'] = stage
        grasp = i >= 80 and (program=='CLOSE_KEEP' or i<125)
        row['executed_reference_EEF10'][-1] = float(plan[i]); row['contact_any'] = grasp
        row['finger_q'] = [.018, .018] if grasp else [.042, .042]
        for tick in row['native_contact_events']:
            tick.update(stage=stage, native_grasp=grasp, contact_positive=grasp,
                max_pad_cube_normal_force_N=.1 if grasp else 0., finger_q=row['finger_q'])
    return ep


class SupportedPolicyData(unittest.TestCase):
    def data(self):
        return [supported(s, p, a) for s, p in [('TRAIN', 'P'), ('DEV', 'Q')]
            for a in ['CLOSE_KEEP', 'CLOSE_RELEASE']]

    def test_goal_policy_labels_are_nonconstant_and_future_outcomes_cannot_enter_inputs(self):
        e = self.data(); a = dataset(e); changed = copy.deepcopy(e)
        changed[0]['samples'][130]['TCP_m'][0] += 99
        b = dataset(changed)
        np.testing.assert_array_equal(a['x'], b['x']); np.testing.assert_array_equal(a['labels'], b['labels'])
        self.assertEqual(set(a['labels'].argmax(-1)), {0, 1})
        self.assertFalse(a['exact_fork_oracle']); self.assertEqual(a['decision_atom'], 112)
        self.assertEqual(a['x'].shape[0], 8)

    def test_measured_returns_come_from_both_executed_programs_not_unexecuted_relabel(self):
        e = self.data(); a = dataset(e, teacher='MEASURED_PARENT_MEAN_RETURN')
        self.assertEqual(set(a['labels'].argmax(-1)), {0, 1})
        with self.assertRaisesRegex(ValueError, 'BOTH_REAL_QUALIFIED'):
            dataset([e[1], e[3]])
        e[0]['split'] = 'TEST'
        with self.assertRaisesRegex(ValueError, 'WHOLE_PARENT'): dataset(e)

    def test_candidate_plan_is_identical_prefix_and_only_qualified_keep_or_release(self):
        e = self.data()[0]['episode']; p = candidate_plans(e)
        self.assertEqual(p.shape, (2, 64)); self.assertTrue(np.all(p[0]<p[1]))
        with self.assertRaisesRegex(ValueError, 'CLOSE56'):
            candidate_plans(dict(e, close_start_atom=64))

    def test_supported_policy_actual_trainer_restores_and_learns_both_goal_actions(self):
        from scripts.wm02.train_pilot2_supported_policy import run
        from go2arx5_vla.world_model.storage import digest
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bundle=root/'bundle.gz';bundle.write_bytes(gzip.compress(json.dumps(dict(
                schema='WM02_PILOT2_SUPPORTED_ACTUAL_BUNDLE_V1',episodes=self.data())).encode(),mtime=0))
            out=root/'fit';out.mkdir();s=dict(source_sha='SYNTHETIC_SOFTWARE_TEST_NOT_NATIVE',device='cpu',
                kind='CONTACT_POLICY',policy_recipe='WM02_PILOT2_SUPPORTED_GOAL_POLICY_V1',
                bundles=[dict(path=str(bundle),sha256=digest(bundle))],teacher_method='GOAL_FSM',
                cpus=[0],seed=363201,learning_rate=.01,batch_size=8,endpoint=200,
                deadline_epoch=time.time()+60,registered_root=str(root))
            r=run(s,out);self.assertEqual(r['main_Actor_updates'],200)
            self.assertEqual(r['complete_restore_error'],0);self.assertEqual(r['endpoint_reload_error'],0)
            self.assertEqual(r['metrics']['DEV']['teacher_agreement'],1.)
            self.assertEqual(r['metrics']['DEV']['action_counts'],[2,2])
            self.assertFalse(r['native_policy_evaluated']);self.assertFalse(r['TEST_read'])
            s.update(resume=str(out/'checkpoint-200.pt'),resume_sha256=r['checkpoint_sha256'],endpoint=4)
            continued=root/'continued';continued.mkdir();r=run(s,continued)
            self.assertEqual(r['prior_Actor_updates'],200);self.assertEqual(r['complete_restore_error'],0)
            s.update(normalizer_reference=str(out/'checkpoint-200.pt'),normalizer_reference_sha256=digest(out/'checkpoint-200.pt'))
            with self.assertRaisesRegex(ValueError,'FRESH_NORMALIZER_CONTROL'):run(s,root/'invalid-control')
            s.pop('resume');s.pop('resume_sha256');fresh=root/'normalizer-control';fresh.mkdir();r=run(s,fresh)
            self.assertEqual(r['normalization_origin'],'PINNED_PRIOR_TRAIN_INPUTS_FRESH_MODEL_CONTROL')
            self.assertEqual(r['prior_Actor_updates'],0);self.assertEqual(r['complete_restore_error'],0)


if __name__=='__main__': unittest.main()
