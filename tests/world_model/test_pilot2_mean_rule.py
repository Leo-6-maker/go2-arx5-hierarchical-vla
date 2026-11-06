import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
from go2arx5_vla.world_model.pilot2_mean_rule import fit,responses
from go2arx5_vla.world_model.storage import digest


def records(parents):
    return [dict(run=f'{p}-{a}-{r}',parent=p,split='TRAIN',action=a,repeat=r,
        response_xy_m=[i*.0001+a*.0002+r*.00001,-a*.0001])
        for i,p in enumerate(parents)for a in range(4)for r in range(2)]


class StrongMeanRule(unittest.TestCase):
    def test_parent_and_restart_means_use_actual_responses_not_goals(self):
        rr=records(['A','B']);m=fit(rr,['A','B'],'task')
        np.testing.assert_allclose(m['mean_response_xy_m'],[[.000055,0],[.000255,-.0001],[.000455,-.0002],[.000655,-.0003]],atol=1e-15)
        self.assertFalse(m['exact_fork_oracle']);self.assertFalse(m['rule_fit_changes_task_or_score'])
        self.assertEqual(m['episodes'],16)

    def test_partition_incomplete_duplicate_and_unqualified_action_rejected(self):
        rr=records(['A','B'])
        for bad in [rr[:-1],rr+[rr[0]],rr[1:]+[dict(rr[0],split='DEV')],rr[1:]+[dict(rr[0],action=4)]]:
            with self.subTest(case=bad[-1]),self.assertRaises(ValueError):fit(bad,['A','B'],'task')

    def test_new_baseline_leaves_original_task_and_historical_fallback_unchanged(self):
        task=dict(mean_response_xy_m=[[0.,0.]]*4,success_tolerance_m=.000185);before=copy.deepcopy(task)
        baseline,pin=responses({},task);np.testing.assert_array_equal(baseline,np.zeros((4,2)));self.assertIsNone(pin)
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'mean.json';p.write_text(json.dumps(fit(records(['A','B']),['A','B'],'task')))
            s=dict(train_mean_response_model=str(p),train_mean_response_model_sha256=digest(p),task_sha256='task')
            mean,pin=responses(s,task);self.assertEqual(pin,digest(p));self.assertNotEqual(mean[1,0],0)
            self.assertEqual(task,before)
            s['task_sha256']='other'
            with self.assertRaises(ValueError):responses(s,task)

    def test_cli_real_evidence_seam_rejects_failed_peer_and_seals_complete_panel(self):
        from scripts.wm02.fit_pilot2_mean_rule import run
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);parents=[f'P{i}'for i in range(12)];task=root/'task.json';task.write_text('{}')
            panel=root/'panel.json';panel.write_text(json.dumps(dict(schema='WM02_PILOT2_B0_FROZEN_ACTION_PANEL_V1',scenes=[dict(parent=p,split='TRAIN')for p in parents])))
            batch=root/'batch';(batch/'batch-01').mkdir(parents=True);bp=batch/'batch-01/RESULT.json';bp.write_text(json.dumps(dict(status='TWO_ENVIRONMENT_NATIVE_PILOT_PASSED')))
            entries=[]
            for r in records(parents):
                rr=root/r['run'];native=rr/'native-01';native.mkdir(parents=True)
                values=dict(spec=dict(parallel_batch_spec=str(batch/'SPEC.json'),task_sha256=digest(task)),
                    episode=dict(schema='WM02_PILOT2_COVERAGE_EPISODE_V1',purpose='ACTION_COVERAGE',split='TRAIN',parent=r['parent'],forced_action=r['action'],repeat_index=r['repeat']),
                    result=dict(status='NATIVE_STAGE_PASSED',shutdown_clean=True,physics_stops=[],actual_target_integrations=4103,native_binary_sha256='synthetic-not-robot'),
                    decision=dict(native_step=3703,future_response_observed=False,selected_action=r['action'],actual_decision_TCP_m=[.3,0,.7]),
                    score=dict(action=r['action'],actual_endpoint_TCP_m=[.3+r['response_xy_m'][0],r['response_xy_m'][1],.7]))
                paths=dict(spec=rr/'EXECUTION_SPEC.json',episode=rr/'EPISODE_MANIFEST.private.json',result=native/'RESULT.json',decision=native/'PILOT1_DECISION.private.json',score=native/'PILOT1_SCORE.private.json')
                for k,p in paths.items():p.write_text(json.dumps(values[k]))
                entries.append(dict(run=str(rr),**{k+'_sha256':digest(p)for k,p in paths.items()}))
            m=dict(schema='WM02_PILOT2_MEAN_RULE_INPUT_V1',TRAIN_parents=parents,task=str(task),task_sha256=digest(task),panel=str(panel),panel_sha256=digest(panel),native_binary_sha256='synthetic-not-robot',episodes=entries)
            bp.write_text(json.dumps(dict(status='TWO_ENVIRONMENT_NATIVE_PILOT_FAILED_PRESERVED')))
            with self.assertRaisesRegex(ValueError,'FAILED_PEER'):run(m,root/'bad')
            bp.write_text(json.dumps(dict(status='TWO_ENVIRONMENT_NATIVE_PILOT_PASSED')))
            result=run(m,root/'good');self.assertEqual(result['real_episodes'],96);self.assertEqual(result['optimizer_updates'],0)


if __name__=='__main__':unittest.main()
