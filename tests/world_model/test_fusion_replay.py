import unittest
import numpy as np
from go2arx5_vla.world_model.fusion_replay import response_window


class CausalFusionReplay(unittest.TestCase):
    def fixture(self):
        state=np.random.default_rng(366).normal(size=(25,106))
        return dict(episode=dict(split='TRAIN',purpose='TRAIN_FEEDBACK',inference_mode='CALIBRATION_NO_VLA',
                    desired_outcome_provenance='CALIBRATION_INTENT',parent='fixture'),
            decision=dict(native_step=3703,future_response_observed=False,split='TRAIN',parent='fixture',
                selected_action=1,actual_history=state[:9].tolist(),source_age_s=.002,
                past_reference=np.zeros(17).tolist(),nominal_action=1,desired_xy_m=[.001,0]),
            native=dict(status='NATIVE_STAGE_PASSED',shutdown_clean=True,physics_stops=[],actual_target_integrations=4103),
            score=dict(selected_action=1,split='TRAIN',parent='fixture',local_XY_success=False,actual_XY_error_m=.002),
            states=state,native_steps=3503+25*np.arange(25),windows=[dict(native_start_step=3703+25*i,
                native_end_step=3728+25*i,mode='WBC',reference_sha256='r',target_trajectory_sha256='t',
                solver_policies_with_exact_installed_target=[{}])for i in range(16)])

    def test_valid_negative_outcomes_stay_real_data_without_unmeasured_clock_or_expert_labels(self):
        value=response_window(**self.fixture());self.assertFalse(value['task_success'])
        self.assertFalse(value['expert_action_label']);self.assertEqual(value['physical_future_atoms'],16)
        self.assertEqual(value['source_age_valid'].sum(),1);self.assertTrue(value['source_age_valid'][8])
        np.testing.assert_array_equal(value['goal'][:3],[.001,0,0])

    def test_future_states_are_targets_and_do_not_change_inference_prefix_or_command(self):
        args=self.fixture();before=response_window(**args);args['states'][9:]+=1e6
        after=response_window(**args)
        for key in ['goal','history_goal','source_age']:
            np.testing.assert_array_equal(before[key],after[key])
        np.testing.assert_array_equal(before['state'][:9],after['state'][:9])
        self.assertFalse(np.array_equal(before['state'][9:],after['state'][9:]))

    def test_test_partial_changed_prefix_or_missing_consumption_is_rejected(self):
        for case in ['TEST','PARTIAL','PREFIX','CONSUMPTION']:
            with self.subTest(case=case):
                args=self.fixture()
                if case=='TEST':args['episode']['split']='TEST'
                elif case=='PARTIAL':args['native']['actual_target_integrations']=4102
                elif case=='PREFIX':args['states'][8,0]+=.001
                else:args['windows'][0]['solver_policies_with_exact_installed_target']=[]
                with self.assertRaises(ValueError):response_window(**args)


if __name__=='__main__':unittest.main()
