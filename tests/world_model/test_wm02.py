import copy,json,subprocess,sys,unittest
from types import SimpleNamespace
import numpy as np
from go2arx5_vla.world_model.action_compiler import compile_reference,SOURCE,TARGET,decision_due,discounted_block
from go2arx5_vla.world_model.contracts import FRAME,TCP
from go2arx5_vla.world_model.state_schema import actor_projection,ACTOR_FIELDS,future_inputs,shift_history,analytic_step
from go2arx5_vla.world_model.transitions import TransitionRecorder,validate_record
from go2arx5_vla.world_model.resolution import resolve


class WM02Contracts(unittest.TestCase):
    def test_target_time_map_reuses_real_receiver_abi_without_base(self):
        raw=np.zeros((8,7));raw[:,0]=.001;raw[:,6]=.5
        plan=compile_reference(raw,np.eye(4),10.,backend=TARGET,frame='world',tcp='x5_ee')
        wire=plan['wire_document'];self.assertNotIn('base',wire)
        points=wire['ee']['waypoints'];self.assertEqual(len(points),41)
        self.assertAlmostEqual(points[0]['position'][0],.001)
        self.assertAlmostEqual(points[7]['position'][0],.008)
        for p in points[8:]:self.assertEqual(p['position'],points[7]['position'])
        np.testing.assert_allclose([x['time']for x in points],np.arange(41)*.05)
        self.assertEqual(plan['ACTION_TIME_MAP']['raw_to_control_interval'],list(range(8)))
        self.assertFalse(plan['ACTION_TIME_MAP']['target_dynamic_qualified'])

    def test_unknown_backend_or_frame_and_nonprimitive_actions_are_rejected(self):
        raw=np.zeros((8,7));anchor=np.eye(4)
        for backend,frame,tcp in [('ISAAC',FRAME,TCP),(TARGET,FRAME,TCP),(SOURCE,'world','x5_ee')]:
            with self.assertRaises(ValueError):compile_reference(raw,anchor,0,backend=backend,frame=frame,tcp=tcp)
        raw[:,0]=.011
        with self.assertRaises(ValueError):compile_reference(raw,anchor,0,backend=SOURCE,frame=FRAME,tcp=TCP)

    def test_actor_two_decisions_in_16_atomic_steps_and_partial_discount(self):
        self.assertEqual([i for i in range(16)if decision_due(i)],[0,8])
        ret,discount=discounted_block([1.,2.,3.])
        self.assertAlmostEqual(ret,1+.997*2+.997**2*3);self.assertAlmostEqual(discount,.997**3)
        with self.assertRaises(ValueError):discounted_block([])

    def test_future_controller_or_context_cannot_leak(self):
        result=future_inputs({'tcp':np.eye(4)},{'time_s':.05},{'goal':[1,2,3]})
        self.assertEqual(result['time_s'],.05)
        for name in('wbc_hidden','wbc_observation_stack','mpc_policy_cache','solver_warmstart'):
            with self.assertRaises(ValueError):future_inputs({name:[99]}, {}, {})
        with self.assertRaises(ValueError):future_inputs({}, {}, {'native_progress':.5})
        h=list(range(8));self.assertEqual(shift_history(h,'predicted','command')[-1],('predicted','command'))
        self.assertEqual(analytic_step({'time_s':0},'same_reference',0)['reference_elapsed_s'],.05)

    def test_actor_whitelist_drops_teacher_robot_and_reset(self):
        state={k:0 for k in ACTOR_FIELDS};state.update(teacher_phase='carry',joint_position=[1],wbc_hidden=[2],reset_category='R3',native_success=True)
        self.assertEqual(set(actor_projection(state)),set(ACTOR_FIELDS))

    def row(self):
        return dict(run_id='r',episode_id='e',parent_id='p',scene_id='s',split='TRAIN',executor_id='S',actor_version='v',
            t0=0.,t1=.05,physics_ticks=25,action_raw=np.zeros(7),reference_compiled=np.zeros((40,10)),
            reference_installed=np.zeros(10),reference_consumed=[dict(time_s=k*.002,eef10=np.zeros(10))for k in range(25)],
            state_before=dict(time_s=0.),state_after=dict(time_s=.05),control_restore_ref='actual-complete-state-sha',
            terminal=False,truncated=False,safety_reason=None,valid_state_target=True)

    def test_zero_physics_rejection_is_only_event(self):
        recorder=TransitionRecorder(native_dt=.002);row=self.row();row.update(physics_ticks=0,t1=0,terminal=True);row['state_after']['time_s']=0
        with self.assertRaises(ValueError):recorder.append(row)
        recorder.reject(episode_id='e',reason='S1',state={},action=[])
        self.assertEqual(len(recorder.rows),0);self.assertFalse(recorder.events[0]['executed_expert_label'])

    def test_terminal_reset_and_parent_split_boundaries(self):
        row=self.row();row['terminal']=True;r=TransitionRecorder(native_dt=.002);r.append(row)
        second=self.row();second.update(t0=.05,t1=.1);second['state_before']['time_s']=.05;second['state_after']['time_s']=.1
        with self.assertRaises(ValueError):r.append(second)
        second=self.row();second['state_after']['reset']=True
        with self.assertRaises(ValueError):validate_record(second,native_dt=.002)
        second=self.row();second.update(episode_id='other',split='CALIB')
        with self.assertRaises(ValueError):r.append(second)

    def test_resolver_cannot_convert_closed_metadata_to_target_grant(self):
        b=dict(experiment_id='WM02',handoff_sha='ddc801f30e288365925c61c37f7ee445ddc29712',source_sha='a'*40,
            target=dict(backend=TARGET,permission=dict(status='CLOSED',native_safety_quarantine_preserved=True),
                        frozen_members=[dict(match=True)],full_control_restore_available=False),source_diagnostic={})
        result=resolve(b);self.assertEqual(result['target_status'],'TARGET_BLOCKED')
        self.assertFalse(result['target_dynamic_authorized']);self.assertIn('TARGET_S1_T2_QUARANTINE_NOT_RELEASED',result['target_blockers'])

    def test_cli_help_does_not_import_physics_models_or_network(self):
        for module in('resolve','qualify'):
            code="import runpy,sys;sys.argv=['cli','--help'];\ntry:runpy.run_module('scripts.wm02."+module+"',run_name='__main__')\nexcept SystemExit as e:\n if e.code:raise\nfor forbidden in ('torch','mujoco','isaacgym','rclpy','huggingface_hub'):\n if forbidden in sys.modules:raise RuntimeError(forbidden)"
            p=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)


if __name__=='__main__':unittest.main()
