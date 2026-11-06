import copy
import gzip
import json
import tempfile
import time
import unittest
import subprocess
import sys
from pathlib import Path
import numpy as np
from go2arx5_vla.world_model.pilot2_contact_model import replay, planned_openings


def episode(split='TRAIN', parent='P', run='run'):
    root = Path(__file__).parents[2]
    e = json.loads((root/'experiments/wm02/pilot2_supported_collection_v1.json').read_text())
    e.update(split=split, parent=parent, close_start_atom=64, command_program='CLOSE_RELEASE')
    plan = planned_openings(e); rows = []
    for i, command in enumerate(plan):
        start = 2503+25*i; grasp = 88 <= i < 125; contact = 83 <= i < 130
        finger = .018 if grasp else .042; pose = np.eye(4); pose[0, 3] = i*.000001
        ticks = [dict(native_step=s, native_grasp=grasp, table_support=True, contact_positive=contact,
            max_pad_cube_normal_force_N=.1 if contact else 0., finger_q=[finger, finger])
            for s in range(start+1, start+26)]
        rows.append(dict(native_start_step=start, native_end_step=start+25, qpos=[0.]*34,
            qvel=[0.]*32, ctrl=[0.]*20, TCP_m=[i*.000001, 0., .7], object_pose=pose.tolist(),
            finger_q=[finger, finger], contact_any=contact, native_contact_events=ticks,
            executed_reference_EEF10=[0., 0., .7, 1., 0., 0., 0., 1., 0., command],
            actual_gripper_internal_memory=dict(status='ACTUAL_OWNED_CHILD_GLOBALS_READ_ONLY',
                values=dict(steps=start+25, target=[finger, finger], opening=command,
                    expiry=(start+25)*.002+.35))))
    return dict(run_id=run, parent=parent, split=split, episode=e, samples=rows)


class SupportedDirectPrediction(unittest.TestCase):
    def test_prefix_training_restores_and_architectures_cannot_be_mixed(self):
        from scripts.wm02.train_pilot2_contact_wm import run
        from go2arx5_vla.world_model.storage import digest
        from go2arx5_vla.world_model.pilot2_contact_numpy import predict
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); bundle = root/'bundle.gz'
            eps = [episode(), episode('DEV', 'Q', 'other')]
            bundle.write_bytes(gzip.compress(json.dumps(dict(schema='WM02_PILOT2_SUPPORTED_ACTUAL_BUNDLE_V1',
                complete_MPC_WBC_hidden_state=False, episodes=eps)).encode(), mtime=0))
            s = dict(kind='CONTACT_WM', contact_recipe='WM02_PILOT2_PREFIX_CONTACT_V1',
                device='cpu', source_sha='SYNTHETIC_PREFIX_SOFTWARE_TEST_NOT_NATIVE', cpus=[0], seed=363101,
                hidden=8, learning_rate=.001, batch_size=8, endpoint=3, normalizer_fit='OWN_ARM_TRAIN',
                bundles=[dict(path=str(bundle), sha256=digest(bundle))],
                deadline_epoch=time.time()+60, registered_root=str(root))
            out = root/'fit'; out.mkdir(); result = run(s, out)
            self.assertEqual(result['architecture'], 'HORIZON_PREFIX_V1')
            self.assertTrue(result['prefix_invariant_by_construction'])
            self.assertEqual(result['main_WM_updates'], 6)
            self.assertEqual(result['restore_probe_optimizer_calls'], 2)
            data = replay(eps)
            for arm in result['arms'].values():
                self.assertEqual(arm['full_restore_error'], 0)
                self.assertEqual(arm['endpoint_reload_error'], 0)
                self.assertLess(arm['numpy_torch_max_error'], 1e-4)
            with np.load(out/'ACTION_CONDITIONED/DIRECT_CONTACT_NUMPY.npz') as z:
                arrays = {k: z[k].copy() for k in z.files}
            x = np.c_[data['history'][:4], data['plan'][:4]]; changed = x.copy(); changed[:, -63:] += 1
            np.testing.assert_array_equal(predict(arrays, x)[:, 0], predict(arrays, changed)[:, 0])
            s['resume_arms'] = {arm: dict(path=str(out/arm/'checkpoint-3.pt'), sha256=value['checkpoint_sha256'])
                for arm, value in result['arms'].items()}
            continued = root/'continued'; continued.mkdir(); r2 = run(s, continued)
            for arm in r2['arms'].values():
                self.assertEqual(arm['prior_WM_updates'], 3)
                self.assertEqual(arm['full_restore_error'], 0)
            s['contact_recipe'] = 'WM02_PILOT2_DIRECT_CONTACT_V1'
            wrong = root/'wrong'; wrong.mkdir()
            with self.assertRaisesRegex(ValueError, 'ARCHITECTURE_CHANGE_REQUIRES_FRESH_MODEL'):
                run(s, wrong)

    def test_native_dataset_module_has_no_torch_dependency(self):
        code = '''
import builtins
original = builtins.__import__
def native_import(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise ImportError('TORCH_FORBIDDEN_IN_NATIVE_REPLAY_ENVIRONMENT')
    return original(name, *args, **kwargs)
builtins.__import__ = native_import
from go2arx5_vla.world_model.pilot2_contact_model import replay, planned_openings
from scripts.wm02.build_pilot2_contact import build
from go2arx5_vla.world_model.pilot2_contact_numpy import predict
from go2arx5_vla.world_model.pilot2_contact_observer import record
from go2arx5_vla.world_model.pilot2_static_contact_parallel import validate_single
'''
        subprocess.run([sys.executable, '-B', '-c', code], check=True, capture_output=True, text=True)

    def test_future_observation_cannot_enter_input_and_no_future_program_is_matched(self):
        e = episode(); a = replay([e]); changed = copy.deepcopy(e)
        changed['samples'][9]['TCP_m'][0] += .01
        changed['samples'][9]['actual_gripper_internal_memory']['values']['expiry'] += 99
        b = replay([changed])
        np.testing.assert_array_equal(a['history'][0], b['history'][0])
        np.testing.assert_array_equal(a['plan'][0], b['plan'][0])
        self.assertFalse(np.array_equal(a['delta_mm'][0], b['delta_mm'][0]))
        self.assertTrue(np.any(a['plan'] != a['blind_plan']))

    def test_unexecuted_program_suffix_masked_and_real_failure_retained(self):
        e = episode(); a = replay([e]); e['samples'][65]['executed_reference_EEF10'][-1] = .95
        e['first_blocker'] = 'NATIVE_SAFE_EARLY_RELEASE'; b = replay([e])
        self.assertGreater(a['valid'].sum(), b['valid'].sum())
        np.testing.assert_array_equal(a['history'][0], b['history'][0])
        self.assertTrue(b['valid'][:30, 0].all())

    def test_adaptive_program_not_given_to_wm_before_actual_policy_decision(self):
        e=episode();e['program_known_from_anchor']=111;r=replay([e])
        self.assertEqual(len(r['history']),56)
        self.assertEqual(r['identities'][0]['anchor_native_step'],5303)

    def test_parent_test_clock_and_actual_contact_contracts(self):
        e = episode()
        with self.assertRaisesRegex(ValueError, 'WHOLE_PARENT'):
            replay([e, episode('DEV', 'P', 'second')])
        with self.assertRaisesRegex(ValueError, 'WHOLE_PARENT'):
            replay([episode('TEST')])
        with self.assertRaisesRegex(ValueError, 'DUPLICATE'):
            replay([e, copy.deepcopy(e)])
        e['samples'][0]['native_contact_events'][0]['contact_positive'] = True
        with self.assertRaisesRegex(ValueError, 'CONTACT_FORCE'): replay([e])

    def test_actual_training_seam_restores_full_sampler_model_optimizer_and_rng(self):
        from scripts.wm02.train_pilot2_contact_wm import run
        from go2arx5_vla.world_model.storage import digest
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); bundle = root/'bundle.gz'
            value = dict(schema='WM02_PILOT2_SUPPORTED_ACTUAL_BUNDLE_V1',
                complete_MPC_WBC_hidden_state=False,
                episodes=[episode(), episode('DEV', 'Q', 'other')])
            bundle.write_bytes(gzip.compress(json.dumps(value).encode(), mtime=0)); out = root/'fit'; out.mkdir()
            s = dict(kind='CONTACT_WM', contact_recipe='WM02_PILOT2_DIRECT_CONTACT_V1',
                device='cpu', source_sha='SYNTHETIC_SOFTWARE_TEST_NOT_NATIVE', cpus=[0], seed=363101,
                hidden=8, learning_rate=.001, batch_size=8, endpoint=4,
                bundles=[dict(path=str(bundle), sha256=digest(bundle))],
                deadline_epoch=time.time()+60, registered_root=str(root))
            r = run(s, out)
            self.assertEqual(r['main_WM_updates'], 8); self.assertEqual(r['restore_probe_optimizer_calls'], 2)
            self.assertEqual(r['TRAIN_parents'], ['P']); self.assertEqual(r['DEV_parents'], ['Q'])
            self.assertFalse(r['TEST_read']); self.assertFalse(r['free_rollout_qualified'])
            for arm in r['arms'].values():
                self.assertEqual(arm['full_restore_error'], 0); self.assertEqual(arm['endpoint_reload_error'], 0)
            inspection=root/'inspection.json';manifest=root/'inspect-manifest.json'
            manifest.write_text(json.dumps(dict(schema='WM02_PILOT2_FROZEN_CONTACT_INSPECTION_V1',
                source_sha=s['source_sha'],bundles=s['bundles'],models=[dict(name=arm,arm=arm,
                    path=str(out/arm/'DIRECT_CONTACT_NUMPY.npz'),sha256=value['portable_sha256'])
                    for arm,value in r['arms'].items()])))
            code='''
import builtins,sys
original=builtins.__import__
def native_import(name,*args,**kwargs):
    if name=='torch'or name.startswith('torch.'):raise ImportError('TORCH_FORBIDDEN_IN_INSPECTION')
    return original(name,*args,**kwargs)
builtins.__import__=native_import
from scripts.wm02.inspect_pilot2_contact_predictions import main
raise SystemExit(main(sys.argv[1:]))
'''
            subprocess.run([sys.executable,'-B','-c',code,'--manifest',str(manifest),'--output',str(inspection)],
                check=True,capture_output=True,text=True)
            inspected=json.loads(inspection.read_text());self.assertEqual(inspected['model_updates'],0)
            timing=inspected['baselines']['PERSISTENCE']['DEV']['grid_transition_timing']['events']['NATIVE_GRASP_END']
            self.assertGreater(timing['missed_actual_transitions'],0)
            self.assertIsNone(timing['detected_only_timing_MAE_s'])
            s['resume_arms'] = {arm: dict(path=str(out/arm/'checkpoint-4.pt'), sha256=r['arms'][arm]['checkpoint_sha256']) for arm in r['arms']}
            out2 = root/'continued'; out2.mkdir(); r2 = run(s, out2)
            for arm in r2['arms'].values(): self.assertEqual(arm['prior_WM_updates'], 4)
            s.pop('resume_arms'); s['normalizer_fit'] = 'OWN_ARM_TRAIN'
            out3 = root/'own-arm'; out3.mkdir(); r3 = run(s, out3)
            for arm in r3['arms'].values():
                self.assertEqual(arm['normalization_mode'], 'OWN_ARM_TRAIN')
                self.assertLess(arm['numpy_torch_max_error'], 1e-4)
            with np.load(out3/'ACTION_CONDITIONED/DIRECT_CONTACT_NUMPY.npz') as a, np.load(out3/'NO_FUTURE_PROGRAM/DIRECT_CONTACT_NUMPY.npz') as b:
                np.testing.assert_array_equal(a['mean'][:-64], b['mean'][:-64])
                self.assertFalse(np.array_equal(a['mean'][-64:], b['mean'][-64:]))


if __name__=='__main__': unittest.main()
