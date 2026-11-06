import copy,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from go2arx5_vla.world_model.fusion_response import ResponseModel,prepare,validate_data
from go2arx5_vla.world_model.fusion_response_numpy import predict


class ResponseLearning(unittest.TestCase):
    def test_native_model_requires_frozen_primary_seed_full_restore_and_closed_training(self):
        from go2arx5_vla.world_model.fusion_response_inference import validate_asset
        from go2arx5_vla.world_model.storage import digest
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);(root/'control').mkdir();fit=root/'fit-01';fit.mkdir()
            model=fit/'366301-ACTION_CONDITIONED/RESPONSE_NUMPY.npz';model.parent.mkdir();model.write_bytes(b'fixture weights')
            rows=[dict(seed=s,arm=a,numpy_sha256=digest(model),full_restore_error=0,endpoint_reload_error=0,numpy_max_error_m=0,checkpoint_sha256='fixture')for s in (366301,366302,366303)for a in ('ACTION_CONDITIONED','NO_FUTURE_ACTION')]
            value=dict(schema='WM02_FUSION_MATCHED_RESPONSE_STUDY_RESULT_V1',status='TRAINED_EXACTLY_RESTORED_AND_NUMPY_EXPORTED',
                main_WM_updates=12000,restore_probe_optimizer_calls=6,primary_seed=366301,TEST_read=False,ARX_sha256='arx',
                replay_sha256='replay',TRAIN_mean_response_rule_sha256='static',results=rows)
            p=fit/'RESULT.private.json';p.write_text(json.dumps(value));lp=root/'control/LEASE.json'
            lp.write_text(json.dumps(dict(status='CLOSED',worker_exit_code=0)))
            spec=dict(response_study_result=str(p),response_study_result_sha256=digest(p),response_arm='ACTION_CONDITIONED',
                response_numpy=str(model),response_numpy_sha256=digest(model),ARX_sha256='arx',train_mean_response_model_sha256='static')
            self.assertEqual(validate_asset(spec)['training_replay_sha256'],'replay')
            for change in [dict(primary_seed=366302),dict(main_WM_updates=0),dict(TEST_read=True),dict(TRAIN_mean_response_rule_sha256='other')]:
                p.write_text(json.dumps(dict(value,**change)));spec['response_study_result_sha256']=digest(p)
                with self.assertRaises(PermissionError):validate_asset(spec)
            p.write_text(json.dumps(value));spec['response_study_result_sha256']=digest(p)
            lp.write_text(json.dumps(dict(status='ACTIVE',worker_exit_code=0)))
            with self.assertRaises(PermissionError):validate_asset(spec)

    def test_native_numpy_candidate_bridge_blinds_both_paths_without_changing_joint_predictions(self):
        from tests.world_model.test_pilot1_actor import fixture
        from go2arx5_vla.world_model.fusion_response_numpy import candidate_forecasts
        from go2arx5_vla.world_model.pilot1_numpy import actor_features
        meta,teacher,data=fixture();data={k:v[:1].copy()for k,v in data.items()}
        forecasts=np.stack([teacher.predict(data,np.array([0]))[0]+i*.00001 for i in range(4)])
        dim=actor_features(data,meta,np.array([0]),np.zeros((1,2))).shape[-1]
        model=ResponseModel(dim);arrays=model.arrays();sl=teacher.layout['tcp_position_m']
        ac=candidate_forecasts(arrays,data,meta,forecasts,teacher.layout,'ACTION_CONDITIONED')
        np.testing.assert_allclose(ac[:,:,sl],forecasts[:,:,sl],atol=1e-7,rtol=0)
        arrays['b1'][:]=.2
        blind=candidate_forecasts(arrays,data,meta,forecasts,teacher.layout,'NO_FUTURE_ACTION')
        for a in range(4):
            np.testing.assert_array_equal(blind[a,:,sl],blind[0,:,sl])
            np.testing.assert_array_equal(blind[a,:,:86],forecasts[a,:,:86])
        changed=copy.deepcopy(data);changed['state'][:,9:]=1e10
        np.testing.assert_array_equal(blind,candidate_forecasts(arrays,changed,meta,forecasts,teacher.layout,'NO_FUTURE_ACTION'))

    def test_future_state_and_removed_candidate_cannot_enter_blind_features_or_base(self):
        from tests.world_model.test_pilot1_actor import fixture
        meta,teacher,data=fixture();data['history_goal'][:]=0;data['goal'][:]=0;data['goal'][:,0]=.001
        ac=prepare(teacher,data,meta,'ACTION_CONDITIONED');blind=prepare(teacher,data,meta,'NO_FUTURE_ACTION')
        changed=copy.deepcopy(data);changed['state'][:,9:]=1e10;changed['goal'][:,0]=-.001
        other=prepare(teacher,changed,meta,'NO_FUTURE_ACTION')
        for name in ('x','base','observed'):np.testing.assert_array_equal(blind[name],other[name])
        self.assertFalse(np.array_equal(ac['x'],blind['x']))
        self.assertFalse(np.array_equal(ac['base'],blind['base']))
        changed['goal'][:]=data['goal']
        for name in ('x','base','observed'):np.testing.assert_array_equal(ac[name],prepare(teacher,changed,meta,'ACTION_CONDITIONED')[name])

    def test_nonzero_model_update_numpy_export_and_complete_shuffle_optimizer_restore(self):
        from go2arx5_vla.world_model.actor_sampling import ExactSampler
        from go2arx5_vla.world_model.neural_training import rng_state,restore_rng,tree_error
        torch.manual_seed(366301);x=torch.randn(7,12);base=torch.randn(7,16,3)*.001
        model=ResponseModel(12);model.fit_normalizer(x);opt=torch.optim.AdamW(model.parameters(),lr=.0003)
        sampler=ExactSampler(7,'SEEDED_GLOBAL_SHUFFLE',366301)
        def update(m,o,s):
            ids=torch.as_tensor(s.take(32));p=m(x[ids],base[ids]);loss=(((p-(base[ids]+.0002))/.001)**2).mean()
            o.zero_grad();loss.backward();o.step()
        torch.testing.assert_close(model(x,base),base,atol=0,rtol=0)
        update(model,opt,sampler)
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp)/'cp.pt';torch.save(dict(model=model.state_dict(),opt=opt.state_dict(),sampler=sampler.state_dict(),rng=rng_state()),path)
            update(model,opt,sampler);expected=rng_state()
            clone=ResponseModel(12);co=torch.optim.AdamW(clone.parameters(),lr=.0003);cs=ExactSampler(7,'SEEDED_GLOBAL_SHUFFLE',366301)
            saved=torch.load(path,weights_only=True);clone.load_state_dict(saved['model']);co.load_state_dict(saved['opt']);cs.load_state_dict(saved['sampler']);restore_rng(saved['rng']);update(clone,co,cs)
            self.assertEqual(tree_error(model.state_dict(),clone.state_dict()),0)
            self.assertEqual(tree_error(opt.state_dict(),co.state_dict()),0)
            self.assertEqual(tree_error(sampler.state_dict(),cs.state_dict()),0)
            self.assertEqual(tree_error(expected,rng_state()),0)
        np.testing.assert_allclose(predict(model.arrays(),x.numpy(),base.numpy()),model(x,base).detach().numpy(),atol=1e-9,rtol=0)
        self.assertGreater(float((model(x,base)-base).detach().abs().sum()),0)

    def test_incomplete_unbalanced_or_test_data_rejected_and_validity_mask_is_explicit(self):
        from go2arx5_vla.world_model.fusion_contract import OFFSETS
        actions=np.tile(np.repeat(np.arange(4),2),6);parents=np.repeat(np.arange(6).astype(str),8)
        data=dict(state=np.zeros((48,25,106)),goal=np.zeros((48,17)),history_goal=np.zeros((48,8,17)),
            source_age=np.zeros((48,25)),source_age_valid=np.zeros((48,25),bool),future_prefix_remaining=np.full(48,7.),
            window_start_s=np.zeros(48),parent=parents,split=np.where(np.arange(48)<32,'TRAIN','DEV'),observed_action=actions,
            repeat_index=np.tile([0,1],24),run_id=np.arange(48).astype(str))
        data['goal'][:,:3]=OFFSETS[actions];data['source_age_valid'][:,8]=True
        self.assertEqual(len(validate_data(data)['TRAIN']),32)
        bad=copy.deepcopy(data);bad['split'][0]='TEST'
        with self.assertRaises(ValueError):validate_data(bad)
        bad=copy.deepcopy(data);bad['goal'][0,0]=.1
        with self.assertRaises(ValueError):validate_data(bad)
        bad=copy.deepcopy(data);bad['source_age_valid'][0,9]=True
        with self.assertRaises(ValueError):validate_data(bad)
        bad=copy.deepcopy(data);bad['goal'][0,3]=.1
        with self.assertRaises(ValueError):validate_data(bad)
        bad=copy.deepcopy(data);bad['repeat_index']=bad['repeat_index'].astype(float)+.1
        with self.assertRaises(ValueError):validate_data(bad)

    def test_response_gpu_requires_real_balanced_sealed_feedback_and_headroom(self):
        from go2arx5_vla.world_model.fusion_scope import validate_gpu
        from go2arx5_vla.world_model.storage import digest
        gpu='GPU-fef928e1-b12e-2f20-4b2e-a868e6e0faad'
        with tempfile.TemporaryDirectory()as tmp:
            r=Path(tmp);(r/'control').mkdir();(r/'replay-01').mkdir();p=r/'replay-01/RESULT.private.json'
            value=dict(schema='WM02_FUSION_REAL_RESPONSE_REPLAY_V1',eligible_windows=48,real_post_choice_transitions=768,
                physical_parents={str(i):'TRAIN'if i<4 else'DEV'for i in range(6)},excluded=[],TEST_read=False,replay_sha256='actual-hash')
            p.write_text(json.dumps(value));(r/'control/LEASE.json').write_text(json.dumps(dict(status='CLOSED')))
            s=dict(execution_site='CORE_FUSION_NEURAL',profile='wm02-fusion-response-train',kind='FUSION_RESPONSE_WM',physics=False,
                device='cuda',GPU_uuid=gpu,allocator_limit_bytes=2*1024**3,GR00T_updates=0,numeric_mode='FP32_TF32_OFF',
                endpoint=2000,seeds=[366301,366302,366303],arms=['ACTION_CONDITIONED','NO_FUTURE_ACTION'],
                training_replay_result=str(p),training_replay_result_sha256=digest(p),replay_sha256='actual-hash')
            with patch.dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),patch('go2arx5_vla.world_model.fusion_scope.validate'):
                validate_gpu(s,dict(uuid=gpu,free_bytes=5*1024**3))
                with self.assertRaises(PermissionError):validate_gpu(s,dict(uuid=gpu,free_bytes=2*1024**3))
                for change in [dict(eligible_windows=47),dict(excluded=[dict(reason='partial')]),dict(replay_sha256='other')]:
                    p.write_text(json.dumps(dict(value,**change)));s['training_replay_result_sha256']=digest(p)
                    with self.assertRaisesRegex(PermissionError,'ACTUAL_COMPLETE_BALANCED'):validate_gpu(s,dict(uuid=gpu,free_bytes=5*1024**3))


if __name__=='__main__':unittest.main()
