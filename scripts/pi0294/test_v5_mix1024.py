import copy
from collections import Counter
import unittest
from types import SimpleNamespace
from scripts.pi0294.pv1 import mix1024 as m
from scripts.pi0294.pv1.sampler import MatchedSampler
from scripts.pi0294.pv1.recipe import validate_primary
from scripts.pi0294.test_pv1_recipe import old_recipe
from scripts.pi0294.test_pv1_sampler import dataset
from scripts.pi0294.e1.scope import STATS

class MatchedMixTests(unittest.TestCase):
    def setUp(self):self.d=dataset('COVERAGE_FIRST','B_COVERAGE')
    def test_all_32768_positions_paired_per_batch_not_expectations(self):
        c=m.MixSampler(self.d,arm='C50');d=m.MixSampler(self.d,arm='D25');old=MatchedSampler(self.d,stop=m.VISITS)
        for start in range(0,m.VISITS,32):
            left=[c.selection_at(i) for i in range(start,start+32)];right=[d.selection_at(i) for i in range(start,start+32)]
            self.assertEqual(Counter(x['lane'] for x in left),dict(C1_T1=8,original_T1=4,new_T1=4,original_T2=8,new_T2=8))
            self.assertEqual(Counter(x['lane'] for x in right),dict(C1_T1=8,original_T1=6,new_T1=2,original_T2=12,new_T2=4))
            for i,(a,b) in enumerate(zip(left,right),start):
                self.assertEqual(a,old.selection_at(i))
                if b['lane'].startswith('new_'):self.assertEqual(a,b)
                if a['lane']==b['lane']:self.assertEqual(a,b)
                self.assertEqual(a['task'],b['task'])
    def test_recipe_only_arm_and_mix_differ_and_old_validator_refuses(self):
        old=old_recipe();c=m.recipe(old,'C50');d=m.recipe(old,'D25')
        for k in c:
            if k not in ['arm','new_expert_fraction','entire_branch_new_fraction']:self.assertEqual(c[k],d[k])
        for r in [c,d]:
            with self.assertRaises(PermissionError):validate_primary(r,old)
            for field,value in [('max_steps',4096),('seed',20261020),('warmup_updates',32),('statistics_fit',True)]:
                bad=copy.deepcopy(r);bad[field]=value
                with self.assertRaises(PermissionError):m.validate(bad,old)
    def test_resume_is_exact_suffix_and_state_identity_differs(self):
        for arm in m.ARMS:
            full=m.MixSampler(self.d,arm=arm);baseline=list(full)
            for step in [1,256,1023]:
                part=m.MixSampler(self.d,arm=arm,start=step*32)
                self.assertEqual(baseline[step*32:],list(part));self.assertEqual(full.state_dict(step*32),part.state_dict(step*32))
            with self.assertRaises(ValueError):full.state_dict(17)
    def test_every_lr_and_endpoints_match_new_schedule(self):
        old=old_recipe();c=m.recipe(old,'C50');d=m.recipe(old,'D25');a,b=m.lr_schedule(c),m.lr_schedule(d)
        self.assertEqual(a,b);self.assertEqual(len(a),1025);self.assertEqual(a[0],0);self.assertEqual(a[64],1e-5);self.assertAlmostEqual(a[-1],1e-6)
        from scripts.pi0294.e1.trainer import learning_rate_multiplier
        self.assertNotEqual(a[1024],1e-5*learning_rate_multiplier(1024,4096,64))
        self.assertEqual(m.trainer_arguments(c,old)['max_steps'],1024)
    def test_exact_stop_and_pending_unknown_refusal(self):
        for step in range(1025):self.assertEqual(m.step_boundary(step,step*32,pending=[],unknown=[])['should_training_stop'],step==1024)
        for args in [(1025,32800,[],[]),(2,63,[],[]),(2,64,[{}],[]),(2,64,[],['UNKNOWN'])]:
            with self.assertRaises(PermissionError):m.step_boundary(args[0],args[1],pending=args[2],unknown=args[3])
    def test_full_resume_rejects_optimizer_stats_seed_or_cursor_changes(self):
        s=m.MixSampler(self.d,arm='C50');r=m.recipe(old_recipe(),'C50');step=256
        files={n:'FAKE_MEMBER_SHA' for n in ['config.json','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','SAMPLER_STATE.json','processor/config.json','model.safetensors']};files['statistics.json']=STATS
        complete=dict(global_step=step,files=files,full_resume=True);saved=dict(recipe=r,sampler=s.state_dict(step*32),manifest_sha256=self.d.manifest_sha,statistics_sha256=STATS)
        self.assertEqual(m.resume_state(saved,complete,dict(global_step=step),r,s,files),step)
        for changed in ['optimizer.pt','statistics.json']:
            bad=dict(files);bad[changed]='CHANGED'
            with self.assertRaises(PermissionError):m.resume_state(saved,complete,dict(global_step=step),r,s,bad)
        wrong_arm=m.recipe(old_recipe(),'D25');wrong_saved=copy.deepcopy(saved);wrong_saved['recipe']=wrong_arm
        with self.assertRaises(PermissionError):m.resume_state(wrong_saved,complete,dict(global_step=step),wrong_arm,s,files)
        bad=copy.deepcopy(saved);bad['sampler']['cursor']-=32
        with self.assertRaises(PermissionError):m.resume_state(bad,complete,dict(global_step=step),r,s,files)
    def test_split_or_task_contamination_rejected_by_mature_support_guard(self):
        self.d.rows[0][0]['split']='DEV'
        with self.assertRaises(PermissionError):m.MixSampler(self.d,arm='D25')
    def test_old_run_identity_or_reseed_is_not_accepted(self):
        for arm in ['A_CONTROL','B_COVERAGE','D50']:
            with self.assertRaises(PermissionError):m.MixSampler(self.d,arm=arm)
        with self.assertRaises(PermissionError):m.MixSampler(self.d,arm='D25',seed=20261020)
    def test_interrupted_256_save_is_full_resume_not_legacy_inference_only(self):
        import sys,tempfile,json,hashlib
        from pathlib import Path
        from types import ModuleType
        from unittest.mock import patch
        class FakeFramework:
            @staticmethod
            def _save_checkpoint(obj,model,trial):
                p=Path(obj.args.output_dir)/('checkpoint-'+str(obj.state.global_step));p.mkdir()
                for n in ['config.json','optimizer.pt','scheduler.pt','rng_state.pth','model.safetensors']:(p/n).write_text('TEMP_CPU_FIXTURE')
                (p/'trainer_state.json').write_text(json.dumps(dict(global_step=obj.state.global_step)))
                if obj.args.save_only_model:raise AssertionError('interrupted save lost optimizer')
        class FakeMature:pass
        modules={n:ModuleType(n) for n in ['gr00t','gr00t.experiment','gr00t.experiment.trainer']}
        modules['gr00t.experiment.trainer'].Gr00tTrainer=FakeFramework
        with tempfile.TemporaryDirectory() as tmp,patch.dict(sys.modules,modules),patch('scripts.pi0294.mp1.trainer.make_trainer_class',return_value=FakeMature),patch('scripts.pi0294.e1.trainer.check_trace',return_value={'fixture':'TRACE_ALREADY_CHECKED'}):
            root=Path(tmp);data=root/'dataset';data.mkdir();(data/'statistics.json').write_text('{}')
            stats=hashlib.sha256((data/'statistics.json').read_bytes()).hexdigest();(root/'RUN_MANIFEST.json').write_text(json.dumps(dict(code=dict(vla_commit='TEMP_CPU_FIXTURE'))))
            output=root/'training/C50';output.mkdir(parents=True)
            class Processor:
                def save_pretrained(self,path):path.mkdir();(path/'config.json').write_text('{}')
            self.d.root=data;self.d.statistics_sha=stats;self.d.processor=Processor();self.d.trace_root=root/'worker_trace'
            with patch.object(m,'STATS',stats):
                cls=m.make_trainer_class();obj=object.__new__(cls);obj.consumed=256*32;obj.start_cursor=0;obj.state=SimpleNamespace(global_step=256);obj.args=SimpleNamespace(output_dir=str(output),save_only_model=True)
                obj.train_dataset=self.d;obj.sampler=m.MixSampler(self.d,arm='C50');obj.recipe=m.recipe(old_recipe(),'C50')
                obj._save_checkpoint(None,None)
            complete=json.loads((output/'checkpoint-256/CHECKPOINT_COMPLETE.json').read_text())
            self.assertTrue(complete['full_resume']);self.assertIn('optimizer.pt',complete['files']);self.assertIn('processor/config.json',complete['files'])
