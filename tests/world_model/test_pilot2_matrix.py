import unittest,tempfile,json,time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from scripts.wm02.train_pilot2_matrix import validate_design,PLAN_SHA
from go2arx5_vla.world_model.actor_sampling import MODES


class Pilot2Matrix(unittest.TestCase):
    def spec(self):
        return dict(pilot2_plan_sha=PLAN_SHA,study_arms=[dict(dataset=d,sampler=m,seed=s)
            for d in ('OLD_24','EXPANDED_53')for m in MODES for s in (36001,36002,36003)],
            endpoint=96000,arm_endpoint=8000,batch_size=128,learning_rate=.001,physics=False,kind='ACTOR_STUDY',
            study_datasets=dict(OLD_24=dict(TRAIN_windows=24),EXPANDED_53=dict(TRAIN_windows=53)))

    def test_full_factorial_and_actual_total_updates(self):
        s=self.spec();self.assertEqual(len(validate_design(s)),12)
        self.assertEqual(len(s['study_arms'])*s['arm_endpoint'],s['endpoint'])
        for epochs in (20,40):
            self.assertEqual(epochs*24*256//128*128/(24*256),epochs)
            self.assertEqual(epochs*53*256//128*128/(53*256),epochs)

    def test_missing_duplicate_arm_or_wrong_recipe_is_rejected(self):
        for change in ('duplicate','endpoint','data'):
            s=self.spec()
            if change=='duplicate':s['study_arms'][-1]=s['study_arms'][0]
            elif change=='endpoint':s['endpoint']=8000
            else:s['study_datasets']['OLD_24']['TRAIN_windows']=25
            with self.assertRaises(ValueError):validate_design(s)

    def test_continuation_counts_only_remaining_models(self):
        s=self.spec();s.update(completed_arm_proofs={'0':{}},endpoint=88000)
        self.assertEqual(len(validate_design(s)),12)
        s['endpoint']=96000
        with self.assertRaises(ValueError):validate_design(s)

    def test_all_twelve_model_progress_updates_use_replaceable_records(self):
        # Synthetic orchestration only; no CUDA, optimizer or physics executes.
        from scripts.wm02.train_pilot2_matrix import run
        from go2arx5_vla.world_model.storage import digest
        with tempfile.TemporaryDirectory()as tmp:
            r=Path(tmp);out=r/'fit';out.mkdir();fixture=r/'fixture';fixture.write_text('SYNTHETIC_NOT_PHYSICAL')
            s=self.spec();s.update(source_sha='fixture',source_root=str(r),cpus=[0],allocator_limit_bytes=10,
                deadline_epoch=time.time()+60,registered_root=str(r),campaign_root=str(r),frozen_task=str(fixture),
                frozen_task_sha256=digest(fixture),ARX=str(fixture),ARX_sha256=digest(fixture))
            for d in s['study_datasets'].values():d.update(path=str(fixture),sha256=digest(fixture),legacy_checkpoint=str(fixture),legacy_checkpoint_sha256=digest(fixture))
            def fake_train(spec,root):
                (root/'OPTIMIZER_STEPS.jsonl').write_text(json.dumps(dict(update=8000))+'\n')
                (root/'RESTORE_PROBE.json').write_text(json.dumps(dict(actual_optimizer_calls=1)))
                return dict(main_Actor_updates=8000,restore_probe_optimizer_calls=1)
            with patch('scripts.wm02.train_pilot2_matrix.audit_checkpoint',return_value=dict(SYNTHETIC=True)),\
                 patch('scripts.wm02.train_pilot1_actor.run',side_effect=fake_train),\
                 patch('go2arx5_vla.world_model.pilot1_storage.local_path'),\
                 patch('torch.set_num_threads'),patch('torch.use_deterministic_algorithms'),\
                 patch('torch.cuda.set_per_process_memory_fraction'),\
                 patch('torch.cuda.get_device_properties',return_value=SimpleNamespace(total_memory=1024)),\
                 patch('torch.cuda.empty_cache'):
                result=run(s,out)
            self.assertEqual(result['main_Actor_updates'],96000)
            self.assertEqual(len(json.loads((out/'COMPLETED_ARMS.json').read_text())),12)
            self.assertEqual(json.loads((out/'PROGRESS.json').read_text())['arm_index'],11)


if __name__=='__main__':unittest.main()
