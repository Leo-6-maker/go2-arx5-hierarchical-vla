"""Planning fixtures only, not actual checkpoint, SSH, model or physics evidence."""
import copy
import unittest

from scripts.pi0292.relay_protocol import digest
from .relay_transport import model_registry


class ModelRegistryTests(unittest.TestCase):
    def fixture(self,step=512):
        cards=[];rows=[]
        split='TRAIN' if step==512 else 'DEV'
        for index in range(2 if step==512 else 4):
            group=f'PI0294/{split}/F0/{index:03d}'
            for task in ('T1','T2'):
                card=dict(id=f'PI0294-{split}-F0-{index:03d}-{task}',task=task,task_id=task,
                    instruction='Pick the cube' if task=='T1' else 'Push the cube without lifting',
                    split=split,scene_group=group,group_id=group,seed=index,initial_state_sha256=digest({'initial':index}),
                    source_xy=[.45,-.08],goal_xy=[.45,.08])
                cards.append(card)
                rows.append(dict(original_attempt_id=card['id'],original_card_sha256=digest(card),
                    scene_group=group,initial_state_sha256=card['initial_state_sha256'],task=task))
        panel=dict(status='FROZEN_BEFORE_FIRST_V3_NEURAL_REQUALIFICATION_AND_MAIN_UPDATE',
            run_uuid='fixture',frozen_epoch=1791920000,panels={'M512_TRAIN' if step==512 else 'SHARED_DEV':rows})
        files={n:dict(bytes=1,sha256='a'*64) for n in ('config.json','model.safetensors','optimizer.pt',
            'scheduler.pt','rng_state.pth','SAMPLER_STATE.json')}
        inventory=dict(kind='main',global_step=step,status='FULL_STATE_BYTES_INVENTORIED_NOT_INDEPENDENT_RESUME_PROOF',
            model_loaded=False,optimizer_pickle_loaded=False,files=files,statistics_sha256='b'*64,manifest_sha256='c'*64)
        delivery=dict(kind='main',global_step=step,status='PASS_COMPLETE_PRIVATE_CHECKPOINT_FIXED_REVISION_READBACK',
            manifest_sha256='d'*64,revision='e'*40,prefix='fixture/complete',files=copy.deepcopy(files))
        return dict(panel=panel,cards=cards,inventory=inventory,delivery=delivery,step=step,run_uuid='fixture',
            core=dict(source_sha='f'*40,manifest_sha256='1'*64),
            hopper=dict(source_sha='2'*40,manifest_sha256='3'*64,data_manifest_sha256='c'*64,statistics_sha256='b'*64),
            deadline_epoch=1791924000,now=1791920100,inventory_sha256='d'*64,
            processor_sha256='4'*64,live_readiness_sha256='5'*64)

    def test_one_delivered_checkpoint_for_both_languages_and_same_noise_layout(self):
        args=self.fixture();value=model_registry(**args)
        self.assertEqual(len(value['cases']),4)
        self.assertEqual(value['status'],'PLANNED_NOT_DISPATCHED_OR_MODEL_BEHAVIOR_QUALIFIED')
        self.assertEqual(value['checkpoint_identity']['HF_revision'],'e'*40)
        cases=list(value['cases'].values())
        for left,right in zip(cases[::2],cases[1::2]):
            self.assertEqual(left['identity'],right['identity'])
            self.assertEqual(left['layout_id'],right['layout_id'])
            self.assertEqual(left['card']['initial_state_sha256'],right['card']['initial_state_sha256'])
            self.assertNotEqual(left['card']['instruction'],right['card']['instruction'])
            self.assertEqual(left['core_request']['profile'],'B1')
        self.assertEqual(value['selected'],'NONE')
        self.assertEqual(len({c['hopper_session_relative'] for c in cases}),1)
        self.assertEqual(args['cards'][0]['id'],'PI0294-TRAIN-F0-000-T1')

    def test_DEV_uses_exact_eight_frozen_cases(self):
        for step in (2048,4096,6144):
            with self.subTest(step=step):
                value=model_registry(**self.fixture(step))
                self.assertEqual(value['category'],'DEV')
                self.assertEqual(len(value['cases']),8)
                self.assertTrue(all(c['card']['split']=='DEV' for c in value['cases'].values()))

    def test_engineering_partial_or_unverified_delivery_rejected(self):
        for key,change in [('inventory',dict(kind='engineering')),('inventory',dict(global_step=4)),
                ('inventory',dict(model_loaded=True)),('delivery',dict(status='PAYLOAD_COMMITTED_FIXED_READBACK_PENDING')),
                ('delivery',dict(manifest_sha256='9'*64)),('hopper',dict(statistics_sha256='9'*64)),
                ('hopper',dict(data_manifest_sha256='9'*64))]:
            with self.subTest(change=change):
                args=self.fixture();args[key].update(change)
                with self.assertRaises(PermissionError):model_registry(**args)

    def test_changed_weight_member_or_missing_resume_state_rejected(self):
        args=self.fixture();args['delivery']['files']['model.safetensors']['sha256']='9'*64
        with self.assertRaises(PermissionError):model_registry(**args)
        args=self.fixture();del args['inventory']['files']['optimizer.pt']
        with self.assertRaises(PermissionError):model_registry(**args)

    def test_changed_language_split_or_pair_physics_rejected(self):
        for key,value in [('instruction','Changed language'),('split','TRAIN'),('source_xy',[.46,-.08])]:
            with self.subTest(key=key):
                args=self.fixture(2048);args['cards'][0][key]=value
                with self.assertRaises(PermissionError):model_registry(**args)
        args=self.fixture();args['cards'][1]['source_xy']=[.46,-.08]
        args['panel']['panels']['M512_TRAIN'][1]['original_card_sha256']=digest(args['cards'][1])
        with self.assertRaisesRegex(PermissionError,'ONLY_LANGUAGE'):model_registry(**args)

    def test_denominator_and_original_cutoff_cannot_be_shrunk_or_extended(self):
        args=self.fixture();args['panel']['panels']['M512_TRAIN'].pop()
        with self.assertRaises(PermissionError):model_registry(**args)
        args=self.fixture();args['deadline_epoch']=1791939601
        with self.assertRaises(PermissionError):model_registry(**args)


if __name__=='__main__':unittest.main()
