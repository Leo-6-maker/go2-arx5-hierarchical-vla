from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.pi0294.e1.journal import Journal,MutationOutcomeUnknown
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.pv1 import probes
from scripts.pi0294.pv1.probes import plan,LANES,ANCHORS


def fixture(margin):
    rows=[]
    for lane in LANES:
        for i in range(2):
            task='T1' if lane.endswith('T1') else 'T2'
            row=dict(id=lane+'-'+str(i)+('-'+margin if lane=='new_T2' else ''),
                scene_group='PV1/TRAIN/group-'+str(i),PV1_lane=lane,task=task,split='TRAIN')
            rows.extend((row,anchor) for anchor in ANCHORS)
    return rows


class FixedProbeContracts(unittest.TestCase):
    def test_twelve_fixed_train_only_draws_not_task_success_or_extra_physics(self):
        v=plan(fixture('C04'))
        self.assertEqual(v['count'],12);self.assertEqual(sum(r['task']=='T1' for r in v['windows']),6)
        self.assertEqual({r['split'] for r in v['windows']},{'TRAIN'})
        self.assertEqual(v['physical_rollouts'],0);self.assertFalse(v['DEV_or_LOCKBOX_used'])
        self.assertTrue(v['query_plan_is_not_a_task_success_or_retraining_gate'])

    def test_teacher_variant_ids_cannot_change_paired_parent_anchor_or_noise(self):
        self.assertEqual(plan(fixture('C04')),plan(fixture('C12')))

    def test_missing_lane_anchor_or_wrong_task_rejected_before_any_draw(self):
        for change in ('lane','anchor','task'):
            rows=deepcopy(fixture('C04'))
            if change=='lane':rows=[(r,a) for r,a in rows if r['PV1_lane']!='new_T2']
            if change=='anchor':rows=rows[1:]
            if change=='task':rows[0][0]['task']='T2'
            with self.subTest(change=change),self.assertRaises(PermissionError):plan(rows)

    def test_probe_interruption_reuses_known_draws_and_generates_only_missing_fixed_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dataset,instance=self.probe_inputs(root)
            operations=[];draw=self.journal_draw(root,operations)
            guard_calls=[]
            def interrupt():
                guard_calls.append(True)
                if len(guard_calls)==4:raise TimeoutError('NO_NEW_DRAW_AFTER_OWNER_LOSS')
            with patch.object(probes.model,'payload',return_value={'fixed':'observation'}),\
                    patch.object(probes.model,'fixed_draw',side_effect=draw),\
                    patch.object(probes,'error_metrics',return_value={'fixture':True}),\
                    patch.object(probes,'gate_action',return_value={'decision':'REJECT'}):
                with self.assertRaises(TimeoutError):
                    probes.run(root,'A_CONTROL-1024',instance,dataset,None,interrupt)
                self.assertEqual(len(operations),3)
                partial=json.loads((root/'evidence/A_CONTROL-1024/TRAIN_PROBES/RESULT.json').read_text())
                self.assertEqual(partial['actual_new_queries_this_invocation'],3)
                self.assertEqual(partial['reused_completed_draws'],0)
                self.assertTrue(instance.model.training)
                result=probes.run(root,'A_CONTROL-1024',instance,dataset,None,lambda:None)
                self.assertEqual(len(operations),12)
                self.assertTrue(result['complete']);self.assertEqual(result['new_queries'],12)
                self.assertEqual(result['actual_new_queries_this_invocation'],9)
                self.assertEqual(result['reused_completed_draws'],3)
                self.assertFalse(result['physical_success_gate'])
                self.assertEqual({r['original_gate']['decision'] for r in result['records']},{'REJECT'})
                again=probes.run(root,'A_CONTROL-1024',instance,dataset,None,lambda:None)
                self.assertEqual(result['records'],again['records'])
                self.assertEqual(again['actual_new_queries_this_invocation'],0)
                self.assertEqual(again['reused_completed_draws'],12)
                self.assertEqual(again['new_queries'],12)
                self.assertEqual(len(operations),12)
                self.assertTrue(instance.model.training)
                dataset.manifest_sha='CHANGED_BANK'
                with self.assertRaises(PermissionError):
                    probes.run(root,'A_CONTROL-1024',instance,dataset,None,lambda:None)
                self.assertEqual(len(operations),12)

    def test_unknown_probe_intent_and_unbound_prior_folder_are_never_replayed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dataset,instance=self.probe_inputs(root);operations=[]
            def uncertain():raise RuntimeError('UNKNOWN_NATIVE_QUERY_EFFECT')
            journal=Journal(root/'reservations/probe-generations',root.name)
            with self.assertRaises(RuntimeError):
                journal.once('probe-A_CONTROL-1024-original_T1-000',{'fixed':'observation'},uncertain)
            with patch.object(probes.model,'payload',return_value={'fixed':'observation'}),\
                    patch.object(probes.model,'fixed_draw',side_effect=self.journal_draw(root,operations)):
                with self.assertRaises(MutationOutcomeUnknown):
                    probes.run(root,'A_CONTROL-1024',instance,dataset,None,lambda:None)
            self.assertEqual(operations,[]);self.assertTrue(instance.model.training)
            folder=root/'evidence/B_COVERAGE-1024/TRAIN_PROBES';folder.mkdir(parents=True)
            (folder/'prior.npz').write_text('PRESERVE_UNKNOWN_PRIOR_OUTPUT')
            with self.assertRaises(PermissionError):
                probes.run(root,'B_COVERAGE-1024',instance,dataset,None,lambda:None)
            self.assertEqual((folder/'prior.npz').read_text(),'PRESERVE_UNKNOWN_PRIOR_OUTPUT')

    def test_probe_binding_symlink_is_rejected_without_read_or_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dataset,instance=self.probe_inputs(root)
            folder=root/'evidence/A_CONTROL-1024/TRAIN_PROBES';folder.mkdir(parents=True)
            original=root/'DO_NOT_READ_OR_OVERWRITE.json';original.write_text('UNRELATED_ORIGINAL_BYTES')
            (folder/'BINDING.json').symlink_to(original)
            with patch.object(probes.model,'fixed_draw') as draw,self.assertRaises(PermissionError):
                probes.run(root,'A_CONTROL-1024',instance,dataset,None,lambda:None)
            draw.assert_not_called();self.assertEqual(original.read_text(),'UNRELATED_ORIGINAL_BYTES')
            self.assertTrue(instance.model.training)

    @staticmethod
    def probe_inputs(root):
        path=root/probes.PLAN;path.parent.mkdir(parents=True)
        rows=fixture('C04');path.write_text(json.dumps(plan(rows)))
        (root/'RUN_MANIFEST.json').write_text(json.dumps(dict(execution=dict(sealed_inputs={probes.PLAN:sha(path)}))))
        dataset=SimpleNamespace(rows=rows,manifest_sha='FROZEN_BANK',raw=lambda index:([],[],[],''))
        model=SimpleNamespace(training=True)
        def train(mode=True):model.training=mode
        model.train=train;model.eval=lambda:train(False)
        instance=SimpleNamespace(model=model,identity=dict(global_step=1024,checkpoint_manifest_sha256='EXACT_1024'))
        return dataset,instance

    @staticmethod
    def journal_draw(root,operations):
        def draw(_root,label,identifier,instance,payload,seed,target,ledger,**kwargs):
            def generate():
                operations.append(identifier);target.write_text(json.dumps({'seed':seed}))
                return dict(native_expert_roundtrip_maximum_absolute_difference=0,
                    noise={'seed':seed},inference_precision={'frozen':True},arrays_sha256=sha(target))
            metadata=Journal(root/'reservations/probe-generations',root.name).once('probe-'+label+'-'+identifier,payload,generate)
            if sha(target)!=metadata['arrays_sha256']:raise ValueError('CHANGED_DRAW_ARRAY')
            return [],metadata
        return draw
