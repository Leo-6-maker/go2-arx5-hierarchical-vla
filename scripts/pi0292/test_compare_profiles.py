import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .compare_profiles import PROFILES, aggregate, verify_arrays
from .eval_contract import error_metrics, gate_action
from .train_adapter import sha


class TestPairedOfflineAttribution(unittest.TestCase):
    def setUp(self):
        self.state = np.array([.4, 0., .5, 1., 0., 0., 0., 1., 0., 1.])
        self.expert = np.repeat(self.state[None], 40, axis=0)
        self.noise_bits = np.zeros((1,40,264), dtype=np.uint8)
        self.noise = dict(shape=[1,40,132],dtype='torch.bfloat16',
                          sha256=hashlib.sha256(self.noise_bits.tobytes()).hexdigest())
        self.probe = dict(status='SEALED_BEFORE_MAIN', expert_rejects=0,
            noise_banks=[29201,29202], native_inference_steps=4, per_profile_queries=4,
            probes=[dict(id=identifier,anchor=0,split=split,stages=['start','close'],
                samples_sha256='a'*64,numeric_sha256='b'*64,
                noise_seeds={'29201':'1','29202':'2'})
                for identifier,split in [('T291000','TRAIN'),('D291000','DEV')]])
        self.records = {}
        for profile, displacement in zip(PROFILES,(.02,.01,.005)):
            rows=[]
            for window in self.probe['probes']:
                for bank in self.probe['noise_banks']:
                    prediction=self.expert.copy();prediction[:,0]+=displacement
                    rows.append(dict(**copy.deepcopy(window),profile=profile,noise_bank=bank,
                        query_id=f"{window['id']}-000-{bank}",noise=copy.deepcopy(self.noise),
                        native_inverse_clip=True,extra_correction=False,
                        gate=dict(decision='REJECT' if profile=='B0' else 'ACCEPT',
                            issues=[dict(code='SYNTHETIC_REJECT')] if profile=='B0' else []),
                        errors=error_metrics(prediction,self.expert,self.state)))
            self.records[profile]=rows

    def test_paired_difference_keeps_rejected_rows_and_fixed_candidate(self):
        result=aggregate(self.probe,self.records)
        delta=result['paired_error_differences']['TRAIN']['S4096_minus_B0']['episode_macro_of_mean_differences']
        self.assertAlmostEqual(delta['H40']['XYZ_euclidean_m']['mean'],-.015)
        self.assertEqual(result['profiles']['B0']['TRAIN']['decisions'],{'REJECT':2})
        self.assertEqual(result['stage_attribution']['TRAIN']['B0']['close']['queries'],2)
        self.assertEqual(result['B1'],'FIXED_S4096')
        self.assertEqual(result['checkpoint_selection'],'NONE')
        self.assertFalse(result['rejected_rows_excluded'])

    def test_missing_duplicate_and_extra_query_are_rejected(self):
        for kind in ('missing','duplicate','extra'):
            with self.subTest(kind=kind):
                rows=copy.deepcopy(self.records)
                if kind=='missing':rows['B0'].pop()
                elif kind=='duplicate':rows['B0'].append(copy.deepcopy(rows['B0'][0]))
                else:
                    extra=copy.deepcopy(rows['B0'][0]);extra['anchor']=1
                    rows['B0'].append(extra)
                with self.assertRaisesRegex(ValueError,'QUERY_SET_INCOMPLETE_DUPLICATE_OR_EXTRA'):
                    aggregate(self.probe,rows)

    def test_actual_noise_and_source_identity_must_match(self):
        for kind in ('noise','numeric','split','stage'):
            with self.subTest(kind=kind):
                rows=copy.deepcopy(self.records)
                row=rows['S4096'][0]
                if kind=='noise':row['noise']['sha256']='c'*64
                elif kind=='numeric':row['numeric_sha256']='c'*64
                elif kind=='split':row['split']='DEV'
                else:row['stages']=['hold']
                with self.assertRaises(ValueError):aggregate(self.probe,rows)

    def test_incomplete_errors_remain_in_denominator_and_null_macro(self):
        rows=copy.deepcopy(self.records)
        rows['S4096'][0]['errors']['pointwise']['XYZ_euclidean_m'][0]=None
        result=aggregate(self.probe,rows)
        macro=result['paired_error_differences']['TRAIN']['S4096_minus_B0']['episode_macro_of_mean_differences']
        self.assertIsNone(macro['H40']['XYZ_euclidean_m']['mean'])
        own=result['profiles']['S4096']['TRAIN']['episodes']['T291000']['H40']['XYZ_euclidean_m']
        self.assertEqual(own['expected_points'],80)
        self.assertEqual(own['finite_points'],79)

    def test_events_report_missing_and_spurious_without_dropping_cases(self):
        rows=copy.deepcopy(self.records)
        rows['S4096'][0]['errors']['close_events'].update(expert_indices=[3],predicted_indices=[],missing=True)
        result=aggregate(self.probe,rows)
        events=result['events']['TRAIN']['S4096']['close_events']
        self.assertEqual(events['planned_queries'],2)
        self.assertEqual(events['missing'],1)
        self.assertEqual(events['expected_events'],1)
        self.assertEqual(events['both_present'],0)

    def test_saved_arrays_recompute_errors_and_original_noise(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'record.npz'
            row=copy.deepcopy(self.records['S4096'][0])
            prediction=self.expert.copy();prediction[:,0]+=.005
            row['gate']=gate_action(prediction,self.state,0,row['query_id'])
            normalized=np.zeros((1,40,132),dtype=np.float32)
            def save():
                np.savez_compressed(path,normalized_action=normalized,decoded_action=prediction,
                    expert=self.expert,actual_state=self.state,noise_bits=self.noise_bits)
                row.update(array_sha256=sha(path),normalized_outside_unit_components=0)
            save()
            anchors=verify_arrays(path,row)
            self.assertEqual(anchors['actual_state'],hashlib.sha256(self.state.tobytes()).hexdigest())
            row['array_sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'ARRAY_BYTES_CHANGED'):verify_arrays(path,row)
            save();row['noise']['sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'NATIVE_NOISE_CHANGED'):verify_arrays(path,row)
            row['noise']=copy.deepcopy(self.noise);row['errors']['pointwise']['XYZ_euclidean_m'][0]=4.
            with self.assertRaisesRegex(ValueError,'ERROR_RECORD_DOES_NOT_MATCH'):verify_arrays(path,row)
            row['errors']=error_metrics(prediction,self.expert,self.state)
            row['gate']=dict(decision='ACCEPT',issues=[])
            with self.assertRaisesRegex(ValueError,'GATE_RECORD_DOES_NOT_MATCH'):verify_arrays(path,row)

    def test_partial_profile_or_changed_decode_contract_cannot_pass(self):
        rows=copy.deepcopy(self.records);rows.pop('S2048')
        with self.assertRaisesRegex(ValueError,'ALL_THREE_REGISTERED'):aggregate(self.probe,rows)
        rows=copy.deepcopy(self.records);rows['B0'][0]['extra_correction']=True
        with self.assertRaisesRegex(ValueError,'POSTPROCESSING_CHANGED'):aggregate(self.probe,rows)


if __name__=='__main__':unittest.main()
