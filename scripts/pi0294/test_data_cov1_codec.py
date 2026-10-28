"""Synthetic numerical receipts, never evidence of real official qualification."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts.pi0294.data_cov1.codec import Trace,audit
from scripts.pi0294.pv1.export import packet,codec_packet,committed_packet
from scripts.pi0294.test_pv1_export import fixture,FakeProcessor


class DataCov1Codec(unittest.TestCase):
    def test_observer_does_not_change_legacy_receipt_or_source_arrays(self):
        card,samples=fixture();arrays=packet(samples,card);before=deepcopy(arrays)
        enums=SimpleNamespace(ActionType=SimpleNamespace(EEF=1),ActionFormat=SimpleNamespace(XYZ_ROT6D=1))
        tag=SimpleNamespace(EmbodimentTag=SimpleNamespace(NEW_EMBODIMENT='new_embodiment'))
        with patch.dict('sys.modules',{'gr00t.data.types':enums,'gr00t.data.embodiment_tags':tag}):
            old=codec_packet(arrays,FakeProcessor());new=audit(arrays,FakeProcessor(),lambda:None)
        self.assertEqual(old,{k:v for k,v in new.items() if k!='diagnostics'})
        self.assertIsNone(new['diagnostics']['first_error'])
        self.assertEqual(new['diagnostics']['windows_actually_observed'],601)
        for key,value in arrays.items():self.assertTrue(np.array_equal(value,before[key]))
        for key,value in packet(samples,card).items():self.assertTrue(np.array_equal(value,committed_packet(samples,card)[key]))

    def test_first_error_is_earliest_anchor_then_point_not_largest_error(self):
        card,samples=fixture();arrays=packet(samples,card);trace=Trace()
        target=arrays['committed_absolute10'][:40];state=arrays['actual_state10'][0]
        normalized=target.copy();normalized[25,2]=1.2
        actual=target.copy();actual[25,2]+=.00002;actual[30,2]+=.003
        trace(anchor=0,state=state,target=target,expected_relative=target,native_relative=target[:,:9],
            normalized=normalized,packed=normalized.astype(np.float32),actual=actual)
        self.assertEqual(trace.first_error['action_index'],25)
        self.assertEqual(trace.first_error['normalized_channel_indices_with_inverse_clip'],[2])
        self.assertFalse(trace.first_error['relative_coordinate_2_is_world_Z'])
        trace(anchor=1,state=state,target=target,expected_relative=target,native_relative=target[:,:9],
            normalized=normalized,packed=normalized.astype(np.float32),actual=actual)
        self.assertEqual(trace.first_error['window_anchor'],0)
        with self.assertRaises(ValueError):
            trace(anchor=3,state=state,target=target,expected_relative=target,native_relative=target[:,:9],
                normalized=normalized,packed=normalized.astype(np.float32),actual=actual)


if __name__=='__main__':unittest.main()
