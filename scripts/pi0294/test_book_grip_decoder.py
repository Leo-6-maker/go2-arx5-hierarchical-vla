import unittest
from unittest.mock import patch
import numpy as np
from scripts.pi0294.multitask import book_grip_decoder as decoder
from scripts.pi0294.multitask.book_policy import Student


class GripDecoderTests(unittest.TestCase):
    def test_only_grip_changes_and_both_prototypes_are_reachable(self):
        h=np.zeros((40,10),dtype=np.float64);h[:,:9]=np.arange(9)
        h[:,9]=np.linspace(0,1,40);before=h.copy();out=decoder.decode(h)
        np.testing.assert_array_equal(h,before)
        np.testing.assert_array_equal(out[:,:9],h[:,:9])
        self.assertEqual(set(out[:,9]),{decoder.CLOSED,decoder.OPEN})
        h[0,9]=.5;self.assertEqual(decoder.decode(h)[0,9],decoder.OPEN)

    def test_actual_student_path_retains_raw_and_gates_installed_variant(self):
        class Generator:
            def generate(self,request,seed):
                h=np.tile([.2,0,.5,1,0,0,0,1,0,.43],(40,1)).astype(float)
                return dict(physical_H40=h,normalized=h.copy())
        for mode in (None,decoder.VERSION):
            rows=[];s=Student(Generator(),'BOOK_DEV000',lambda kind,v:rows.append((kind,v)),lambda:None,gripper_decoder=mode)
            s.index=0;s.state=np.array([.2,0,.5,1,0,0,0,1,0,.32]);s.images=[np.zeros((256,256,3),dtype=np.uint8)]*2;s.instruction='place the book'
            with patch('scripts.pi0294.multitask.book_policy.payload',return_value={'images':[{'sha256':'a'},{'sha256':'b'}]}),patch('scripts.pi0294.multitask.book_policy.gate_action',return_value={'decision':'ACCEPT'})as gate:
                plan=s.plan()
            expected=.43 if mode is None else decoder.CLOSED
            np.testing.assert_array_equal(plan['opening'],np.full(40,expected))
            value=rows[-1][1]['value'];self.assertEqual(s.queries,1)
            np.testing.assert_array_equal(gate.call_args.args[0],value['physical_H40'])
            self.assertEqual('raw_physical_H40'in value,mode is not None)
            if mode is not None:np.testing.assert_array_equal(value['raw_physical_H40'][:,9],np.full(40,.43))

    def test_explicit_diagnostic_manifest(self):
        decoder.validate_manifest(dict(kind=decoder.KIND,grip_decoder=decoder.VERSION,limits={'optimizer_updates':0}))
        with self.assertRaises(ValueError):decoder.validate_manifest(dict(kind='V7_BOOK_FROZEN_DEV10',grip_decoder=decoder.VERSION,limits={'optimizer_updates':0}))


if __name__=='__main__':unittest.main()
