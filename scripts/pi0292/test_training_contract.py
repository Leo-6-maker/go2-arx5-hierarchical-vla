"""CPU contract tests; these are not native model qualification."""
import json
from pathlib import Path
import tempfile
import unittest

from .train_adapter import CyclicSampler
from .trainer_adapter import validate_worker_trace


class IdentityDataset:
    def __len__(self):return 17
    def identity(self,index):return dict(episode_id='synthetic',split='TRAIN',anchor=index)


class TestTrainingContract(unittest.TestCase):
    def test_positioned_sampler_resume_and_full_cycles(self):
        full=list(CyclicSampler(17,stop=70,positioned=True))
        self.assertEqual([n for n,_ in full],list(range(70)))
        for cycle in range(4):self.assertEqual(sorted(i for _,i in full[cycle*17:(cycle+1)*17]),list(range(17)))
        for start in (1,16,17,32,64):
            self.assertEqual(list(CyclicSampler(17,start=start,stop=70,positioned=True)),full[start:])
        self.assertEqual(CyclicSampler(17,stop=70).state_dict(32)['cursor'],32)

    def test_consumed_trace_excludes_prefetch_and_rejects_wrong_ids(self):
        dataset=IdentityDataset()
        rows=[dict(cursor=n,index=i,**dataset.identity(i)) for n,i in CyclicSampler(17,stop=70,positioned=True)]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'worker-1.jsonl'
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            result=validate_worker_trace(directory,dataset,0,32)
            self.assertEqual(result['count'],32)
            rows[10]['index']+=1
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with self.assertRaisesRegex(ValueError,'ORDER_MISMATCH'):validate_worker_trace(directory,dataset,0,32)

    def test_missing_and_duplicate_consumed_records_fail(self):
        dataset=IdentityDataset()
        rows=[dict(cursor=n,index=i,**dataset.identity(i)) for n,i in CyclicSampler(17,stop=32,positioned=True)]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'worker-1.jsonl'
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows[:-1]))
            with self.assertRaises(ValueError):validate_worker_trace(directory,dataset,0,32)
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows+[rows[0]]))
            with self.assertRaisesRegex(ValueError,'DUPLICATE'):validate_worker_trace(directory,dataset,0,32)


if __name__=='__main__':unittest.main()
