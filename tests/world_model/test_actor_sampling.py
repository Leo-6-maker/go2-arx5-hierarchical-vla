import unittest
from copy import deepcopy
import numpy as np
from go2arx5_vla.world_model.actor_sampling import ExactSampler, MODES


class ActorSampling(unittest.TestCase):
    def test_original_order_is_exact_across_batches_and_epochs(self):
        s=ExactSampler(5,MODES[0],36001)
        np.testing.assert_array_equal(np.concatenate([s.take(3),s.take(8),s.take(4)]),np.arange(15)%5)
        np.testing.assert_array_equal(s.exposure_counts(),np.full(5,3))

    def test_shuffle_is_uniform_without_replacement_and_seeded(self):
        a=ExactSampler(101,MODES[1],36001);b=ExactSampler(101,MODES[1],36001);c=ExactSampler(101,MODES[1],36002)
        x=a.take(101);np.testing.assert_array_equal(x,b.take(101))
        np.testing.assert_array_equal(np.sort(x),np.arange(101));self.assertFalse(np.array_equal(x,c.take(101)))
        self.assertFalse(np.array_equal(x,a.take(101)))

    def test_resume_partial_batch_and_exact_epoch_includes_future_generator(self):
        for mode in MODES:
            for consumed in [3,7,14,23]:
                with self.subTest(mode=mode,consumed=consumed):
                    a=ExactSampler(7,mode,36001);a.take(consumed);state=a.state_dict();b=ExactSampler(7,mode,36001);b.load_state_dict(state)
                    for _ in range(5):np.testing.assert_array_equal(a.take(5),b.take(5))
                    self.assertEqual(a.state_dict(),b.state_dict());self.assertEqual(state['draws'],consumed)
                    np.testing.assert_array_equal(a.exposure_counts(),b.exposure_counts())

    def test_corrupt_or_different_sampler_state_rejected(self):
        a=ExactSampler(7,MODES[1],36001);a.take(3);state=a.state_dict()
        for change in [dict(seed=1),dict(size=8),dict(mode=MODES[0]),dict(offset=8),dict(draws=2),dict(permutation=[0]*7)]:
            with self.subTest(change=change),self.assertRaises(ValueError):
                a.load_state_dict(dict(deepcopy(state),**change))


if __name__=='__main__':unittest.main()
