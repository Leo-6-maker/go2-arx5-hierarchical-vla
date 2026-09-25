import unittest

from scripts.pi0246_paired_screen_PT_extension import PROFILES, PT_FAMILY, wanted


class PTScreenBindingTests(unittest.TestCase):
    def test_only_authorized_pt_checkpoint_profiles_are_dispatchable(self):
        self.assertEqual(PROFILES, {"PT1000": ("PT", 1000), "PT2000": ("PT", 2000)})
        self.assertEqual(PT_FAMILY, "PI0246_PT_PHYSPOS_PHYSDELTA_M16_2K_V1")

    def test_inference_scope_excludes_trainer_state_payload(self):
        self.assertTrue(wanted("config.json"))
        self.assertTrue(wanted("model.safetensors.index.json"))
        self.assertTrue(wanted("model-00001-of-00002.safetensors"))
        self.assertFalse(wanted("optimizer.pt"))
        self.assertFalse(wanted("scheduler.pt"))


if __name__ == "__main__":
    unittest.main()
