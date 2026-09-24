import unittest

from scripts.pi0246_gate_pass_train import CampaignError, validate_execution_profile


class ExecutionProfileTests(unittest.TestCase):
    def test_h200_remains_pbs_only(self):
        result = validate_execution_profile("H200_PBS", "job-1", 1, "NVIDIA H200")
        self.assertEqual(result["launcher"], "PBS")
        with self.assertRaises(CampaignError):
            validate_execution_profile("H200_PBS", None, 1, "NVIDIA H200")

    def test_4080s_is_an_explicit_direct_profile(self):
        result = validate_execution_profile("CORELAB_4080S_DIRECT", None, 1, "NVIDIA GeForce RTX 4080 SUPER")
        self.assertEqual(result["launcher"], "direct")
        with self.assertRaises(CampaignError):
            validate_execution_profile("CORELAB_4080S_DIRECT", "job-2", 1, "NVIDIA GeForce RTX 4080 SUPER")
        with self.assertRaises(CampaignError):
            validate_execution_profile("CORELAB_4080S_DIRECT", None, 2, "NVIDIA GeForce RTX 4080 SUPER")
        with self.assertRaises(CampaignError):
            validate_execution_profile("CORELAB_4080S_DIRECT", None, 1, "NVIDIA GeForce RTX 4080")


if __name__ == "__main__":
    unittest.main()
