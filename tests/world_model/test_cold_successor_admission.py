import unittest,tempfile,json
from pathlib import Path
from go2arx5_vla.world_model.storage import digest
from go2arx5_vla.world_model.auto1_native import standing_repair_ancestor,phase_limit

class ColdSuccessorAdmission(unittest.TestCase):
    def test_applied_clock_successor_requires_both_exact_headers(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)
            for n in ('binary','sdk','control_clock.hpp','applied_control_clock.hpp'):(p/n).write_text(n)
            version='FROZEN_PERIOD_PD_POLICY_APPLIED_CLOCK_V2'
            build=dict(source_sha='source',sdk_source=str(p/'sdk'),ancestor_binary_sha256='original-negative',physical_callback_clock=version)
            receipt=dict(status='INIT_CANDIDATE_COMPILED_AND_LOADER_VERIFIED_NOT_NATIVE_QUALIFIED',source_sha='source',binary_sha256=digest(p/'binary'),cold_history_init='ORIGINAL_BUFFER_RESET_FIRST_REAL_OBSERVATION_V1',ongoing_control_math_unchanged=True,original_R1_math_assertions_pass=True,original_SDK_sha256=digest(p/'sdk'),physical_callback_clock=version,control_clock_header_sha256=digest(p/'control_clock.hpp'),applied_control_clock_header_sha256=digest(p/'applied_control_clock.hpp'))
            (p/'build').write_text(json.dumps(build));(p/'receipt').write_text(json.dumps(receipt))
            spec=dict(native_binary=str(p/'binary'),cold_history_build_spec=str(p/'build'),cold_history_build_spec_sha256=digest(p/'build'),cold_history_build_receipt=str(p/'receipt'),cold_history_build_receipt_sha256=digest(p/'receipt'))
            self.assertEqual(standing_repair_ancestor(spec),'original-negative')
            (p/'applied_control_clock.hpp').write_text('unbound edit')
            with self.assertRaisesRegex(PermissionError,'EXACT_APPLIED_CLOCK'):standing_repair_ancestor(spec)
            (p/'applied_control_clock.hpp').unlink()
            with self.assertRaisesRegex(PermissionError,'EXACT_APPLIED_CLOCK'):standing_repair_ancestor(spec)

    def test_successor_keeps_negative_ancestor_and_requires_new_reference_pass(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp);(p/'binary').write_bytes(b'new cold-only candidate')
            (p/'original-sdk.cpp').write_bytes(b'frozen original SDK source')
            receipt=dict(status='INIT_CANDIDATE_COMPILED_AND_LOADER_VERIFIED_NOT_NATIVE_QUALIFIED',binary_sha256=digest(p/'binary'),cold_history_init='ORIGINAL_BUFFER_RESET_FIRST_REAL_OBSERVATION_V1',ongoing_control_math_unchanged=True,original_R1_math_assertions_pass=True,source_sha='source',original_SDK_sha256=digest(p/'original-sdk.cpp'))
            build=dict(source_sha='source',sdk_source=str(p/'original-sdk.cpp'),ancestor_binary_sha256='old-binary')
            (p/'receipt').write_text(json.dumps(receipt));(p/'build').write_text(json.dumps(build))
            s=dict(native_binary=str(p/'binary'),cold_history_build_receipt=str(p/'receipt'),cold_history_build_spec=str(p/'build'),cold_history_build_receipt_sha256=digest(p/'receipt'),cold_history_build_spec_sha256=digest(p/'build'))
            self.assertEqual(standing_repair_ancestor(s),'old-binary')
            previous=dict(phase='g2-reference-qualification',status='NATIVE_STAGE_PASSED',shutdown_clean=True,owned_process_exits=[dict(exit_code=0)])
            self.assertEqual(phase_limit('g2-standing-hold-short-v2',previous,s),2528)
            with self.assertRaises(PermissionError):phase_limit('g2-standing-hold-short-v2',dict(previous,status='FAILED'),s)
            with self.assertRaises(PermissionError):phase_limit('g2-standing-hold-short-v2',previous,{})
            (p/'receipt').write_text(json.dumps(dict(receipt,ongoing_control_math_unchanged=False)))
            with self.assertRaises(PermissionError):standing_repair_ancestor(s)
            s['cold_history_build_receipt_sha256']=digest(p/'receipt')
            with self.assertRaises(PermissionError):standing_repair_ancestor(s)

if __name__=='__main__':unittest.main()
