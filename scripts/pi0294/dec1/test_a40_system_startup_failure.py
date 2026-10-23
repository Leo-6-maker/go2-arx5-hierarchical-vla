import copy,inspect,unittest
from . import a40_system_startup_failure as f

def actual_metadata_fixture():
    now=1792745411.0643072
    j=dict(job_state='F',Exit_status='-3',run_count=21,Hold_Types='s',Rerunable='False',
        Job_Owner='e1192548@vanda',project='CFP04-CF-007',Job_Name='pi0294_dec1_a40_train',
        Variable_List=dict(PV1_REQUEST_ID=f.REQUEST,PV1_SUBMISSION_NONCE='886b1d63-2119-4ee0-85b2-0d26d3482c9d'),
        comment='job held, too many failed attempts to run and terminated',ctime='Fri Oct 23 16:10:40 2026',
        history_timestamp=1792743464,Resource_List=dict(ngpus=1,ncpus=36))
    h=dict(source=f.SOURCE,all_sealed_SHA_match=True,sealed_members=939,owned_unique_inode_output_upper_bytes=11669504,observed_epoch=now)
    g=dict(source=f.SOURCE,all_sealed_SHA_match=True,sealed_members=972,owned_unique_inode_output_upper_bytes=30242426,observed_epoch=now,
        request_sha256=f.REQUEST_SHA,process_final_absent=True,all_model_and_effect_intents_absent=True,
        recorded_process_terminal=False,scientific_completion=False,raw_native=j,entire_single_request_lifetime_s=424,
        conservative_all_attempts_GPU_s=484,conservative_all_allocated_CPU_s=15384)
    return dict(status='INDEPENDENT_NATIVE_STARTUP_F21_NO_MODEL_EFFECTS_ALL_INPUTS_AND_ENTIRE_LIFETIME_COST_BOUNDS',
        observed_epoch=now,rows=dict(vanda=g,hopper=h),combined_CPU_GPU_roots_actual_output_upper_bytes=41911930,
        already_prepaid_small_output_upper_bytes=67108864,additional_output_charge_needed_bytes=0)

class StartupFailure(unittest.TestCase):
    def test_all_attempts_whole_lifetime_upper_not_zero_or_AMS_pending_units(self):
        v=actual_metadata_fixture();n,e=f.proof(v,now_epoch=v['observed_epoch'])
        self.assertEqual((n['conservative_GPU_s'],n['conservative_CPU_s']),(484,15384))
        self.assertFalse(n['recorded_process_terminal']);self.assertFalse(n['scientific_completion']);self.assertEqual(n['schema'],f.SCHEMA)
        self.assertEqual((e['start_native_step'],e['last_native_step'],e['optimizer_updates_committed'],e['query_attempts'],e['model_load_attempts']),(2048,2048,0,0,0))
    def test_native_identity_F_minus3_and_all21_attempts_never_relabelled(self):
        changes=dict(job_state='H',Exit_status=0,run_count=1,Hold_Types='n',Job_Owner='other@vanda',project='OTHER',Job_Name='other',comment='normal successful completion')
        for key,value in changes.items():
            with self.subTest(key=key):
                v=actual_metadata_fixture();v['rows']['vanda']['raw_native'][key]=value
                with self.assertRaises(PermissionError):f.proof(v,now_epoch=v['observed_epoch'])
    def test_input_effect_proof_and_no_fabricated_process_final(self):
        for key,value in dict(source='f'*40,all_sealed_SHA_match=False,sealed_members=971,process_final_absent=False,all_model_and_effect_intents_absent=False,recorded_process_terminal=True,scientific_completion=True,conservative_all_attempts_GPU_s=0,conservative_all_allocated_CPU_s=0).items():
            with self.subTest(key=key):
                v=actual_metadata_fixture();v['rows']['vanda'][key]=value
                with self.assertRaises(PermissionError):f.proof(v,now_epoch=v['observed_epoch'])
    def test_no_stale_clock_unpriced_output_or_unknown_CPU_profile(self):
        v=actual_metadata_fixture()
        with self.assertRaises(PermissionError):f.proof(v,now_epoch=v['observed_epoch']+121)
        with self.assertRaises(PermissionError):f.proof(v,now_epoch=float('nan'))
        v['rows']['vanda']['raw_native']['Resource_List']['ncpus']=72
        with self.assertRaises(PermissionError):f.proof(v,now_epoch=v['observed_epoch'])
        v=actual_metadata_fixture();v['additional_output_charge_needed_bytes']=1
        with self.assertRaises(PermissionError):f.proof(v,now_epoch=v['observed_epoch'])
        v=actual_metadata_fixture();v['rows']['hopper']['owned_unique_inode_output_upper_bytes']=64*1024**2
        with self.assertRaises(PermissionError):f.proof(v,now_epoch=v['observed_epoch'])
    def test_original_scientific_process_guard_stays_intact(self):
        before=inspect.getsource(f.original.settle);adapted=f.administrative_arithmetic()
        self.assertEqual(inspect.getsource(f.original.settle),before);self.assertIn('recorded_process_terminal=True',before)
        self.assertNotEqual(adapted.__name__,f.original.settle.__name__)
    def test_missing_exact_prepaid_or_bound_request_rejects_before_arithmetic(self):
        v=actual_metadata_fixture()
        with self.assertRaises(PermissionError):f.settle({}, {},evidence=v,now_epoch=v['observed_epoch'])
if __name__=='__main__':unittest.main()
