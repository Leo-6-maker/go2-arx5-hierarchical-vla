import copy
import unittest

from .mp1.behavior_matrix import assemble,PANELS
from .mp1.plans import evaluation_plan

CHECKS=['raw_manifest_all_sizes_and_hashes','actual_execution_source_not_later_root_manifest','adjudication_result_hash',
    'original_submitted_installed_committed_prefix_labels','original_20Hz_clock_tolerance','independent_saved_state_score_exact',
    'actual_contact_tick_coverage','actual_highrate_clock_and_count','original_S1_no_action_modification',
    'original_S1_live_shadow_parity','shadow_did_not_mutate_live','S1_rejection_accounting','paired_actual_camera_names','actual_RGB_count']


def plan():
    cards=[]
    for split,groups in [('TRAIN',[('F0','000'),('F1','000')]),('DEV',[('F0','000'),('F1','000'),('F1','001'),('F2','000')])]:
        for f,n in groups:
            for t in ['T1','T2']:cards.append(dict(id=f'original-{split}-{f}-{n}-{t}',split=split,scene_group=f'PI0294/{split}/{f}/{n}',task=t))
    return evaluation_plan(cards,'a'*64)


def receipt(card,status='SUCCESS',stage=None):
    rejection=stage=='S1_PRE_EXECUTION_REJECT';steps=600 if rejection else 16000
    return dict(attempt_id=card['id'],model=card['mp1_model_label'],task=card['task'],split=card['split'],scene_group=card['scene_group'],
        status=status,failure_stage=stage or ('COMPLETE_SUCCESS' if status=='SUCCESS' else 'OBJECT_TASK_RESULT'),pipeline_trustworthy=status!='ENGINEERING_INCOMPLETE',
        duration_s=steps*.002,main_integrations=steps,RGB_pairs=steps//25+1,S1_rejections=int(rejection),live_native_failures=0,teacher_takeover=False,result_sha256='e'*64)


def add(a,audits,chains,checkpoints,registries):
    name=a['attempt_id'];panel=a['model'];spec=PANELS[panel];label=panel+'-ALLOCATION-02';ep='episode-'+name
    weight=dict(global_step=spec['step'],checkpoint_manifest_sha256=checkpoints[spec['branch']+'-'+str(spec['step'])],label=label)
    reg=registries.setdefault(label,dict(run_uuid='source-pair-01',checkpoint_identity=weight,cases={}))
    reg['cases'][ep]=dict(attempt_id=name,card=dict(id=name,task=a['task'],split=a['split'],scene_group=a['scene_group']))
    audits[name]=dict(attempt_id=name,status='PASS_INDEPENDENT_ACTUAL_EVIDENCE_AUDIT',adjudication=copy.deepcopy(a),
        checks={k:True for k in CHECKS},errors=[],actual_native_safety_flag_rows=0,score=dict(persistent_success=a['status']=='SUCCESS'),
        video=dict(sha256='f'*64,bytes=100,decoded_frames=a['RGB_pairs'],actual_paired_frames=a['RGB_pairs']))
    executed=a['main_integrations']//200;count=executed+a['S1_rejections']
    chains[name]=dict(status='PASS_COMPLETE_SEALED_CONDITION_FRESH_DUAL_SSH_CHAINS',read_only=True,scientific_inputs_modified=False,new_physics=0,new_model_queries=0,
        core=dict(status='PASS_ACTUAL_FRESH_OBSERVATION_ACTION_EXECUTED_ACK_AND_REJECTED_ZERO_LIVE_PREFIX',episode_uuid=ep,integrations=a['main_integrations'],executed_ACKs=executed,rejected_unexecuted_prefixes=a['S1_rejections'],
            distinct_snapshots=count,raw_camera_hashes_matched=2*count,rows=[dict(sequence=i,query_uuid=name+'-'+str(i)) for i in range(count)]),
        hopper=dict(status='PASS_MATCHING_NATIVE_GENERATIONS_AND_UNMODIFIED_ACTION_RECEIPTS',draws=count,all_arrays_sha256_pass=True,weight_binding=weight,precision_modes=[dict(mode='ORIGINAL_E0_BF16_AUTOCAST_FP32_MATMUL_HIGHEST',bf16_autocast=True,cuda_matmul_allow_tf32=False)]))


class SourceMatrixTests(unittest.TestCase):
    def setUp(self):
        self.plan=plan();self.checkpoints={b+'-'+str(s):str(i)*64 for i,(b,s) in enumerate([(b,s) for b in ['S-BC','S-CORR'] for s in [512,2048]],1)}
        self.audits={};self.chains={};self.registries={}

    def matrix(self,rows,**kwargs):return assemble(self.plan,rows,self.audits,self.chains,self.checkpoints,self.registries,**kwargs)

    def test_all32_preserved_and_no_partial_success_rate(self):
        card=self.plan['panels']['S-BC-2048-DEV8']['cards'][0];a=receipt(card);add(a,self.audits,self.chains,self.checkpoints,self.registries)
        result=self.matrix([a]);self.assertEqual(len(result['cells']),32)
        self.assertEqual(sum(r['status']=='PENDING_OR_IN_PROGRESS' for r in result['cells']),31)
        self.assertIsNone(result['panels']['S-BC-2048-DEV8']['tasks']['T1']['success_fraction_only_if_complete'])
        self.assertEqual(sum(r['status']=='NOT_RUN' for r in self.matrix([a],closed=True)['cells']),31)

    def test_safety_rejection_and_full_safe_object_failure_stay_distinct(self):
        cards=self.plan['panels']['S-BC-2048-TRAIN4']['cards'];rows=[receipt(cards[0],'VALID_TASK_FAIL','S1_PRE_EXECUTION_REJECT'),receipt(cards[1],'VALID_TASK_FAIL')]
        for a in rows:add(a,self.audits,self.chains,self.checkpoints,self.registries)
        result=self.matrix(rows);self.assertEqual([r['failure_stage'] for r in result['actual_attempts']],['S1_PRE_EXECUTION_REJECT','OBJECT_TASK_RESULT'])
        self.assertTrue(all(r['qualified_policy_outcome'] for r in result['actual_attempts']));self.assertFalse(result['actual_safety_stop_present'])

    def test_missing_audit_or_registry_is_a_gap_not_retry_permission(self):
        a=receipt(self.plan['panels']['S-BC-2048-TRAIN4']['cards'][0]);add(a,self.audits,self.chains,self.checkpoints,self.registries)
        self.registries.clear();result=self.matrix([a]);row=result['actual_attempts'][0]
        self.assertFalse(row['qualified_policy_outcome']);self.assertEqual(result['cells'][4]['status'],'UNVERIFIED_POLICY_OUTCOME')
        self.assertFalse(row['missing_audit_is_retry_authority']);self.audits.clear();self.assertFalse(self.matrix([a])['actual_attempts'][0]['raw_audit_verified'])

    def test_swapped_same_model_same_length_chain_rejected_by_actual_registry(self):
        rows=[receipt(c) for c in self.plan['panels']['S-BC-2048-TRAIN4']['cards'][:2]]
        for a in rows:add(a,self.audits,self.chains,self.checkpoints,self.registries)
        x,y=[a['attempt_id'] for a in rows];self.chains[x],self.chains[y]=self.chains[y],self.chains[x]
        with self.assertRaisesRegex(ValueError,'CHAIN_PROGRESS'):self.matrix(rows)

    def test_wrong_checkpoint_or_precision_is_not_comparable(self):
        a=receipt(self.plan['panels']['S-BC-2048-TRAIN4']['cards'][0]);add(a,self.audits,self.chains,self.checkpoints,self.registries)
        original=copy.deepcopy(self.chains[a['attempt_id']]);self.chains[a['attempt_id']]['hopper']['weight_binding']['global_step']=512
        with self.assertRaises(ValueError):self.matrix([a])
        self.chains[a['attempt_id']]=copy.deepcopy(original);self.chains[a['attempt_id']]['hopper']['precision_modes'][0]['cuda_matmul_allow_tf32']=True
        with self.assertRaises(ValueError):self.matrix([a])

    def test_old_model_or_replacement_of_valid_failure_refused(self):
        card=self.plan['panels']['S-BC-2048-TRAIN4']['cards'][0];a=receipt(card,'VALID_TASK_FAIL','S1_PRE_EXECUTION_REJECT');add(a,self.audits,self.chains,self.checkpoints,self.registries)
        replacement=dict(a,attempt_id=a['attempt_id']+'-REPLACEMENT-01')
        with self.assertRaises(PermissionError):self.matrix([a,replacement])
        with self.assertRaises(PermissionError):self.matrix([dict(a,model='M2-BC')])

    def test_one_engineering_replacement_retains_both_original_and_success(self):
        card=self.plan['panels']['S-BC-2048-TRAIN4']['cards'][0];old=receipt(card,'ENGINEERING_INCOMPLETE');new=receipt(dict(card,id=card['id']+'-REPLACEMENT-01'))
        add(new,self.audits,self.chains,self.checkpoints,self.registries);result=self.matrix([old,new]);cell=next(c for c in result['cells'] if c['condition_id']==card['id'])
        self.assertEqual(cell['actual_attempts'],2);self.assertEqual(cell['status'],'SUCCESS');self.assertEqual(len(result['actual_attempts']),2)

    def test_geometry_or_wrong_public_video_not_borrowed(self):
        a=receipt(self.plan['panels']['S-BC-2048-TRAIN4']['cards'][0]);add(a,self.audits,self.chains,self.checkpoints,self.registries)
        with self.assertRaises(PermissionError):self.matrix([a],geometries={a['attempt_id']:dict(geom='table')})
        with self.assertRaises(ValueError):self.matrix([a],public_videos={a['attempt_id']:dict(url='https://example.com/video',sha256='0'*64,bytes=100)})

    def test_all32_verified_completion_does_not_select_a_candidate(self):
        rows=[receipt(c) for p in self.plan['panels'].values() for c in p['cards']]
        for a in rows:add(a,self.audits,self.chains,self.checkpoints,self.registries)
        result=self.matrix(rows);self.assertEqual(result['status'],'COMPLETE_ALL32_VERIFIED_POLICY_RESULTS');self.assertEqual(result['selected'],'NONE')
        self.assertEqual(result['panels']['S-CORR-2048-DEV8']['tasks']['T2']['success_fraction_only_if_complete'],1)

    def test_actual_safety_is_preserved_and_blocks_replacement(self):
        card=self.plan['panels']['S-BC-2048-TRAIN4']['cards'][0];a=receipt(card,'ACTUAL_SAFETY_FAIL','LIVE_NATIVE_SAFETY');a.update(pipeline_trustworthy=False,live_native_failures=1)
        result=self.matrix([a]);self.assertTrue(result['actual_safety_stop_present']);self.assertEqual(result['cells'][4]['status'],'ACTUAL_SAFETY_FAIL')
        a['status']='ENGINEERING_INCOMPLETE';replacement=receipt(dict(card,id=card['id']+'-REPLACEMENT-01'))
        with self.assertRaises(PermissionError):self.matrix([a,replacement])

    def test_false_decode_or_native_safety_claim_is_not_verified(self):
        a=receipt(self.plan['panels']['S-BC-2048-TRAIN4']['cards'][0]);add(a,self.audits,self.chains,self.checkpoints,self.registries)
        self.audits[a['attempt_id']]['video']['decoded_frames']=1
        with self.assertRaises(ValueError):self.matrix([a])
        self.audits[a['attempt_id']]['video']['decoded_frames']=641;self.audits[a['attempt_id']]['actual_native_safety_flag_rows']=1
        with self.assertRaises(ValueError):self.matrix([a])

    def test_reference_gate_without_its_own_chain_proof_is_an_explicit_gap(self):
        a=receipt(self.plan['panels']['S-BC-2048-TRAIN4']['cards'][0],'VALID_TASK_FAIL','ORIGINAL_REFERENCE_GATE_REJECT')
        add(a,self.audits,self.chains,self.checkpoints,self.registries);result=self.matrix([a])
        self.assertFalse(result['actual_attempts'][0]['qualified_policy_outcome']);self.assertEqual(result['actual_attempts'][0]['failure_stage'],'ORIGINAL_REFERENCE_GATE_REJECT')
        self.assertEqual(result['cells'][4]['status'],'UNVERIFIED_POLICY_OUTCOME')

    def test_a_different_DEV_layout_cannot_silently_enter_the_fixed_plan(self):
        panel=self.plan['panels']['S-BC-2048-DEV8'];card=panel['cards'][0];card.update(scene_group='PI0294/DEV/F9/000',id='PI0294-MP1-S-BC-2048-DEV8-F9-000-T1');panel['attempt_ids'][0]=card['id']
        with self.assertRaises(PermissionError):self.matrix([])


if __name__=='__main__':unittest.main()
