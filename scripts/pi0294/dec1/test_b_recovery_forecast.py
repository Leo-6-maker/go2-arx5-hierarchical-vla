"""Full remaining scientific envelope under the exact five-change grant."""
from copy import deepcopy
import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch
from . import future_budget as budget,b_heartbeat_recovery as b,segmented_training as seg
from .test_future_budget import fixtures,amendment
from .test_a40_allocation import raw
from .test_b_heartbeat_recovery import scope

ROOT=Path(__file__).resolve().parents[3]
DESIGN=ROOT/'reports/PI0294/design'

def inputs():
    s,p,q=fixtures();now=b.APPROVED_EPOCH+100
    draft=__import__('json').loads((DESIGN/'parallel-science-20261022/B1_FAILED1829_RECOVERY_SCOPE_DRAFT_20261023.json').read_bytes())
    dimensions=draft['conservative_projection_all_dimensions']
    s.update(observed_epoch=now,used={k:v['global_actual_or_conservative_classified_used'] for k,v in dimensions.items()},
        dec1_used={k:v['DEC1_actual_or_conservative_classified_used'] for k,v in dimensions.items()})
    for prefix,vector in (('',s['used']),('dec1_',s['dec1_used'])):
        s[prefix+'storage_peaks']={k:vector[k] for k in ('Hopper_storage_bytes','Core_storage_bytes','local_storage_bytes')}
    r=scope();r.update(arm='B_COVERAGE')
    top={k:v for k,v in r.items() if k not in ('arm','training_lineage_key')}
    for x in (s,p,q):x.update(top)
    s['source_recovery_grant_sha256']=b.zero.recovery.GRANT_SHA
    p.update(observed_epoch=now,source_recovery_authorization_release=b.zero.recovery.RELEASE,
        known_irrecoverable_optimizer_updates=1024,heartbeat_failed_paid_updates=771,failed_lineage_preserved_step=2819,
        B_failed_paid_lost_updates=805,B_failed_lineage_preserved_step=1829)
    for arm,step,key,cp,probe in (('A_CONTROL',4096,b.zero.HR.LINEAGE,'a'*64,'c'*64),
            ('B_COVERAGE',1024,b.LINEAGE,b.SEED_SHA,b.PROBE_SHA)):
        p['candidates'][arm]=dict(optimizer_updates_committed=step,training_lineage_key=key,
            probe_1024_complete=True,checkpoint_complete_sha256=cp,probe_receipt_sha256=probe)
    from .long_training import POLICY_SHA
    q.update(native_platform='VANDA_A40',seconds_per_update_upper=seg.RATE,qualification_load_save_close_upper_s=3600,
        long_allocation_policy_sha256=POLICY_SHA,training_jobs=[],
        heartbeat_remaining_control_upper=dict(Hopper_CPU_s=5520,local_CPU_s=3600,transfer_bytes=805306368))
    for start,stop in ((1024,2048),(2048,4096)):
        q['training_jobs'].append(dict(r,work_id=f'TRAIN:B_COVERAGE:SEGMENT:{stop}',start_step=start,stop_step=stop,
            same_allocation_1024_probes=True,CPU_envelope_profile=seg.CPU_PROFILE,walltime_upper_s=22500,
            resources=dict(H200_s=22560,PBS_submissions=1,optimizer_updates=stop-start,neural_queries=29,
                Hopper_CPU_s=54000,Core_CPU_s=180,local_CPU_s=120,engine_starts=3,
                transfer_bytes=2*b.GIB,Hopper_storage_bytes=70000000000)))
    q['storage_lifecycle']['new_output_peak_upper_bytes']=dict(Hopper_storage_bytes=588617300992,
        Core_storage_bytes=111014838272,local_storage_bytes=8*b.GIB)
    args=dict(snapshot=s,progress_bytes=raw(p),plan=q,now_epoch=now,amendment_bytes=raw(amendment()),
        recovery_bytes=(DESIGN/'VAP1_A40_LOST_SAVE_RECOVERY_AUTHORIZATION_20261022.json').read_bytes(),
        segment_bytes=(DESIGN/'VAP1_A40_SEGMENTED_RECOVERY_AUTHORIZATION_20261022.json').read_bytes(),
        storage_bytes=(DESIGN/'VAP1_STORAGE_ONLY_AUTHORIZATION_20261022.json').read_bytes(),
        heartbeat_bytes=(DESIGN/'parallel-science-20261022/A2_HEARTBEAT_RECOVERY_AUTHORIZATION_20261023.json').read_bytes(),
        zero_update_bytes=(DESIGN/'parallel-science-20261022/ZERO_UPDATE_RESUME_AUTHORIZATION_20261023.json').read_bytes(),
        b_recovery_bytes=(DESIGN/'parallel-science-20261022/B1_HEARTBEAT_RECOVERY_AUTHORIZATION_20261023.json').read_bytes())
    return args,p

def forecast(args):
    with patch.object(budget,'VAP1_ADOPTION',hashlib.sha256(args['amendment_bytes']).hexdigest()):
        return budget.forecast(**args)

class BRecoveryForecastTests(unittest.TestCase):
    def test_complete_remaining_matrix_final_and_all_prior_costs_fit_with_exact_margins(self):
        args,p=inputs();r=forecast(args)
        self.assertTrue(r['admissible_forecast'])
        self.assertEqual(r['future_required_additional']['optimizer_updates'],3072)
        self.assertEqual(r['future_required_additional']['PBS_submissions'],5)
        self.assertEqual(r['future_required_additional']['H200_s'],131700)
        self.assertEqual(r['effective_global_caps']['PBS_submissions'],25)
        self.assertEqual(r['effective_DEC1_VAP1_caps']['PBS_submissions'],14)
        self.assertEqual(r['effective_DEC1_VAP1_caps']['optimizer_updates'],10792)
        self.assertEqual(r['original_final_protection']['H200_s'],72000)
        self.assertEqual(r['effective_DEC1_VAP1_caps']['H200_s']-args['snapshot']['dec1_used']['H200_s']-131700,336)
        self.assertEqual(len(r['mandatory_jobs']),3)

    def test_805_losses_wrong_old_B_lineage_or_unbound_scope_cannot_get_credit(self):
        for mode in ('lost805','old_step','scope','old_lineage','step','missing_probe','grant','time','GPU'):
            a,p=inputs()
            if mode=='lost805':p['B_failed_paid_lost_updates']=0
            elif mode=='old_step':p['B_failed_lineage_preserved_step']=1024
            elif mode=='scope':a['plan'].pop(b.SCOPE_KEYS[0])
            elif mode=='old_lineage':p['candidates']['B_COVERAGE']['training_lineage_key']='B_COVERAGE'
            elif mode=='step':p['candidates']['B_COVERAGE']['optimizer_updates_committed']=1829
            elif mode=='missing_probe':p['candidates']['B_COVERAGE']['probe_1024_complete']=False
            elif mode=='grant':a['b_recovery_bytes']=None
            elif mode=='time':a['now_epoch']=b.CUTOFF
            else:a['snapshot']['dec1_used']['H200_s']+=337
            a['progress_bytes']=raw(p)
            with self.subTest(mode=mode):
                if mode=='GPU':self.assertFalse(forecast(a)['admissible_forecast'])
                else:
                    with self.assertRaises(PermissionError):forecast(a)

    def test_legacy_start1024_cannot_use_new_phase_without_exact_request(self):
        events=[];never=lambda *a,**kw:self.fail('NO_NEW_PROBE_OR_MODEL_LOAD')
        kw=dict(start_step=1024,stop_step=2048,resume_checkpoint='checkpoint-1024',checkpoint='checkpoint-1024',
            train=lambda **x:events.append(x) or dict(global_step=x['stop']),probe=never,unload=never,load=never,guard=never)
        with self.assertRaises(PermissionError):seg.execute(**kw)
        r=scope();r.update(expected_native_start_step=1024,segment_stop_step=2048,CPU_envelope_profile=seg.CPU_PROFILE)
        result,_=seg.execute(**kw,recovery_request=r)
        self.assertEqual(result['global_step'],2048)
        self.assertEqual(events,[dict(stop=2048,phase='B_RECOVERY_REENTRY',checkpoint='checkpoint-1024')])

if __name__=='__main__':unittest.main()
