"""Synthetic CPU native-queue contracts, never a submission or authentication."""
from copy import deepcopy
import hashlib
import json
import unittest
from unittest.mock import patch

from scripts.pi0294.dec1 import dual_platform as d


def fixture():
    budget=dict(additional_GPU_hours=0,additional_PBS_submissions=0,global_GPU_hours=96,DEC1_GPU_hours=48,
        global_PBS_limit=24,DEC1_new_PBS_limit_including_old_attempts=8,global_final_GPU_protected_s=72000,
        original_T0_epoch=1792348196,original_hard_close_epoch=1792952996)
    grant=dict(schema='VAP1_DUAL_PLATFORM_CONCURRENCY_ADOPTION_V1',policy_release=d.RELEASE,
        maximum_running_scientific_GPUs=2,maximum_running_GPUs_by_platform=dict(HOPPER_H200=1,VANDA_A40=1),
        owner='EXISTING_LOCAL_PC_ONLY',original_T0_epoch=1792348196,original_hard_close_epoch=1792952996,budget=budget,
        known_user_confirmed_independent_job=dict(job_id='680209.hopper-m-02',project='CFP04-CF-031',job_name='ock-e1192548-1'))
    raw=json.dumps(grant).encode()
    jobs={p:{} for p in d.PLATFORMS};items=[]
    for platform,job_id,name in [('HOPPER_H200','13.hopper-m-02','pi0294_pv1_13'),('VANDA_A40','14.stdct-mgmt-02','pi0294_dec1_train')]:
        jobs[platform][job_id]=dict(Job_Name=name,project='CFP04-CF-007',Job_Owner='fixture@host',job_state='R',Resource_List=dict(ngpus=1))
        item=dict(job_id=job_id,job_name=name,project='CFP04-CF-007',status='SUBMITTED_PENDING_NATIVE')
        if platform=='VANDA_A40':item['native_platform']='VANDA_A40'
        items.append(item)
    return raw,jobs,items


class DualPlatformTests(unittest.TestCase):
    def observe(self,raw,jobs,items,**kw):
        with patch.object(d,'GRANT_SHA',hashlib.sha256(raw).hexdigest()):
            return d.observe(grant_bytes=raw,site_jobs=jobs,ledger_items=items,user='fixture',
                now_epoch=1000,observed_epochs=kw.get('epochs',{p:1000 for p in d.PLATFORMS}))

    def test_two_platforms_and_external_project_are_visible_separate(self):
        raw,jobs,items=fixture()
        jobs['HOPPER_H200']['680209.hopper-m-02']=dict(Job_Name='ock-e1192548-1',project='CFP04-CF-031',Job_Owner='fixture@host',job_state='R')
        jobs['VANDA_A40']['1472463.stdct-mgmt-02']=dict(Job_Name='other',project='personal-fixture',Job_Owner='fixture@host',job_state='R')
        view=self.observe(raw,jobs,items)
        self.assertEqual(view['status'],'PASS_TWO_PLATFORM_CAMPAIGN_VIEW');self.assertEqual(view['account_owned_job_count'],4)
        self.assertEqual(view['campaign_running_by_platform'],dict(HOPPER_H200=1,VANDA_A40=1))
        self.assertEqual(len(view['user_confirmed_independent_jobs']),1)
        self.assertEqual(len(view['outside_registered_campaign_jobs']),1)
        self.assertFalse(view['account_view_filtered_or_forged_empty'])

    def test_same_platform_duplicate_or_missing_active_handle_is_blocked(self):
        raw,jobs,items=fixture();duplicate=deepcopy(items[0]);duplicate['job_id']='15.hopper-m-02'
        items.append(duplicate);jobs['HOPPER_H200'][duplicate['job_id']]=deepcopy(next(iter(jobs['HOPPER_H200'].values())))
        with self.assertRaises(PermissionError):self.observe(raw,jobs,items)
        raw,jobs,items=fixture();jobs['VANDA_A40'].clear()
        self.assertEqual(self.observe(raw,jobs,items)['status'],'STOP_UNKNOWN_CAMPAIGN_STATE')

    def test_unregistered_campaign_name_identity_change_hold_and_stale_refused(self):
        raw,jobs,items=fixture();jobs['HOPPER_H200']['15.hopper-m-02']=dict(Job_Name='pi0294_pv1_unknown',project='other',Job_Owner='fixture@host',job_state='Q')
        self.assertEqual(self.observe(raw,jobs,items)['status'],'STOP_UNKNOWN_CAMPAIGN_STATE')
        raw,jobs,items=fixture();jobs['HOPPER_H200']['13.hopper-m-02']['project']='changed'
        with self.assertRaises(PermissionError):self.observe(raw,jobs,items)
        raw,jobs,items=fixture();jobs['HOPPER_H200']['13.hopper-m-02']['job_state']='H'
        self.assertEqual(self.observe(raw,jobs,items)['status'],'STOP_UNKNOWN_CAMPAIGN_STATE')
        with self.assertRaises(PermissionError):self.observe(raw,jobs,items,epochs=dict(HOPPER_H200=800,VANDA_A40=1000))

    def test_original_gpu_sum_budget_final_reserve_and_grant_are_not_relaxed(self):
        raw,jobs,items=fixture();view=self.observe(raw,jobs,items)
        with patch.object(d,'GRANT_SHA',hashlib.sha256(raw).hexdigest()):
            d.reserve_check(grant_bytes=raw,view=view,used={},reserved={},extra=dict(H200_s=2*8*3600,PBS_submissions=2))
            with self.assertRaises(PermissionError):
                d.reserve_check(grant_bytes=raw,view=view,used=dict(H200_s=80*3600),reserved={},extra=dict(H200_s=1))
        with self.assertRaises(PermissionError):d.grant(raw)
        bad=json.loads(raw);bad['budget']['global_GPU_hours']=192;changed=json.dumps(bad).encode()
        with patch.object(d,'GRANT_SHA',hashlib.sha256(changed).hexdigest()),self.assertRaises(PermissionError):d.grant(changed)


if __name__=='__main__':unittest.main()
