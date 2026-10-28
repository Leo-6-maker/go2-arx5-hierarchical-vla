"""Cross-module storage scope routing with pure metadata and temporary files."""
import base64
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.pi0294.dec1 import a40_accounting as accounting
from scripts.pi0294.dec1 import a40_allocation as allocation
from scripts.pi0294.dec1 import a40_future as future
from scripts.pi0294.dec1 import a40_training_context as context
from scripts.pi0294.dec1 import dual_platform as dual
from scripts.pi0294.dec1 import future_budget as budget
from scripts.pi0294.dec1 import storage_expansion as storage
from scripts.pi0294.dec1.test_storage_expansion import settled,CARD,arguments
from scripts.pi0294.dec1.test_lost_save_recovery import recovered
from scripts.pi0294.dec1.test_segmented_training import setup
from scripts.pi0294.dec1.test_a40_allocation import raw,fixture as native_fixture
from scripts.pi0294.dec1.test_a40_training_runtime import put
from scripts.pi0294.pv1 import allocation as legacy_allocation,pbs_owner
from scripts.pi0294.test_pv1_allocation import fixture as legacy_fixture

GIB=1024**3


@contextmanager
def scoped_inputs():
    # Reuse the established four-segment fixture, then advance only its known
    # completed A1 in exactly the same way as the actual metadata preparation.
    with recovered() as (ff,hh,dd,cc):
        _,_,values=setup(ff,hh,dd,cc)
    with settled() as (f,h,d,c,cc):
        h,d=storage.adopt(h,d,**arguments(f,c,cc))
        h['new_storage_current_bytes']=h['new_storage_peak_bytes']=270*GIB
        d['new_GPU_site_output_allocated_upper_bytes']=250*GIB
        r=json.loads(values['request_bytes']);s=json.loads(values['snapshot_bytes'])
        p=json.loads(values['progress_bytes']);plan=json.loads(values['plan_bytes'])
        r.update(expected_native_start_step=2048,submit_count=len(h['PBS_submissions'])+1,
            storage_authorization_release=storage.RELEASE,storage_grant_sha256=storage.GRANT_SHA)
        r['training_block'].update(stop_after_step=4096,resume_checkpoint_relative='training/A_CONTROL/checkpoint-2048')
        p['candidates']['A_CONTROL'].update(optimizer_updates_committed=2048,
            checkpoint_complete_sha256='c'*64,probe_1024_complete=True,probe_receipt_sha256='d'*64)
        plan['training_jobs']=plan['training_jobs'][1:]
        plan['storage_lifecycle']['new_output_peak_upper_bytes']['Hopper_storage_bytes']=352*GIB
        for v in (s,plan):v.update(storage_authorization_release=storage.RELEASE,storage_grant_sha256=storage.GRANT_SHA)
        values.update(request_bytes=raw(r),progress_bytes=raw(p),plan_bytes=raw(plan),storage_bytes=CARD.read_bytes())
        refresh(values,h,d,s)
        with patch.object(budget,'VAP1_ADOPTION',hashlib.sha256(values['amendment_bytes']).hexdigest()):
            yield f,h,d,values


def refresh(values,h,d,s=None):
    s=deepcopy(s) if s is not None else json.loads(values['snapshot_bytes'])
    s['used'].update(h['used'],Hopper_CPU_s=h['CPU_s'],Hopper_storage_bytes=h['new_storage_current_bytes'])
    s['reserved']=dict(h['reserved'],Hopper_CPU_s=h['CPU_reserved_s'],Hopper_storage_bytes=h['storage_reserved_bytes'])
    s['dec1_used'].update(H200_s=d['GPU_allocated_s'],PBS_submissions=len(d['new_submissions']),
        optimizer_updates=d['optimizer_updates'],neural_queries=d['neural_queries'],Hopper_CPU_s=d['CPU_s'],
        Hopper_storage_bytes=d['new_GPU_site_output_allocated_upper_bytes'],episode_starts=d['episode_starts'],
        physics_ticks=d['physics_ticks'],engine_starts=d['constructors'],transfer_bytes=d.get('network_transfer_accounted_bytes',0))
    s['dec1_reserved']=dict(H200_s=d['GPU_reserved_s'],Hopper_CPU_s=d['CPU_reserved_s'],
        optimizer_updates=d['optimizer_updates_reserved'],neural_queries=d['neural_queries_reserved'],
        Hopper_storage_bytes=d['GPU_site_output_reserved_bytes'],transfer_bytes=d.get('network_transfer_reserved_bytes',0))
    s['storage_peaks']['Hopper_storage_bytes']=h['new_storage_peak_bytes']
    s['dec1_storage_peaks']['Hopper_storage_bytes']=d['new_GPU_site_output_allocated_upper_bytes']
    values['ledger_bytes']=raw(h);values['DEC1_ledger_bytes']=raw(d)
    s['actual_Hopper_ledger_sha256']=hashlib.sha256(values['ledger_bytes']).hexdigest()
    s['actual_DEC1_ledger_sha256']=hashlib.sha256(values['DEC1_ledger_bytes']).hexdigest()
    values['snapshot_bytes']=raw(s)


def reserved_inputs(f,h,d,values):
    proof=future.recompute(**values)
    with patch.object(dual,'GRANT_SHA',hashlib.sha256(f['grant_bytes']).hexdigest()):
        v,dd=accounting.reserve(h,d,request_bytes=values['request_bytes'],budget_proof=proof,
            grant_bytes=f['grant_bytes'],campaign_view=f['campaign_view'],now_epoch=f['now'])
    r=json.loads(values['request_bytes'])
    v,dd=accounting.bind_job(v,dd,request_id=r['request_id'],job_id=f['job_id'],now_epoch=f['now'])
    native=deepcopy(values);native.update(purpose='NATIVE_PRETRAIN',job_id=f['job_id'])
    s=json.loads(native['snapshot_bytes']);s['current_native_request_work_id']='TRAIN:A_CONTROL:SEGMENT:4096'
    s['work_reservations']=[dict(work_id=s['current_native_request_work_id'],job_id=f['job_id'],kind='TRAIN',
        submission_already_charged=True,resources=dict(v['reserved'],Hopper_CPU_s=v['CPU_reserved_s'],
        Hopper_storage_bytes=v['storage_reserved_bytes']))]
    refresh(native,v,dd,s)
    return v,dd,native


class StorageScopeIntegrationTests(unittest.TestCase):
    def test_effective_caps_changes_only_storage_and_requires_full_authority_chain(self):
        with scoped_inputs() as (_,_,__,values):
            old=budget.effective_caps(values['amendment_bytes'],values['recovery_bytes'],values['segment_bytes'])
            new=budget.effective_caps(values['amendment_bytes'],values['recovery_bytes'],values['segment_bytes'],values['storage_bytes'])
            self.assertEqual(old['Hopper_storage_bytes'],176*GIB)
            self.assertEqual(new['Hopper_storage_bytes'],352*GIB)
            self.assertEqual({k:v for k,v in old.items() if k!='Hopper_storage_bytes'},
                {k:v for k,v in new.items() if k!='Hopper_storage_bytes'})
            for key in ('segment_bytes','recovery_bytes','amendment_bytes'):
                args={k:values[k] for k in ('amendment_bytes','recovery_bytes','segment_bytes','storage_bytes')};args[key]=None
                with self.subTest(missing=key),self.assertRaises(PermissionError):budget.effective_caps(**args)

    def test_full_remaining_forecast_and_recompute_bind_opt_in_without_relaxing_old_scope(self):
        with scoped_inputs() as (_,h,d,values):
            proof=future.recompute(**values)
            self.assertTrue(proof['forecast']['admissible_forecast'])
            self.assertEqual(proof['forecast']['effective_global_caps']['Hopper_storage_bytes'],419*GIB)
            self.assertEqual(proof['forecast']['effective_DEC1_VAP1_caps']['Hopper_storage_bytes'],352*GIB)
            self.assertEqual(proof['forecast']['future_required_additional']['optimizer_updates'],6144)
            self.assertEqual(proof['forecast']['original_final_protection']['Hopper_storage_bytes'],40*GIB)
            for kind in ('no_bytes','wrong_bytes','legacy_request','snapshot_no_scope','plan_no_scope','one_marker_missing'):
                v=deepcopy(values)
                if kind=='no_bytes':v.pop('storage_bytes')
                elif kind=='wrong_bytes':v['storage_bytes']+=b' '
                elif kind=='legacy_request':
                    r=json.loads(v['request_bytes']);r.pop('storage_authorization_release');r.pop('storage_grant_sha256');v['request_bytes']=raw(r)
                elif kind=='one_marker_missing':
                    dd=deepcopy(d);dd.pop(storage.FIELD);refresh(v,h,dd)
                else:
                    field='snapshot_bytes' if kind=='snapshot_no_scope' else 'plan_bytes'
                    obj=json.loads(v[field]);obj.pop('storage_authorization_release');v[field]=raw(obj)
                with self.subTest(kind=kind),self.assertRaises(PermissionError):future.recompute(**v)
            legacy=deepcopy(values);legacy.pop('storage_bytes')
            for field in ('snapshot_bytes','plan_bytes','request_bytes'):
                obj=json.loads(legacy[field]);obj.pop('storage_authorization_release',None);obj.pop('storage_grant_sha256',None);legacy[field]=raw(obj)
            with self.assertRaises(PermissionError):future.recompute(**legacy)

    def test_reservation_retains_exact70GB_cpu_updates_and_rejects_new_cap_overflow_or_wrong_proof(self):
        with scoped_inputs() as (f,h,d,values):
            proof=future.recompute(**values)
            with patch.object(dual,'GRANT_SHA',hashlib.sha256(f['grant_bytes']).hexdigest()):
                v,dd=accounting.reserve(h,d,request_bytes=values['request_bytes'],budget_proof=proof,
                    grant_bytes=f['grant_bytes'],campaign_view=f['campaign_view'],now_epoch=f['now'])
                self.assertEqual(v['storage_reserved_bytes'],70000000000)
                self.assertEqual(dd['GPU_site_output_reserved_bytes'],70000000000)
                self.assertEqual(v['CPU_reserved_s'],54000);self.assertEqual(v['reserved']['optimizer_updates'],2048)
                self.assertEqual(v['PBS_submissions'][-1]['storage_authorization_release'],storage.RELEASE)
                for kind in ('proof_scope','DEC1_overflow','global_overflow'):
                    hh,ddd=deepcopy(h),deepcopy(d);bad=deepcopy(proof)
                    if kind=='proof_scope':bad.pop('storage_grant_sha256')
                    elif kind=='DEC1_overflow':ddd['new_GPU_site_output_allocated_upper_bytes']=352*GIB-70000000000+1
                    else:hh['new_storage_current_bytes']=419*GIB-40*GIB-70000000000+1
                    bad['global_ledger_canonical_sha256']=accounting.digest(hh)
                    bad['DEC1_ledger_canonical_sha256']=accounting.digest(ddd)
                    with self.subTest(kind=kind),self.assertRaises(PermissionError):
                        accounting.reserve(hh,ddd,request_bytes=values['request_bytes'],budget_proof=bad,
                            grant_bytes=f['grant_bytes'],campaign_view=f['campaign_view'],now_epoch=f['now'])

    def test_own_native_allocation_uses_new_proof_caps_but_still_requires_real_quota(self):
        with scoped_inputs() as (f,h,d,values):
            v,dd,inputs=reserved_inputs(f,h,d,values);proof=future.recompute(**inputs)
            args=native_fixture();r=json.loads(values['request_bytes'])
            args.update(request_bytes=values['request_bytes'],ledger=v,budget_proof=proof)
            args['job']['Resource_List']['walltime']='06:15:00'
            args['quota']['available_bytes']=419*GIB
            with patch.object(allocation,'GRANT_SHA',hashlib.sha256(f['grant_bytes']).hexdigest()),\
                    patch.object(dual,'GRANT_SHA',hashlib.sha256(f['grant_bytes']).hexdigest()):
                result=allocation.validate_native(**args)
                self.assertFalse(result['model_loaded'])
                for kind in ('quota','missing_scope'):
                    bad=deepcopy(args)
                    if kind=='quota':bad['quota']['available_bytes']=70000000000+40*GIB-1
                    else:bad['budget_proof'].pop('storage_grant_sha256')
                    with self.subTest(kind=kind),self.assertRaises(PermissionError):allocation.validate_native(**bad)

    def test_legacy_global_guards_honor_adopted_cap_without_minting_more_gpu_or_accepting_fake_marker(self):
        with scoped_inputs() as (f,h,d,values):
            request=dict(stage='B0_DEV',evaluation_candidate='B0_CORR',walltime_s=7200,
                additional_output_upper_bytes=64*GIB,dependency_afterok=None)
            pbs_owner.budget_request(h,request)
            for kind in ('old_scope','fake_marker','GPU_reserve'):
                bad=deepcopy(h)
                if kind=='old_scope':bad.pop(storage.FIELD)
                elif kind=='fake_marker':bad[storage.FIELD]['grant_sha256']='0'*64
                else:bad['used']['H200_s']=76*3600
                with self.subTest(kind=kind),self.assertRaises(PermissionError):pbs_owner.budget_request(bad,request)
            args=legacy_fixture();now=f['now']+100
            args['now']=now;args['adoption']['run_uuid']=h['campaign_run_uuid']
            args['request'].update(run_uuid=h['campaign_run_uuid'],created_epoch=now-120)
            args['jobs'][args['job_id']]['stime']=now-60
            args['project_proof'].update(run_uuid=h['campaign_run_uuid'],observed_epoch=now-10)
            args['queue_proof']['observed_epoch']=now-130;args['quota']['report_epoch']=now-10
            oldrow=args['ledger']['PBS_submissions'][0]
            args['ledger']=deepcopy(h);args['ledger']['PBS_submissions'].append(oldrow)
            args['ledger']['used']['PBS_submissions']=len(args['ledger']['PBS_submissions'])
            args['ledger']['reserved']={'H200_s':7200};args['ledger']['storage_reserved_bytes']=64*GIB
            self.assertEqual(legacy_allocation.validate_native(**args)['status'],'PASS_PV1_OWN_NATIVE_H200_ALLOCATION')
            args['ledger'].pop(storage.FIELD)
            with self.assertRaises(PermissionError):legacy_allocation.validate_native(**args)

    def test_native_first_update_forwards_storage_bytes_to_actual_recomputation(self):
        with scoped_inputs() as (f,h,d,values):
            v,dd,inputs=reserved_inputs(f,h,d,values);expected=future.recompute(**inputs)
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);directory=root/'pbs/alloc-17';directory.mkdir(parents=True)
                r=json.loads(inputs['request_bytes']);put(directory/'REQUEST.json',r)
                request_path=directory/'REQUEST.json';request_path.chmod(0o600)
                request_path.write_bytes(inputs['request_bytes']);request_path.chmod(0o400)
                names={'plan':'plans/A40_SOURCE_PAIR_REMAINING_WORK.json',
                    'amendment':'plans/VAP1_ADOPTION_AND_DELTA_BUDGET.json',
                    'recovery':'plans/DEC1_LOST_SAVE_RECOVERY_AUTHORIZATION.json',
                    'segment':'plans/DEC1_SEGMENTED_RECOVERY_AUTHORIZATION.json','storage':storage.PLAN}
                for key,name in names.items():put(root/name,json.loads(inputs[key+'_bytes']))
                # Preserve the fixed authorization bytes, rather than another JSON rendering.
                for key in names:
                    path=root/names[key];path.chmod(0o600);path.write_bytes(inputs[key+'_bytes']);path.chmod(0o400)
                put(root/'RUN_MANIFEST.json',dict(execution=dict(sealed_inputs={names['plan']:context.sha(root/names['plan'])})))
                material={k+'_base64':base64.b64encode(inputs[k+'_bytes']).decode() for k in ('ledger','DEC1_ledger','snapshot','progress')}
                put(root/'material.json',material);put(root/'evidence/A_CONTROL-1024/TRAIN_PROBES/RESULT.json',{})
                allocation_proof=dict(job_id=f['job_id'],request_sha256=expected['request_sha256'])
                def guard():
                    ready=directory/'A40_PRETRAIN_READY.json';release=directory/'A40_PRETRAIN_RELEASE.json'
                    if ready.exists() and not release.exists():
                        put(release,dict(observed_epoch=f['now'],ready_sha256=context.sha(ready),
                            material_relative='material.json',material_sha256=context.sha(root/'material.json'),
                            plan_relative=names['plan'],recomputed_budget=expected,job_id=f['job_id'],
                            request_sha256=expected['request_sha256'],submission_nonce=r['submission_nonce'],whole_block_stop_step=4096))
                with patch.object(context,'require_training',return_value=directory),\
                        patch.object(context,'probe_receipt',return_value=dict(records=[])),\
                        patch.object(context.time,'time',return_value=f['now']),\
                        patch.object(future,'recompute',wraps=future.recompute) as actual:
                    result=context.authorize_phase(root,directory,allocation_proof,guard,'SEGMENT_REENTRY',sleep=lambda _:None)
                    self.assertEqual(result['status'],'PASS_A40_SAME_JOB_WHOLE_BLOCK_AUTHORIZATION')
                    self.assertEqual(actual.call_args.kwargs.get('storage_bytes'),CARD.read_bytes())
                    self.assertTrue((directory/'A40_FIRST_UPDATE_ADMISSION.json').is_file())


if __name__=='__main__':unittest.main()
