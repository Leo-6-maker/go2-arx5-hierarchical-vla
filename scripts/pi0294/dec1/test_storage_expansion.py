"""Pure storage authorization fixtures; no native mutation or checkpoint IO."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import storage_expansion as storage
from . import segmented_training as seg
from . import lost_save_recovery as recovery
from .test_lost_save_recovery import recovered
from .test_segmented_training import setup
from .test_a40_allocation import raw

CARD=Path(__file__).resolve().parents[3]/'reports/PI0294/design/VAP1_STORAGE_ONLY_AUTHORIZATION_20261022.json'
SEGMENT_CARD=CARD.parent/'VAP1_A40_SEGMENTED_RECOVERY_AUTHORIZATION_20261022.json'


def checksum(data):return hashlib.sha256(data).hexdigest()


@contextmanager
def settled():
    with recovered() as (f,h,d,c):
        h,d,_=setup(f,h,d,c)
        h['optimizer_updates']=h['used']['optimizer_updates']=d['optimizer_updates']=3072
        h['training_runs'][recovery.LINEAGE]=dict(last_native_step=2048,recipe_sha256=recovery.RECIPE)
        row=dict(request_id=storage.REQUEST,job_id=storage.JOB,status='NATIVE_TERMINAL',
            committed_optimizer_updates=2016,reserved_optimizer_updates=0,reserved_neural_queries=0)
        h['PBS_submissions'].append(deepcopy(row));h['used']['PBS_submissions']=len(h['PBS_submissions'])
        d['new_submissions'].append(dict(request_id=storage.REQUEST,job_id=storage.JOB,status='NATIVE_TERMINAL'))
        for prior in d['new_submissions']:prior['status']='NATIVE_TERMINAL'
        closure=dict(status='CLOSED_SEGMENT_FULL_STATE_PRESERVED',job_id=storage.JOB,
            request_id=storage.REQUEST,global_step=2048,recoverable_full_state=True,retired_or_deleted_files=0,
            closed_epoch=f['now']-2,
            effects=dict(arm='A_CONTROL',start_native_step=32,last_native_step=2048,
                optimizer_updates_committed=2016,pending_updates=[],unknown_query_intents=[],
                recipe_sha256=recovery.RECIPE,effect_journal_complete=True),
            settlement=dict(unknown_effects=[],updates=3072))
        closure_bytes=raw(closure)
        control=dict(status='CLOSED_CPU_SEGMENT2048_CONTROL_AFTER_KNOWN_EFFECTS_SETTLEMENT',
            job_id=storage.JOB,unknown_effects=[],new_A_recovery_step=2048,old_A1024_costs_preserved=True,
            closed_epoch=f['now']-1,settled_submission=deepcopy(row),
            GPU_close=dict(lease_status='CLOSED',terminal_receipt_sha256=checksum(closure_bytes),terminal_receipt=closure))
        control_bytes=raw(control)
        with patch.object(storage,'CLOSURE_SHA',checksum(closure_bytes)),\
                patch.object(storage,'CONTROL_CLOSURE_SHA',checksum(control_bytes)),\
                patch.object(seg,'CLOSURE_SHA',h[seg.FIELD]['closed_native_receipt_sha256']):
            yield f,h,d,closure_bytes,control_bytes


def arguments(f,c,cc):
    return dict(grant_bytes=CARD.read_bytes(),closure_bytes=c,control_closure_bytes=cc,now_epoch=f['now'])


class StorageExpansionTests(unittest.TestCase):
    def test_fixed_grant_and_only_explicit_segment_get_new_caps(self):
        self.assertEqual(storage.grant(CARD.read_bytes())['storage_cap_bytes']['DEC1'][1],352*storage.GIB)
        with self.assertRaises(PermissionError):storage.grant(CARD.read_bytes()+b' ')
        self.assertEqual(storage.caps({}),(250*storage.GIB,160*storage.GIB))
        row=dict(segmented_authorization_release=seg.RELEASE,segmented_grant_sha256=seg.GRANT_SHA)
        self.assertEqual(storage.caps(row),(250*storage.GIB,176*storage.GIB))
        row.update(storage_authorization_release=storage.RELEASE,storage_grant_sha256=storage.GRANT_SHA)
        self.assertTrue(storage.enabled(row));self.assertEqual(storage.caps(row),(419*storage.GIB,352*storage.GIB))
        for key,value in (('storage_authorization_release',None),('storage_grant_sha256','0'*64),
                ('segmented_authorization_release',None),('segmented_grant_sha256','0'*64)):
            bad=deepcopy(row);bad[key]=value
            with self.subTest(key=key),self.assertRaises(PermissionError):storage.caps(bad)
        with self.assertRaises(PermissionError):storage.enabled(dict(storage_grant_sha256=storage.GRANT_SHA))

    def test_adoption_only_adds_identical_marker_preserves_every_old_field_and_not_repeated(self):
        with settled() as (f,h,d,c,cc):
            before,dbefore=deepcopy(h),deepcopy(d)
            v,dd=storage.adopt(h,d,**arguments(f,c,cc))
            self.assertEqual(h,before);self.assertEqual(d,dbefore)
            self.assertEqual({k:x for k,x in v.items() if k!=storage.FIELD},h)
            self.assertEqual({k:x for k,x in dd.items() if k!=storage.FIELD},d)
            storage.validate_adopted(v,dd)
            self.assertEqual(storage.global_cap_from_ledger(h),250*storage.GIB)
            self.assertEqual(storage.global_cap_from_ledger(v),419*storage.GIB)
            with self.assertRaises(PermissionError):storage.adopt(v,dd,**arguments(f,c,cc))
            # Adoption is at2048, but legitimate later training does not revoke it.
            v['training_runs'][recovery.LINEAGE]['last_native_step']=4096
            v['optimizer_updates']=v['used']['optimizer_updates']=dd['optimizer_updates']=5120
            storage.validate_adopted(v,dd)
            self.assertEqual(storage.global_cap_from_ledger(v),419*storage.GIB)

    def test_unclosed_unsettled_or_changed_original_trajectory_rejected_without_mutation(self):
        with settled() as (f,h,d,c,cc):
            for kind in ('unknown_h','unknown_d','pending','reserved_h','reserved_cpu','reserved_storage',
                    'reserved_d','reserved_nested','old_A_changed','A_not2048','B_started','counter',
                    'active_global_row','active_DEC1_row','missing_A1','wrong_control_row',
                    'wrong_segment_marker','before_close','job_cutoff','nan_clock','bad_GPU_bytes','bad_control_bytes'):
                v,dd=deepcopy(h),deepcopy(d);kw=arguments(f,c,cc)
                if kind=='unknown_h':v['unknown_effects']=['unknown']
                elif kind=='unknown_d':dd['unknown_effects']=['unknown']
                elif kind=='pending':v['pending_updates']=[{'pending':True}]
                elif kind=='reserved_h':v['reserved']['H200_s']=1
                elif kind=='reserved_cpu':v['CPU_reserved_s']=1
                elif kind=='reserved_storage':v['storage_reserved_bytes']=1
                elif kind=='reserved_d':dd['GPU_site_output_reserved_bytes']=1
                elif kind=='reserved_nested':dd['future_reserved']={'physics_ticks':1}
                elif kind=='old_A_changed':v['training_runs']['A_CONTROL']['last_native_step']=0
                elif kind=='A_not2048':v['training_runs'][recovery.LINEAGE]['last_native_step']=2047
                elif kind=='B_started':v['training_runs']['B_COVERAGE']=dict(last_native_step=1,recipe_sha256=recovery.MATCHED_RECIPES['B_COVERAGE'])
                elif kind=='counter':dd['optimizer_updates']=3071
                elif kind=='active_global_row':v['PBS_submissions'][0]['status']='SUBMITTED_PENDING_NATIVE'
                elif kind=='active_DEC1_row':dd['new_submissions'][0]['status']='SUBMITTED_PENDING_NATIVE'
                elif kind=='missing_A1':v['PBS_submissions'].pop()
                elif kind=='wrong_control_row':v['PBS_submissions'][-1]['committed_optimizer_updates']=2015
                elif kind=='wrong_segment_marker':dd[seg.FIELD]['grant_sha256']='0'*64
                elif kind=='before_close':kw['now_epoch']=f['now']-3
                elif kind=='job_cutoff':kw['now_epoch']=storage.T0+144*3600
                elif kind=='nan_clock':kw['now_epoch']=float('nan')
                elif kind=='bad_GPU_bytes':kw['closure_bytes']+=b' '
                else:kw['control_closure_bytes']+=b' '
                vb,db=deepcopy(v),deepcopy(dd)
                with self.subTest(kind=kind),self.assertRaises(PermissionError):storage.adopt(v,dd,**kw)
                self.assertEqual(v,vb);self.assertEqual(dd,db)

    def test_closure_semantics_are_required_even_with_test_patched_matching_hashes(self):
        with settled() as (f,h,d,c,cc):
            for kind in ('GPU_not_closed','wrong_step','no_full_state','deleted_payload','pending_native',
                    'unknown_settlement','control_not_closed','active_lease','control_wrong_receipt'):
                gpu,control=json.loads(c),json.loads(cc)
                if kind=='GPU_not_closed':gpu['status']='ACTIVE'
                elif kind=='wrong_step':gpu['global_step']=2047
                elif kind=='no_full_state':gpu['recoverable_full_state']=False
                elif kind=='deleted_payload':gpu['retired_or_deleted_files']=1
                elif kind=='pending_native':gpu['effects']['pending_updates']=[{'unknown':True}]
                elif kind=='unknown_settlement':gpu['settlement']['unknown_effects']=[{'unknown':True}]
                elif kind=='control_not_closed':control['status']='ACTIVE'
                elif kind=='active_lease':control['GPU_close']['lease_status']='ACTIVE'
                new_c=raw(gpu);control['GPU_close']['terminal_receipt_sha256']=checksum(new_c)
                control['GPU_close']['terminal_receipt']=deepcopy(gpu)
                if kind=='control_wrong_receipt':control['GPU_close']['terminal_receipt']['global_step']=1024
                new_cc=raw(control)
                with self.subTest(kind=kind),patch.object(storage,'CLOSURE_SHA',checksum(new_c)),\
                        patch.object(storage,'CONTROL_CLOSURE_SHA',checksum(new_cc)),self.assertRaises(PermissionError):
                    storage.adopt(h,d,**arguments(f,new_c,new_cc))

    def test_new_marker_or_original_marker_and_campaign_tampering_never_mints_global_cap(self):
        with settled() as (f,h,d,c,cc):
            v,dd=storage.adopt(h,d,**arguments(f,c,cc))
            for kind in ('release','grant','closure','control_closure','T0','nan_epoch','epoch_cutoff',
                    'extra_cap','unknown_marker','segment_grant','segment_shape','recovery_grant',
                    'recovery_shape','campaign','authority','platform','old_loss'):
                bad=deepcopy(v);marker=bad[storage.FIELD]
                if kind=='release':marker['authority_release']='0'*40
                elif kind=='grant':marker['grant_sha256']='0'*64
                elif kind=='closure':marker['closed_native_receipt_sha256']='0'*64
                elif kind=='control_closure':marker['closed_control_receipt_sha256']='0'*64
                elif kind=='T0':marker['original_T0_epoch']+=1
                elif kind=='nan_epoch':marker['adopted_epoch']=float('nan')
                elif kind=='epoch_cutoff':marker['adopted_epoch']=storage.T0+144*3600
                elif kind=='extra_cap':marker['global_cap']=9999*storage.GIB
                elif kind=='unknown_marker':bad[storage.FIELD]=None
                elif kind=='segment_grant':bad[seg.FIELD]['grant_sha256']='0'*64
                elif kind=='segment_shape':bad[seg.FIELD]['cap']=9999*storage.GIB
                elif kind=='recovery_grant':bad[recovery.FIELD]['grant_sha256']='0'*64
                elif kind=='recovery_shape':bad[recovery.FIELD]['cap']=9999*storage.GIB
                elif kind=='campaign':bad['campaign_run_uuid']='other'
                elif kind=='authority':bad['authority_release']='0'*40
                elif kind=='platform':bad['DEC1_source_pair_platform']='OTHER'
                else:bad['training_runs']['A_CONTROL']['last_native_step']=0
                with self.subTest(kind=kind),self.assertRaises(PermissionError):storage.global_cap_from_ledger(bad)
            mismatch=deepcopy(dd);mismatch[storage.FIELD]['adopted_epoch']+=1
            with self.assertRaises(PermissionError):storage.validate_adopted(v,mismatch)

    def test_native_admitted_exact_readonly_sealed_grant_and_both_original_authorities(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'campaign/children/run';root.mkdir(parents=True)
            manifest=dict(authority=dict(segmented_authorization_release=seg.RELEASE,
                storage_authorization_release=storage.RELEASE,storage_grant_sha256=storage.GRANT_SHA),
                execution=dict(sealed_inputs={seg.PLAN:seg.GRANT_SHA,storage.PLAN:storage.GRANT_SHA}))
            request=dict(segmented_authorization_release=seg.RELEASE,segmented_grant_sha256=seg.GRANT_SHA,
                storage_authorization_release=storage.RELEASE,storage_grant_sha256=storage.GRANT_SHA,
                expected_native_start_step=2048,CPU_envelope_profile=seg.CPU_PROFILE,
                training_block=dict(arm='A_CONTROL',stop_after_step=4096),
                sealed_inputs={seg.PLAN:seg.GRANT_SHA,storage.PLAN:storage.GRANT_SHA})
            for name,card in ((seg.PLAN,SEGMENT_CARD),(storage.PLAN,CARD)):
                local=root/name;local.parent.mkdir(parents=True,exist_ok=True);local.write_bytes(card.read_bytes());local.chmod(0o400)
                native=root.parent.parent/'authority/DEC1'/Path(name).name
                native.parent.mkdir(parents=True,exist_ok=True);native.write_bytes(card.read_bytes());native.chmod(0o400)
            self.assertTrue(storage.admitted(root,manifest,request))
            path=root/storage.PLAN;native=root.parent.parent/'authority/DEC1'/Path(storage.PLAN).name
            for kind in ('writable','local_symlink','modified_bytes','native_writable','native_symlink',
                    'native_mismatch','missing_manifest_release','missing_manifest_hash','missing_manifest_seal','missing_request_seal'):
                m,r=deepcopy(manifest),deepcopy(request)
                if kind=='writable':path.chmod(0o600)
                elif kind=='local_symlink':path.unlink();path.symlink_to(native)
                elif kind=='modified_bytes':path.chmod(0o600);path.write_bytes(CARD.read_bytes()+b' ');path.chmod(0o400)
                elif kind=='native_writable':native.chmod(0o600)
                elif kind=='native_symlink':native.unlink();native.symlink_to(path)
                elif kind=='native_mismatch':native.chmod(0o600);native.write_bytes(CARD.read_bytes()+b' ');native.chmod(0o400)
                elif kind=='missing_manifest_release':m['authority'].pop('storage_authorization_release')
                elif kind=='missing_manifest_hash':m['authority'].pop('storage_grant_sha256')
                elif kind=='missing_manifest_seal':m['execution']['sealed_inputs'].pop(storage.PLAN)
                else:r['sealed_inputs'].pop(storage.PLAN)
                with self.subTest(kind=kind),self.assertRaises(PermissionError):storage.admitted(root,m,r)
                for p in (path,native):
                    if p.is_symlink():p.unlink()
                    if p.exists():p.chmod(0o600)
                    p.write_bytes(CARD.read_bytes());p.chmod(0o400)
            self.assertFalse(storage.admitted(root,{},{}))


if __name__=='__main__':unittest.main()
