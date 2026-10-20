"""Tiny closed-sibling metadata fixtures, not a real data-bank qualification."""
from copy import deepcopy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.pi0294.pv1 import bank_reuse as reuse
from scripts.pi0294.pv1.decision import AUTHORITY
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.test_mp1_transport_fixture import put


class BankReuse(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.parent=Path(tmp.name)/'campaign';self.root=self.parent/'children/new';self.source=self.parent/'children/old'
        self.target=self.source/'datasets/FROZEN_TRAIN_BANK';self.target.mkdir(parents=True)
        put(self.target/'DATASET_MANIFEST.json',dict(CPU_fixture=True));put(self.target/'statistics.json',dict(CPU_fixture=True))
        put(self.target/'READY.json',dict(CPU_fixture=True));self.stats=sha(self.target/'statistics.json')
        self.stage=dict(new_bank_manifest_sha256=sha(self.target/'DATASET_MANIFEST.json'),new_bank_ready_sha256=sha(self.target/'READY.json'))
        self.delivery=dict(revision='b'*40);put(self.source/'plans/PV1_HF_BANK_DELIVERY.json',self.delivery)
        put(self.parent/'PV1_ADOPTION_AND_RESOURCE_BINDING.json',dict(run_uuid='CAMPAIGN',authority_release=AUTHORITY))
        self.lease=self.parent/'lease.json';put(self.lease,dict(status='CLOSED',run_uuid='old',authority_release=AUTHORITY,all_registered_workers_terminal=True))
        binding=dict(run_uuid='old',run_root=str(self.source),uid=os.getuid(),lease=str(self.lease))
        put(self.source/'OUTPUT_BINDING.private.json',binding)
        self.m=dict(run_uuid='old',authority=dict(protocol_ref=AUTHORITY),execution=dict(output_binding_sha256=sha(self.source/'OUTPUT_BINDING.private.json')))
        put(self.source/'RUN_MANIFEST.json',self.m)
        self.proof=dict(status='PASS_ACTUAL_FULL_NEW_BANK_CONSUMPTION',full_member_hashes_verified=True,
            dataset_manifest_sha256=self.stage['new_bank_manifest_sha256'],download_revision=self.delivery['revision'],
            statistics_sha256=self.stats,delivery_binding_sha256=sha(self.source/'plans/PV1_HF_BANK_DELIVERY.json'))
        self.receipt=self.source/'evidence/NEW_BANK_CONSUMPTION_READY.private.json';put(self.receipt,self.proof,True)
        self.ledger=dict(campaign_run_uuid='CAMPAIGN',authority_release=AUTHORITY,
            PBS_submissions=[dict(run_uuid='old',status='NATIVE_TERMINAL')],unknown_effects=[],pending_updates=[])
        put(self.parent/'HOPPER_RESOURCE_LEDGER.json',self.ledger)
        self.plan=dict(schema='PI0294_PV1_CLOSED_SIBLING_BANK_REUSE_V1',authority_release=AUTHORITY,
            campaign_run_uuid='CAMPAIGN',source_child_root=str(self.source),
            source_manifest_sha256=sha(self.source/'RUN_MANIFEST.json'),source_consumption_receipt_sha256=sha(self.receipt),
            later_data_hot_injected=False,read_only_shared_input=True)
        self.bind()

    def bind(self):
        put(self.root/reuse.PLAN,self.plan,True)
        put(self.root/'RUN_MANIFEST.json',dict(execution=dict(sealed_inputs={reuse.PLAN:sha(self.root/reuse.PLAN)})))

    def validate(self):
        with patch.object(reuse,'STATS',self.stats):
            return reuse.validate(self.root,self.target,self.delivery,self.stage)

    def test_only_exact_closed_sibling_receipt_admitted_and_current_full_hashes_still_required(self):
        value=self.validate()
        self.assertTrue(value['prior_consumption_does_not_replace_current_hash_verification'])
        self.assertEqual(value['new_download_bytes'],0)
        self.assertEqual(value['source_child_uuid'],'old')

    def test_open_lease_active_job_unknown_effect_or_changed_fixed_identity_rejected(self):
        for kind in ('lease','job','unknown','pending','manifest','ready','delivery','proof'):
            with self.subTest(kind=kind):
                files={p:p.read_bytes() for p in (self.lease,self.parent/'HOPPER_RESOURCE_LEDGER.json',
                    self.source/'RUN_MANIFEST.json',self.target/'READY.json',self.source/'plans/PV1_HF_BANK_DELIVERY.json',self.receipt)}
                if kind=='lease':put(self.lease,dict(status='ACTIVE',run_uuid='old',all_registered_workers_terminal=False))
                elif kind in ('job','unknown','pending'):
                    v=deepcopy(self.ledger)
                    if kind=='job':v['PBS_submissions'][0]['status']='SUBMITTED_PENDING_NATIVE'
                    elif kind=='unknown':v['unknown_effects']=['unknown']
                    else:v['pending_updates']=['unknown']
                    put(self.parent/'HOPPER_RESOURCE_LEDGER.json',v)
                elif kind=='manifest':put(self.source/'RUN_MANIFEST.json',dict(WRONG=True))
                elif kind=='ready':put(self.target/'READY.json',dict(WRONG=True))
                elif kind=='delivery':put(self.source/'plans/PV1_HF_BANK_DELIVERY.json',dict(revision='c'*40))
                else:put(self.receipt,dict(self.proof,full_member_hashes_verified=False),True)
                with self.assertRaises(PermissionError):self.validate()
                for p,raw in files.items():p.chmod(0o600);p.write_bytes(raw)
                self.receipt.chmod(0o400)

    def test_unsealed_plan_path_escape_symlink_and_live_process_are_not_a_cache(self):
        self.plan['source_child_root']=str(self.parent/'outside');self.bind()
        with self.assertRaises(PermissionError):self.validate()
        self.plan['source_child_root']=str(self.source);self.bind()
        self.plan['later_data_hot_injected']=True;self.bind()
        with self.assertRaises(PermissionError):self.validate()
        self.plan['later_data_hot_injected']=False;self.bind()
        (self.root/reuse.PLAN).chmod(0o600)
        with self.assertRaises(PermissionError):self.validate()
        (self.root/reuse.PLAN).chmod(0o400)
        put(self.source/'worker_processes/live.json',dict(pid=os.getpid(),proc_start=Path('/proc/self/stat').read_text().split()[21]))
        with self.assertRaisesRegex(PermissionError,'STILL_ACTIVE'):self.validate()

    def cold_fixture(self):
        root=self.root;bank=root/'datasets/FROZEN_TRAIN_BANK'
        put(bank/'DATASET_MANIFEST.json',dict(episodes=[dict(id='parent',file='numeric/parent.npz')]))
        put(bank/'PAYLOAD_MANIFEST.json',[])
        put(bank/'statistics.json',dict(CPU_fixture=True))
        put(bank/'READY.json',dict(payload_manifest_sha256=sha(bank/'PAYLOAD_MANIFEST.json')))
        put(bank/'numeric/parent.npz',dict(CPU_fixture=True));(bank/'raw/parent').mkdir(parents=True)
        raw=root/'plans/raw.json';put(raw,dict(parent=str(bank/'raw/parent')),True)
        put(root/'OUTPUT_BINDING.private.json',dict(CPU_fixture=True),True)
        m=dict(execution=dict(sealed_inputs={'plans/raw.json':sha(raw)}))
        put(root/'RUN_MANIFEST.json',m,True)
        return bank,m,dict(new_dataset=str(bank),new_raw_bindings=str(raw)),dict(
            new_bank_manifest_sha256=sha(bank/'DATASET_MANIFEST.json'),new_bank_ready_sha256=sha(bank/'READY.json'))

    def test_prequeue_rejects_metadata_only_sibling_and_stale_rgb_bindings(self):
        bank,m,inputs,stage=self.cold_fixture()
        with patch.object(reuse,'STATS',sha(bank/'statistics.json')):
            self.assertTrue(reuse.preflight(self.root,m,inputs,stage)['full_member_hashes_still_required_inside_allocation'])
            raw=(bank/'PAYLOAD_MANIFEST.json').read_bytes();(bank/'PAYLOAD_MANIFEST.json').unlink()
            with self.assertRaisesRegex(PermissionError,'COMPLETE_BANK_METADATA_REQUIRED'):
                reuse.preflight(self.root,m,inputs,stage)
            (bank/'PAYLOAD_MANIFEST.json').write_bytes(raw)
            put(Path(inputs['new_raw_bindings']),dict(parent=str(self.target/'raw/parent')),True)
            with self.assertRaisesRegex(PermissionError,'ACTUAL_BANK'):
                reuse.preflight(self.root,m,inputs,stage)

    def test_prequeue_rejects_mutable_final_fixture_and_changed_payload(self):
        bank,m,inputs,stage=self.cold_fixture()
        with patch.object(reuse,'STATS',sha(bank/'statistics.json')):
            (self.root/'plans/raw.json').chmod(0o644)
            with self.assertRaisesRegex(PermissionError,'IMMUTABLE_FILE'):
                reuse.preflight(self.root,m,inputs,stage)
            (self.root/'plans/raw.json').chmod(0o400)
            put(bank/'PAYLOAD_MANIFEST.json',[dict(changed=True)])
            with self.assertRaisesRegex(PermissionError,'METADATA_HASHES'):
                reuse.preflight(self.root,m,inputs,stage)
