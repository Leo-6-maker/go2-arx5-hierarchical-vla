"""Pure bank identity fixtures, not native reader/physics/HF receipts."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from scripts.pi0292.durable import write
from .bank import SCHEMA,closed_child,copy_raw,manifest,raw_file
from .plan import AUTH,R1,candidate,cards,slots


def fixture():
    groups=[candidate(*slot,0) for slot in slots()];rows=[]
    for i,card in enumerate(cards(groups)):
        rows.append(dict(id=card['id'],task=card['task'],scene_group=card['group_id'],
            task_success=True,admitted=True,evidence_qualified=True,native_faults=0,unknown_effects=[],
            R0_PASS_to_R1_FAIL=False,status='PAIRED_FIXED_R0_R1_SAVED_DATA_COMPLETE',transition='FAIL->PASS',
            source_raw_manifest_sha256=hashlib.sha256(str(i).encode()).hexdigest(),
            R1=dict(status='PASS_FIXED_ORIGINAL_V3_0_NO_SILENT_CLIP',actual_statistics_sha256=R1),
            official_reader=dict(R1=dict(statistics_sha256=R1,actual_anchors=[0,80,160,320,480,600],
                actual_dual_camera_frames=True,mask_H40_EEF10_zero_padding=True,deterministic_repeat=True))))
    return groups,rows


class BankTests(unittest.TestCase):
    def make(self,groups,rows):
        return manifest(groups,rows,population_sha256='a'*64,adoption_sha256='b'*64)

    def test_full_denominator_gate_and_explicit_new_R1_not_PV1(self):
        g,r=fixture();v=self.make(g,r)
        self.assertEqual(v['schema'],SCHEMA);self.assertEqual(v['statistics_sha256'],R1)
        self.assertEqual(len(v['episodes']),24);self.assertFalse(v['PV1_bank_changed'])
        self.assertFalse(v['policy_success_claim']);self.assertNotIn('READY',v)
        with self.assertRaises(PermissionError):self.make(g,r[:-1])

    def test_valid_task_failure_kept_in_24_but_not_success_bank(self):
        g,r=fixture();r[0].update(task_success=False,admitted=False)
        v=self.make(g,r)
        self.assertEqual(v['gate']['accounted'],24);self.assertEqual(len(v['episodes']),23)
        self.assertNotIn(r[0]['id'],[x['id'] for x in v['episodes']])
        self.assertEqual(v['all_24_paired_attempts_identity'],self.make(g,r)['all_24_paired_attempts_identity'])

    def test_wrong_statistics_fake_reader_duplicate_raw_and_changed_task_rejected(self):
        for mutation in (
            lambda r:r[0]['R1'].update(actual_statistics_sha256='f'*64),
            lambda r:r[0]['official_reader']['R1'].update(actual_dual_camera_frames=False),
            lambda r:r[0]['official_reader']['R1'].update(actual_anchors=[0]),
            lambda r:r[0].update(source_raw_manifest_sha256=r[1]['source_raw_manifest_sha256']),
            lambda r:r[0].update(task='T2'),
            lambda r:r[0].update(unknown_effects=['UNKNOWN']),
        ):
            g,r=fixture();mutation(r)
            with self.assertRaises(PermissionError):self.make(g,r)

    def test_no_cross_campaign_or_active_export_source(self):
        import os
        with tempfile.TemporaryDirectory() as folder:
            parent=Path(folder);child=parent/'children/export';child.mkdir(parents=True)
            lease=parent/'lease.json';write(lease,dict(status='CLOSED'))
            write(child/'OUTPUT_BINDING.private.json',dict(uid=os.getuid(),host=os.uname().nodename,run_root=str(child),lease=str(lease)))
            write(child/'RUN_MANIFEST.json',dict(role='EXPORT',authority=dict(protocol_release=AUTH)))
            write(child/'RESOURCE_LEDGER.json',dict(status='TERMINAL_EXPORT',unknown_effects=[]))
            self.assertEqual(closed_child(parent,child,'EXPORT','TERMINAL_EXPORT'),child)
            write(lease,dict(status='ACTIVE'))
            with self.assertRaises(PermissionError):closed_child(parent,child,'EXPORT','TERMINAL_EXPORT')
            with self.assertRaises(PermissionError):closed_child(parent.parent,child,'EXPORT','TERMINAL_EXPORT')

    def test_native_0664_raw_copied_to_independent_0400_without_source_chmod(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'source.png';target=Path(folder)/'bank.png'
            source.write_bytes(b'actual fixture bytes');source.chmod(0o664)
            expected=hashlib.sha256(source.read_bytes()).hexdigest()
            copy_raw(source,target,expected)
            self.assertEqual(source.stat().st_mode&0o777,0o664)
            self.assertEqual(target.stat().st_mode&0o777,0o400)
            self.assertNotEqual(source.stat().st_ino,target.stat().st_ino)
            with self.assertRaises(FileExistsError):copy_raw(source,target,expected)
            source.write_bytes(b'changed')
            with self.assertRaises(ValueError):raw_file(source,expected)


if __name__=='__main__':unittest.main()
