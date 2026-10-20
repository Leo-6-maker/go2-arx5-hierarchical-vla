"""CPU fixture banks, not native episode, batch or HF acceptance receipts."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.pi0294.data_cov1.bank import manifest,validate_manifest,parent_files,bank_members,public_json,assembly_plan
from scripts.pi0294.scene_cards import digest
from scripts.pi0294.test_data_cov1_reports import fixtures
from scripts.pi0294.fd1 import sha


class DataCov1Bank(unittest.TestCase):
    def test_final_bound_pack_plan_cannot_receive_substituted_bank_or_source_paths(self):
        value=dict(CPU_fixture=True);sources=dict(parent='REGISTERED_CLOSED_SOURCE_FIXTURE')
        plan=dict(operation='ASSEMBLE_STAGE_BANK',stage_manifest_identity=digest(value),
            qualified_export_sources_identity=digest(sources))
        assembly_plan(plan,value,sources)
        for p,v,s in (({},value,sources),(plan,dict(CPU_fixture=False),sources),(plan,value,dict(parent='OTHER'))):
            with self.assertRaises(PermissionError):assembly_plan(p,v,s)

    def manifest_fixture(self):
        groups,cards,captures,exports=fixtures()
        for row in exports.values():
            row.update(family='DATA_COV1',file='numeric/'+row['id']+'.npz',sha256='e'*64,
                raw_root='EXPLICIT_PRIVATE_BINDING_REQUIRED')
        return groups,cards,captures,exports

    def test_only_complete_real_stage_and_first_four_qualifications_make_a_bank_manifest(self):
        groups,cards,captures,exports=self.manifest_fixture()
        value,reports=manifest(groups,'P24',captures,exports,actual_cards={c['id']:c for c in cards},
            population_sha256='a'*64,adoption_sha256='b'*64)
        self.assertEqual(len(value['episodes']),48);validate_manifest(value)
        self.assertFalse(value['PV1_bank_changed']);self.assertTrue(reports['stage_gate']['count_gate_pass'])
        for field,new in [('statistics_refit',True),('evaluation_rows_used_for_training',True),('PV1_bank_changed',True),('schema','PV1')]:
            bad=dict(value,**{field:new})
            with self.assertRaises(PermissionError):validate_manifest(bad)
        bad=deepcopy(value);bad['episodes'][1]['source_raw_manifest_sha256']=bad['episodes'][0]['source_raw_manifest_sha256']
        with self.assertRaises(PermissionError):validate_manifest(bad)
        with self.assertRaises(PermissionError):manifest(groups,'P24',{}, {},actual_cards={},population_sha256='a'*64,adoption_sha256='b'*64)
        exports[cards[0]['id']]['official_chain']['deterministic_repeat']=False
        with self.assertRaises(PermissionError):manifest(groups,'P24',captures,exports,actual_cards={},population_sha256='a'*64,adoption_sha256='b'*64)

    def numeric_fixture(self,root):
        identifier='PI0294-DATA_COV1-TRAIN-CPU_FIXTURE-T1-C04';numeric=root/'numeric'/f'{identifier}.npz';numeric.parent.mkdir()
        paths=np.asarray([[f'{camera}/{i:04d}.png' for camera in ('body_rgb','wrist_rgb')] for i in range(641)])
        hashes=[]
        for pair in paths:
            row=[]
            for name in pair:
                p=root/'raw'/identifier/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'CPU fixture not actual image')
                p.chmod(0o400);row.append(sha(p))
            hashes.append(row)
        np.savez(numeric,actual_state10=np.zeros((641,10)),committed_absolute10=np.zeros((640,10)),
            valid_window=np.arange(640)<601,rgb_paths=paths,rgb_sha256=np.asarray(hashes),
            label_representation=np.asarray('ABSOLUTE_COMMITTED_EEF10'))
        numeric.chmod(0o400)
        return dict(id=identifier,file='numeric/'+identifier+'.npz',sha256=sha(numeric))

    def test_all_1282_frame_members_and_numeric_must_exist_with_exact_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);row=self.numeric_fixture(root);members=parent_files(root,row,lambda:None)
            self.assertEqual(len(members),1283)
            p=root/'raw'/row['id']/'wrist_rgb/0640.png';p.unlink()
            with self.assertRaises(ValueError):parent_files(root,row,lambda:None)

    def test_metadata_only_or_missing_ready_is_never_a_consumable_bank(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with self.assertRaises(FileNotFoundError):bank_members(root,lambda:None)
            self.assertFalse((root/'READY.json').exists())
            alias=root/'alias';alias.symlink_to(root,target_is_directory=True)
            with self.assertRaises(PermissionError):bank_members(alias,lambda:None)
        for path in ('/var/cache/private/run','/scratch/private/run','/home/private/token','/tmp/binding'):
            with self.assertRaises(PermissionError):public_json(dict(nested=[dict(path=path)]))


if __name__=='__main__':unittest.main()
