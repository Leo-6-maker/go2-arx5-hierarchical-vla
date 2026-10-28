import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.pi0294.native_dataset import sha
from .mp1.student_probe_delivery import publish,consume,run,configuration,MODELS,REPO,PREFIX
from .test_mp1_student_probe_audit import fixture as probe_fixture


def setup(root):
    (root/'artifacts').mkdir()
    child=root/'children/source-pair-01';folder=child/'evidence/S-BC-512-TRAIN4/probes'
    folder.mkdir(parents=True);plan,_=probe_fixture(folder)
    (child/'plans').mkdir();(child/'plans/STUDENT_PROBES.private.json').write_text(json.dumps(plan))
    (child/'RUN_MANIFEST.json').write_text('{}')
    p=child/'training/S-BC/checkpoint-512';p.mkdir(parents=True);(p/'CHECKPOINT_COMPLETE.json').write_text('{}')
    proof=root/'evidence/SOURCE_STUDENT_PROBE_AUDIT_01/RESULT.json';proof.parent.mkdir(parents=True)
    proof.write_text(json.dumps(dict(status='PASS_TWELVE_SAVED_STUDENT_DRAWS_ORIGINAL_E0_INPUTS_NOISE_AND_CHECKPOINT',
        probe_states=12,new_model_queries=0,new_physics=0)))
    return dict(child_manifest_sha256=sha(child/'RUN_MANIFEST.json'),student_audit_sha256=sha(proof)),folder


class StudentDeliveryTest(unittest.TestCase):
    def test_only_three_existing_model_identities_have_distinct_sealed_prefixes(self):
        self.assertEqual(configuration({})[0],'S-BC-512-TRAIN4')
        self.assertEqual(len({configuration(dict(model_label=n))[1] for n in MODELS}),3)
        for name in ['S-CORR-2048-TRAIN4','M1','unbound']:
            with self.assertRaises(PermissionError):configuration(dict(model_label=name))

    def test_other_model_cannot_consume_prior_bc512_transfer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);transfer=dict(status='PASS_PRIVATE_EXISTING_STUDENT_DRAWS_FIXED_REVISION',repo_id=REPO,
                prefix='PI0294/'+root.name+'/'+PREFIX,files={str(i):dict(bytes=1) for i in range(16)})
            for label in ['S-BC-2048-TRAIN4','S-CORR-512-TRAIN4']:
                with self.subTest(label=label),self.assertRaises(PermissionError):
                    consume(root,dict(model_label=label,transfer=transfer),'not-a-real-token')

    def fake_hub(self,private=True,files=()):
        class API:
            def __init__(self,**kwargs):pass
            def repo_info(self,*args,**kwargs):return SimpleNamespace(private=private,sha='a'*40)
            def list_repo_files(self,*args,**kwargs):return list(files)
        return SimpleNamespace(HfApi=API,CommitOperationAdd=object,hf_hub_download=lambda:None)

    def test_private_repo_required_no_visibility_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);spec,_=setup(root)
            with patch.dict(sys.modules,huggingface_hub=self.fake_hub(private=False)):
                with self.assertRaisesRegex(PermissionError,'PRIVATE_EVIDENCE_REPOSITORY'):
                    publish(root,spec,'not-a-real-token')

    def test_existing_prefix_is_not_reuploaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);spec,_=setup(root);name='PI0294/'+root.name+'/'+PREFIX+'/READY.json'
            with patch.dict(sys.modules,huggingface_hub=self.fake_hub(files=[name])):
                with self.assertRaisesRegex(FileExistsError,'NO_BLIND_REUPLOAD'):
                    publish(root,spec,'not-a-real-token')

    def test_tampered_source_array_stops_before_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);spec,folder=setup(root);next(folder.glob('*.npz')).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'STUDENT_ARRAY_SOURCE'):
                publish(root,spec,'not-a-real-token')

    def test_foreign_repo_prefix_or_status_cannot_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            good=dict(status='PASS_PRIVATE_EXISTING_STUDENT_DRAWS_FIXED_REVISION',repo_id=REPO,
                prefix='PI0294/'+root.name+'/'+PREFIX,files={str(i):dict(bytes=1) for i in range(16)})
            for field,value in [('status','UNKNOWN'),('repo_id','foreign/repo'),('prefix','other')]:
                bad=dict(good,**{field:value})
                with self.subTest(field=field),self.assertRaises(PermissionError):
                    consume(root,dict(transfer=bad),'not-a-real-token')

    def test_binding_path_escape_is_rejected_before_manifest_or_token_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            for path in ['/tmp/foreign.json','../foreign.json']:
                with self.subTest(path=path),self.assertRaises(PermissionError):
                    run(Path(tmp),path,'/does-not-exist')


if __name__=='__main__':unittest.main()
