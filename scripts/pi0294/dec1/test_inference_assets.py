"""Tiny opaque fake shards only: no network, model, pickle or allocation."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from . import inference_assets as assets


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')
    return assets.sha(path)


def fixture(root):
    ck=root/'protected';ck.mkdir()
    values={'config.json':dict(model_name='/scratch/private/cosmos',unchanged=[1,2]),
        'processor/processor_config.json':dict(processor_kwargs=dict(model_name='/scratch/private/cosmos'),same=7),
        'processor/embodiment_id.json':dict(EEF10=0),'statistics.json':dict(state=[1,2]),
        'processor/statistics.json':dict(action=[3,4]),
        'model.safetensors.index.json':dict(weight_map={str(i):'model-%05d-of-00003.safetensors'%i for i in (1,2,3)})}
    for n,v in values.items():put(ck/n,v)
    for i in (1,2,3):(ck/('model-%05d-of-00003.safetensors'%i)).write_bytes(('fake-shard-'+str(i)).encode())
    files={p.relative_to(ck).as_posix():assets.sha(p) for p in ck.rglob('*') if p.is_file()}
    parent=put(ck/'CHECKPOINT_COMPLETE.json',dict(global_step=2048,full_resume=True,files=files))
    return ck,files,parent


class FakeHub:
    token='NOT_A_CREDENTIAL'
    def __init__(self,root):self.root=root;self.head='a'*40;self.commits=[]
    def repo_info(self,*a,**kw):return SimpleNamespace(private=False,sha=self.head)
    def list_repo_files(self,*a,**kw):return [p.relative_to(self.root).as_posix() for p in self.root.rglob('*') if p.is_file()]
    def list_repo_tree(self,*a,**kw):
        prefix=kw['path_in_repo'];folder=self.root/prefix
        return [SimpleNamespace(path=p.relative_to(self.root).as_posix(),size=p.stat().st_size,
            lfs=SimpleNamespace(sha256=assets.sha(p)) if p.suffix=='.safetensors' else None)
            for p in folder.rglob('*') if p.is_file()]
    def download(self,repo,name,**kw):
        source=self.root/name
        if 'local_dir' not in kw:return str(source)
        target=Path(kw['local_dir'])/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        return str(target)
    def create_commit(self,*a,**kw):
        if kw['parent_commit']!=self.head:raise ValueError('PARENT_MOVED')
        for op in kw['operations']:
            target=self.root/op.path_in_repo;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(op.path_or_fileobj,target)
        self.commits.append(kw);self.head=('%040x'%len(self.commits));return SimpleNamespace(oid=self.head)


class AssetTests(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);self.root=Path(t.name)
        self.ck,self.files,parent=fixture(self.root)
        self.first=(self.ck/assets.FIRST).read_bytes()
        self.dep=dict(repo_id=assets.REPO,revision=assets.BASE_REV,prefix=assets.BASE_PREFIX,
            name=assets.FIRST,sha256=self.files[assets.FIRST],bytes=len(self.first))
        for name,value in [('PARENT',parent),('R0',self.files['statistics.json']),('FIRST_SHA',self.files[assets.FIRST])]:
            p=patch.object(assets,name,value);p.start();self.addCleanup(p.stop)
        for name,kwargs in [('dependency',dict(return_value=self.dep)),('tensor_header',dict(return_value=None))]:
            p=patch.object(assets,name,**kwargs);p.start();self.addCleanup(p.stop)
        self.hub=FakeHub(self.root/'hub');self.hub.root.mkdir()
        target=self.hub.root/assets.BASE_PREFIX/assets.FIRST;target.parent.mkdir(parents=True);target.write_bytes(self.first)
        for name in assets.LICENSE_NAMES:
            p=self.hub.root/assets.LICENSE_PREFIX/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('Upstream license for fixture only\n')
        self.publisher=self.root/'publisher';self.publisher.mkdir()

    def publish(self):
        return assets.publish(self.publisher,dict(checkpoint=str(self.ck)),api=self.hub,
            add=lambda **kw:SimpleNamespace(**kw),download=self.hub.download)

    def test_delta_never_uploads_frozen_shard_or_full_training_state(self):
        before={p.relative_to(self.ck).as_posix():p.read_bytes() for p in self.ck.rglob('*') if p.is_file()}
        transfer=self.publish()
        self.assertNotIn(assets.FIRST,transfer['files'])
        self.assertFalse(any('optimizer' in n for n in transfer['files']))
        self.assertEqual(len(self.hub.commits),1)
        self.assertNotIn('READY.json',transfer['files'])
        self.assertEqual({p.relative_to(self.ck).as_posix():p.read_bytes() for p in self.ck.rglob('*') if p.is_file()},before)
        manifest=assets.read(self.publisher/'payload/INFERENCE_MANIFEST.json')
        self.assertEqual(assets.read(self.publisher/'payload/config.json'),dict(model_name='nvidia/Cosmos-Reason2-2B',unchanged=[1,2]))
        self.assertFalse(manifest['full_resume']);self.assertFalse(manifest['model_loaded'])

    def test_full_consumer_hashes_reused_local_bytes_then_ready_last(self):
        transfer=self.publish();consumer=self.root/'consumer';consumer.mkdir()
        frozen=self.ck/assets.FIRST;before=frozen.stat()
        result=assets.consume(consumer,dict(transfer=transfer,frozen_local_member=str(frozen)),download=self.hub.download)
        self.assertTrue(os.path.samefile(frozen,consumer/'inference'/assets.FIRST))
        self.assertEqual((before.st_ino,before.st_mode,before.st_mtime_ns),
                         (frozen.stat().st_ino,frozen.stat().st_mode,frozen.stat().st_mtime_ns))
        self.assertEqual(result['inference_files_sha256']['statistics.json'],assets.R0)
        final=self.root/'ready';final.mkdir();shutil.copyfile(consumer/'CONSUMPTION.json',final/'CONSUMPTION.json')
        receipt=assets.ready(final,dict(transfer=transfer),api=self.hub,
            add=lambda **kw:SimpleNamespace(**kw),download=self.hub.download)
        self.assertTrue(receipt['status'].startswith('PASS_FIXED_CORR'))
        self.assertEqual([op.path_in_repo for op in self.hub.commits[-1]['operations']],[assets.PREFIX+'/READY.json'])
        self.assertEqual(len(self.hub.commits),2)

    def test_corrupt_reused_frozen_shard_is_not_trusted_from_old_receipt(self):
        transfer=self.publish();consumer=self.root/'consumer';consumer.mkdir()
        (self.ck/assets.FIRST).write_bytes(b'bad')
        with self.assertRaisesRegex(PermissionError,'REUSED_LOCAL_FROZEN_BYTES'):
            assets.consume(consumer,dict(transfer=transfer,frozen_local_member=str(self.ck/assets.FIRST)),download=self.hub.download)
        self.assertFalse((consumer/'inference').exists())

    def test_unknown_upload_has_no_consume_or_ready_admission(self):
        transfer=self.publish();transfer['status']='UPLOAD_INTENT_UNKNOWN_NO_RETRY'
        consumer=self.root/'consumer';consumer.mkdir()
        with self.assertRaises(PermissionError):
            assets.consume(consumer,dict(transfer=transfer,frozen_local_member=str(self.ck/assets.FIRST)),download=self.hub.download)

    def test_invalid_manifest_or_extra_members_rejected(self):
        self.publish();original=assets.read(self.publisher/'payload/INFERENCE_MANIFEST.json')
        for change in (lambda v:v.update(full_resume=True),lambda v:v.update(statistics_sha256='0'*64),
                lambda v:v['files'].update({'optimizer.pt':dict(bytes=1,sha256='a'*64)}),
                lambda v:v['original_inference_sha256'].update({assets.FIRST:'0'*64}),
                lambda v:v['files']['statistics.json'].update(sha256='0'*64),
                lambda v:v['files']['statistics.json'].update(bytes=True),
                lambda v:v['metadata_changes'].update({'new':'unexpected'})):
            with self.subTest(change=change):
                value=deepcopy(original);change(value)
                with self.assertRaises(PermissionError):assets.validate_manifest(value)

    def test_changed_original_prevents_any_upload(self):
        (self.ck/'statistics.json').write_text('changed')
        with self.assertRaisesRegex(PermissionError,'ORIGINAL_MEMBER_CHANGED'):self.publish()
        self.assertEqual(self.hub.commits,[])

    def test_existing_prefix_preserved_not_overwritten(self):
        p=self.hub.root/assets.PREFIX/'prior';p.parent.mkdir(parents=True);p.write_text('preserve')
        with self.assertRaisesRegex(PermissionError,'FRESH_PREFIX'):self.publish()
        self.assertEqual(p.read_text(),'preserve');self.assertEqual(self.hub.commits,[])

    def test_consumer_qualification_cannot_be_invented_by_ready(self):
        transfer=self.publish();final=self.root/'ready';final.mkdir()
        put(final/'CONSUMPTION.json',dict(status='PASS_FULL_CONSUMER_BYTES_AND_REUSED_FROZEN_SHARD_NO_MODEL'))
        with self.assertRaisesRegex(PermissionError,'CONSUMER_READBACK'):
            assets.ready(final,dict(transfer=transfer),api=self.hub,add=lambda **kw:SimpleNamespace(**kw),download=self.hub.download)
        self.assertEqual(len(self.hub.commits),1)

    def test_standard_guard_rejects_missing_budget_expired_lease_and_mutable_input(self):
        root=self.root/'guard';root.mkdir();binding=root/'OUTPUT_BINDING.private.json'
        lease=self.root/'lease.json';registry=self.root/'registry.json'
        put(lease,dict(status='ACTIVE',run_uuid=root.name,authority_release=assets.AUTHORITY,deadline_epoch=assets.HARD_CLOSE))
        put(registry,dict(entries=[dict(role='one',path=str(root))]))
        small={'BINDINGS.private.json':dict(public_anonymous_read=True),
            'CPU_BUDGET_PREPAID.json':dict(authority_release=assets.AUTHORITY,original_T0_epoch=1792348196,
                campaign_run_uuid='20261018T182956Z-localpc-pv1',costs_inside_original_PV1_and_DEC1=True,
                CPU_upper_s_by_mode=dict(publish=4500,consume=4500,ready=180),
                total_Hopper_and_alternate_CPU_prepaid_s=9180,transfer_reserved_bytes=16*1024**3,unknown_effects=[])}
        for n,v in small.items():put(root/n,v);(root/n).chmod(0o400)
        manifest=dict(schema='DEC1_INFERENCE_ASSET_CHILD_V1',run_uuid=root.name,owner_uid=os.getuid(),mode='consume',
            authority_release=assets.AUTHORITY,parent_authority_release=assets.PARENT_AUTHORITY,
            campaign_run_uuid='20261018T182956Z-localpc-pv1',original_T0_epoch=1792348196,
            deadline_epoch=assets.HARD_CLOSE,CPU_upper_s=4500,GPU_PBS=0,new_models=0,
            transfer_upper_bytes=8*1024**3,execution_profile='vanda-neural-probe',registered_runs_root=str(self.root),
            sealed_inputs={n:assets.sha(root/n) for n in small})
        def seal():
            (root/'RUN_MANIFEST.json').chmod(0o600) if (root/'RUN_MANIFEST.json').exists() else None
            put(root/'RUN_MANIFEST.json',manifest);(root/'RUN_MANIFEST.json').chmod(0o400)
            binding.chmod(0o600) if binding.exists() else None
            put(binding,dict(uid=os.getuid(),role='one',registry=str(registry),lease=str(lease),
                run_manifest_sha256=assets.sha(root/'RUN_MANIFEST.json')));binding.chmod(0o400)
        seal()
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}),patch.object(assets.time,'time',return_value=1792528000):
            self.assertEqual(assets.guard(root,'consume')[0],manifest)
            original=deepcopy(manifest)
            for change in (lambda:manifest.update(CPU_upper_s=1),
                    lambda:manifest['sealed_inputs'].pop('CPU_BUDGET_PREPAID.json'),
                    lambda:manifest.update(original_T0_epoch=1792528000),
                    lambda:manifest.update(execution_profile='unregistered')):
                change();seal()
                with self.assertRaises(PermissionError):assets.guard(root,'consume')
                manifest=deepcopy(original)
            seal();(root/'BINDINGS.private.json').chmod(0o600)
            with self.assertRaises(PermissionError):assets.guard(root,'consume')
            (root/'BINDINGS.private.json').chmod(0o400)
            put(lease,dict(status='CLOSED'))
            with self.assertRaises(PermissionError):assets.guard(root,'consume')

    def test_public_consumer_uses_explicit_false_not_ambient_credentials(self):
        with patch.dict(os.environ,{'HF_TOKEN':'DUMMY_NOT_A_REAL_CREDENTIAL'}):
            self.assertIs(assets.authentication(dict(public_anonymous_read=True),'consume'),False)
        for mode in ('publish','ready'):
            with self.assertRaisesRegex(PermissionError,'CONSUMER_ONLY'):
                assets.authentication(dict(public_anonymous_read=True),mode)
        with self.assertRaises(PermissionError):
            assets.authentication(dict(public_anonymous_read=True,existing_token_file='another-site'),'consume')
        with self.assertRaisesRegex(PermissionError,'LOCAL_EXISTING_WRITER'):
            assets.authentication(dict(existing_token_file=str(self.root/'absent')),'consume')


if __name__=='__main__':unittest.main()
