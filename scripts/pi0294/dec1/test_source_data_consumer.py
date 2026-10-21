import copy
import json
import os
from pathlib import Path
import tempfile
import tarfile
import unittest
from unittest.mock import patch

from scripts.pi0294.dec1 import source_data_consumer as c


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():path.unlink()
    path.write_text(json.dumps(value)+'\n');path.chmod(0o400)
    return c.sha(path)


class SourceCPUContracts(unittest.TestCase):
    def fixture(self,r):
        receipt=dict(status='PASS_FIXED_REVISION_READY_LAST_EXACT_FULL_SET_SIZE_SHA256',
            repo_id='leoyuer/pi0294-public-artifacts',prefix=c.PREFIX,revision=c.REVISION,dataset_manifest_sha256=c.BANK)
        paid=dict(authority_release=c.AUTHORITY,amendment=c.VAP1,original_T0_epoch=1792348196,
            campaign_run_uuid='20261018T182956Z-localpc-pv1',costs_inside_original_PV1_and_DEC1=True,
            CPU_upper_s=c.CPU,transfer_reserved_bytes=c.TRANSFER,output_reserved_bytes=c.OUTPUT,unknown_effects=[])
        proof=dict(source_sha='a'*40,CI_head_sha='a'*40,CI_status='completed',CI_conclusion='success',files_exact_readback=True)
        data={'BINDINGS.private.json':{},'CPU_BUDGET_PREPAID.json':paid,'PUBLICATION_AND_CI.json':proof,
            'plans/PV1_HF_BANK_DELIVERY.json':receipt,'plans/VAP1_ADOPTION_AND_DELTA_BUDGET.json':dict(fixture=True),
            'capsule/code/scripts/pi0294/dec1/source_data_consumer.py':'CPU_FIXTURE_NOT_SOURCE',
            'SOURCE_SEAL.json':dict(source_sha='a'*40)}
        seals={n:put(r/n,v) for n,v in data.items()}
        m=dict(schema=c.SCHEMA,run_uuid=r.name,authority_release=c.AUTHORITY,amendment=c.VAP1,
            parent_authority_release=c.PARENT_AUTHORITY,owner='EXISTING_LOCAL_PC',owner_uid=os.getuid(),
            campaign_run_uuid=paid['campaign_run_uuid'],original_T0_epoch=1792348196,deadline_epoch=1792600000,
            CPU_upper_s=c.CPU,transfer_upper_bytes=c.TRANSFER,output_upper_bytes=c.OUTPUT,
            zero_effects=dict(GPU=0,PBS=0,models=0,queries=0,updates=0,physics=0),
            execution_profile='vanda-neural-probe',registered_runs_root=str(r.parent),sealed_inputs=seals,
            code=dict(vla_commit='a'*40))
        put(r/'RUN_MANIFEST.json',m)
        put(r/'lease.json',dict(status='ACTIVE',owner='EXISTING_LOCAL_PC',authority_release=c.AUTHORITY,
            amendment=c.VAP1,run_uuid=r.name,deadline_epoch=m['deadline_epoch']))
        put(r/'registry.json',dict(entries=[dict(role='fixture',path=str(r))]))
        b=dict(uid=os.getuid(),host=os.uname().nodename,run_root=str(r),run_manifest_sha256=c.sha(r/'RUN_MANIFEST.json'),
            lease=str(r/'lease.json'),registry=str(r/'registry.json'),role='fixture')
        put(r/'OUTPUT_BINDING.private.json',b)
        return m,b,seals

    def test_exact_CPU_only_scope_and_fail_closed_before_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp)/'child';r.mkdir();m,b,seals=self.fixture(r)
            own=r/'capsule/code/scripts/pi0294/dec1/source_data_consumer.py'
            with patch.object(c,'__file__',str(own)),patch.object(c,'DELIVERY',seals['plans/PV1_HF_BANK_DELIVERY.json']),\
                    patch.object(c,'ADOPTION',seals['plans/VAP1_ADOPTION_AND_DELTA_BUDGET.json']),\
                    patch.dict(os.environ,dict(CUDA_VISIBLE_DEVICES='',PBS_JOBID='')):
                self.assertEqual(c.validate(r,now=lambda:1792559000)[0],m)
                mutations=[('amendment','wrong'),('original_T0_epoch',1792559000),('CPU_upper_s',c.CPU+1),
                    ('transfer_upper_bytes',c.TRANSFER+1),('output_upper_bytes',c.OUTPUT+1),
                    ('owner','SECOND_OWNER'),('execution_profile','hopper-neural'),
                    ('zero_effects',dict(GPU=1,PBS=1,models=1,queries=1,updates=1,physics=1))]
                for key,value in mutations:
                    bad=copy.deepcopy(m);bad[key]=value;put(r/'RUN_MANIFEST.json',bad)
                    b['run_manifest_sha256']=c.sha(r/'RUN_MANIFEST.json');put(r/'OUTPUT_BINDING.private.json',b)
                    with self.subTest(key=key),self.assertRaises(PermissionError):c.validate(r,now=lambda:1792559000)
                put(r/'RUN_MANIFEST.json',m);b['run_manifest_sha256']=c.sha(r/'RUN_MANIFEST.json');put(r/'OUTPUT_BINDING.private.json',b)
                with self.assertRaises(PermissionError):c.validate(r,now=lambda:m['deadline_epoch'])
                with patch.dict(os.environ,dict(PBS_JOBID='CPU_FIXTURE_NOT_A_REAL_JOB')),self.assertRaises(PermissionError):
                    c.validate(r,now=lambda:1792559000)
                paid=c.read(r/'CPU_BUDGET_PREPAID.json');paid['CPU_upper_s']=0
                put(r/'CPU_BUDGET_PREPAID.json',paid)
                with self.assertRaises(PermissionError):c.validate(r,now=lambda:1792559000)

    def test_unsealed_and_symlink_input_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);put(r/'x',{})
            c.sealed(r,'x');(r/'x').chmod(0o600)
            with self.assertRaises(PermissionError):c.sealed(r,'x')
            (r/'x').chmod(0o400);(r/'link').symlink_to(r/'x')
            with self.assertRaises(PermissionError):c.sealed(r,'link')

    def test_malformed_delivery_has_no_download_and_preserves_first_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);receipt=dict(prefix=c.PREFIX,files=[dict(path=c.PREFIX+'/../escape',bytes=1)])
            with patch.object(c,'download_dataset') as network,self.assertRaises(PermissionError):
                c.consume(r,receipt,{},lambda *a:None,get=network)
            network.assert_not_called()
            with self.assertRaises(FileExistsError):c.consume(r,receipt,{},lambda *a:None,get=network)

    def test_actual_fixed_archive_extraction_precedes_official_CPU_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);r=base/'child';r.mkdir();payload=base/'payload';payload.mkdir()
            put(payload/'DATASET_MANIFEST.json',dict(episodes=[dict(id=str(i)) for i in range(139)]))
            source_ready=put(payload/'READY.json',dict(CPU_fixture=True))
            archive=base/'metadata.tar.gz';names=('DATASET_MANIFEST.json','READY.json')
            with tarfile.open(archive,'w:gz') as tar:
                for name in names:tar.add(payload/name,arcname=name)
            proof=dict(sha256=c.sha(archive),bytes=archive.stat().st_size,
                member_sha256={name:c.sha(payload/name) for name in names})
            index=base/'ARCHIVE_MEMBER_MANIFEST.json'
            put(index,dict(archives={'metadata.tar.gz':proof},dataset_manifest_sha256=c.sha(payload/'DATASET_MANIFEST.json'),source_ready_sha256=source_ready))
            ready=base/'READY.json';put(ready,dict(archive_index_sha256=c.sha(index),statistics_sha256=c.R0,
                dataset_manifest_sha256=c.sha(payload/'DATASET_MANIFEST.json')))
            files={p.name:p for p in (index,ready,archive)}
            receipt=dict(repo_id='CPU_fixture_not_HF',prefix='fixture',revision='a'*40,
                files=[dict(path='fixture/'+n,bytes=p.stat().st_size,sha256=c.sha(p)) for n,p in files.items()])
            calls=[]
            def get(repo,name,**kw):
                calls.append((name,kw));self.assertIs(kw['token'],False);return files[Path(name).name]
            def members(target,guard):
                actual={n:target/n for n in names}
                for n,p in actual.items():self.assertEqual(c.sha(p),proof['member_sha256'][n])
                return c.read(target/'DATASET_MANIFEST.json'),{},actual
            def official(*args):
                self.assertTrue((r/'RAW_BINDINGS.private.json').exists())
                return dict(actual_Hopper_official_processor_and_collator_batches=[dict(CPU_fixture=True)])
            with patch.object(c,'BANK',c.sha(payload/'DATASET_MANIFEST.json')),patch.object(c,'SOURCE_READY',source_ready),\
                    patch.object(c,'bank_members',side_effect=members),patch.object(c,'official_consumption',side_effect=official):
                result=c.consume(r,receipt,dict(official_source='fixture',cosmos='fixture'),lambda *a:None,get=get)
            self.assertEqual(len(calls),3);self.assertEqual(result['new_optimizer_updates'],0)
            self.assertFalse(result['training_qualification']);self.assertNotIn('actual_Hopper_official_processor_and_collator_batches',result)
            self.assertEqual(result['status'],'PASS_VANDA_CPU_FIXED_PV1_139_TRAIN_FULL_MEMBER_AND_READER')


if __name__=='__main__':unittest.main()
