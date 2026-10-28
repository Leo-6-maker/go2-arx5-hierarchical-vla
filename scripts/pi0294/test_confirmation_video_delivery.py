import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.pi0292.relay_protocol import digest
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.mp1 import confirmation_video_delivery as d


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


class ConfirmationVideoDeliveryTests(unittest.TestCase):
    def test_original_clock_cpu_and_no_science_binding_required_before_publication(self):
        import os
        for defect in [None,'expired','hardclose','T0','GPU_update','query','physics','CPU','payload','profile_binding','CPU_id','unsealed']:
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);adoption=root/d.n1.FIRST_ADOPTION;put(adoption,{});adoption.chmod(0o400)
                put(root/'MP1_ADOPTION_AND_INCREMENTAL_BUDGET.json',{})
                q=dict(parent_run=root.name,repo_id=d.REPO,child=d.config.plan.CHILD,
                    N1_authority_release=d.n1.RELEASE,N1_adoption_sha256=d.n1.ADOPTED_DELTA_SHA,N1_T0_epoch=1,
                    maximum_CPU_s=1800,worker_CPU_s=1620,registration_CPU_upper_s=180,maximum_payload_bytes=d.MAX_BYTES,
                    new_queries=0,new_physics=0,new_training_updates=0,CPU_id=min(os.sched_getaffinity(0)),
                    public_authority='DIRECT_USER_PUBLIC_GENERATED_SOURCE_VIDEO_PERMISSION',created_epoch=10,deadline_epoch=20)
                if defect=='expired':q['deadline_epoch']=11
                elif defect=='hardclose':q['deadline_epoch']=101
                elif defect=='T0':q['N1_T0_epoch']=2
                elif defect=='GPU_update':q['new_training_updates']=1
                elif defect=='query':q['new_queries']=1
                elif defect=='physics':q['new_physics']=1
                elif defect=='CPU':q['worker_CPU_s']=1800
                elif defect=='payload':q['maximum_payload_bytes']+=1
                elif defect=='CPU_id':q['CPU_id']=-1
                binding=root/d.BINDING;put(binding,q);binding.chmod(0o400 if defect!='unsealed' else 0o600)
                runtime=dict(execution=dict(confirmation_video_binding_sha256=sha(binding)))
                if defect=='profile_binding':runtime['execution']['confirmation_video_binding_sha256']='changed'
                def checksum(p):return d.n1.ADOPTED_DELTA_SHA if Path(p)==adoption else sha(p)
                with patch.object(d,'sha',side_effect=checksum),patch.object(d.n1,'validated_delta',return_value=dict(autonomy_started_epoch=1,hard_closeout_epoch=100)):
                    if defect is None:self.assertEqual(d.checked_request(root,runtime,11),q)
                    else:
                        with self.assertRaises(PermissionError):d.checked_request(root,runtime,11)

    def setup_snapshot(self,root):
        child=root/'children'/d.config.plan.CHILD;plan=dict(fixed_test_fixture_not_science=True)
        put(child/d.config.PLAN,plan);row=dict(attempt_id='fixed-attempt',status='ENGINEERING_INCOMPLETE')
        put(child/'episodes/fixed-attempt/MP1_ADJUDICATION.json',row)
        put(child/'RESOURCE_LEDGER.json',dict(reservations={},CPU_reserved_s=0))
        (child/'worker_processes').mkdir();(child/'aux_worker_processes').mkdir()
        proof=dict(schema='MP1_CONFIRM_FINAL_EFFECT_CLOSURE_V1',run_uuid=child.name,
            plan_sha256=sha(child/d.config.PLAN),rows_sha256=digest([row]),unknown_effects=[],pending_queries=0,
            pending_optimizer_updates=0,all_Core_and_Hopper_consumers_terminal=True,all_action_effects_classified=True,
            native_allocations=[dict(job_id='unit-fixture-not-job',state='F',project='CFP04-CF-007',actual_GPU_s=10)])
        put(root/d.CLOSURE,proof);(root/d.CLOSURE).chmod(0o400)
        request=dict(final_effect_closure_sha256=sha(root/d.CLOSURE),attempt_ids=['fixed-attempt'])
        value=dict(actual_videos=1,coverage=dict(status='PARTIAL_ACTUAL_COVERAGE_NO_FABRICATED_RESULTS'),
            rows=[dict(status='ENGINEERING_INCOMPLETE')])
        return child,plan,proof,request,value

    def test_terminal_snapshot_preserves_partial_and_engineering_not_fake_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);child,plan,_,q,value=self.setup_snapshot(root)
            files={'movies/test.mp4':dict(bytes=1,sha256='a'*64)};sources={'movies/test.mp4':child/'test.mp4'}
            with patch.object(d.config,'validate_plan',return_value=plan),patch.object(d,'inventory',return_value=(value,files,sources)):
                index,actual_files,actual_sources=d.closed_snapshot(root,q)
            self.assertFalse(index['complete_confirmation']);self.assertEqual(index['rows'][0]['status'],'ENGINEERING_INCOMPLETE')
            self.assertEqual(files,actual_files);self.assertEqual(sources,actual_sources)
            self.assertTrue(index['all_original_terminal_attempts_in_snapshot'])

    def test_unknown_effect_live_native_cherry_picking_or_changed_source_refuses(self):
        for defect in ['unknown','query','updates','consumer','classification','native','project','GPU','plan','rows','subset','unsealed','reservation']:
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);child,plan,proof,q,_=self.setup_snapshot(root)
                if defect=='unknown':proof['unknown_effects']=['unknown-action']
                elif defect=='query':proof['pending_queries']=1
                elif defect=='updates':proof['pending_optimizer_updates']=1
                elif defect=='consumer':proof['all_Core_and_Hopper_consumers_terminal']=False
                elif defect=='classification':proof['all_action_effects_classified']=False
                elif defect=='native':proof['native_allocations'][0]['state']='R'
                elif defect=='project':proof['native_allocations'][0]['project']='other-project'
                elif defect=='GPU':proof['native_allocations'][0]['actual_GPU_s']=-1
                elif defect=='plan':proof['plan_sha256']='changed'
                elif defect=='rows':proof['rows_sha256']='changed'
                elif defect=='subset':q['attempt_ids']=[]
                elif defect=='reservation':put(child/'RESOURCE_LEDGER.json',dict(reservations=dict(active=True),CPU_reserved_s=0))
                (root/d.CLOSURE).chmod(0o600);put(root/d.CLOSURE,proof)
                if defect!='unsealed':(root/d.CLOSURE).chmod(0o400)
                q['final_effect_closure_sha256']=sha(root/d.CLOSURE)
                with patch.object(d.config,'validate_plan',return_value=plan),self.assertRaises(PermissionError):d.closed_snapshot(root,q)

    def test_live_exact_recorded_process_blocks_even_if_aggregate_claims_terminal(self):
        import os
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);child,plan,_,q,_=self.setup_snapshot(root)
            put(child/'aux_worker_processes/active.json',dict(pid=os.getpid(),proc_start=Path('/proc/self/stat').read_text().split()[21]))
            with patch.object(d.config,'validate_plan',return_value=plan),self.assertRaisesRegex(PermissionError,'ACTIVE_CONFIRMATION'):d.closed_snapshot(root,q)

    def test_bound_payload_size_still_applies_to_actual_video_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,plan,_,q,value=self.setup_snapshot(root)
            for files in [{},{'movies/large.mp4':dict(bytes=d.MAX_BYTES+1,sha256='a'*64)}]:
                with self.subTest(files=bool(files)),patch.object(d.config,'validate_plan',return_value=plan),patch.object(d,'inventory',return_value=(value,files,{})),self.assertRaises(PermissionError):d.closed_snapshot(root,q)

    def test_ready_uploaded_last_only_after_first_exact_readback_and_source_recheck(self):
        events=[];files_written={};revisions=[]
        class Add:
            def __init__(self,**kw):self.__dict__.update(kw)
        class API:
            def __init__(self,token):self.token=token
            def repo_info(self,*a,**k):return SimpleNamespace(private=False,sha='a'*40)
            def list_repo_files(self,*a,**k):return []
            def create_commit(self,*a,**kw):
                events.append([Path(op.path_in_repo).name for op in kw['operations']])
                for op in kw['operations']:files_written[op.path_in_repo]=Path(op.path_or_fileobj).read_bytes()
                rev=str(len(events))*40;revisions.append(rev);return SimpleNamespace(oid=rev)
        def verify(api,prefix,revision,files,download,token):
            events.append('verify-'+revision[0]);revisions.append(revision)
            self.assertEqual(set(files_written),{prefix+'/'+n for n in files})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'artifacts').mkdir();token=root/'token';token.write_text('test-not-real-credential')
            movie=root/'test.mp4';movie.write_bytes(b'unit-test-not-real-video');sources={'movies/test.mp4':movie}
            files={'movies/test.mp4':dict(bytes=movie.stat().st_size,sha256=sha(movie))}
            index=dict(actual_videos=1,complete_confirmation=False)
            fake=SimpleNamespace(HfApi=API,CommitOperationAdd=Add,hf_hub_download=lambda *a,**k:None)
            with patch.dict('sys.modules',{'huggingface_hub':fake}),patch.object(d,'validate',return_value=dict(code=dict(vla_commit='a'*40))),patch.object(d,'checked_request',return_value=dict(deadline_epoch=__import__('time').time()+60,CPU_id=min(__import__('os').sched_getaffinity(0)))),patch.object(d,'closed_snapshot',return_value=(index,files,sources)),patch.object(d,'verify_videos',side_effect=verify),patch.object(d,'charge_cpu'),patch.object(d.signal,'alarm'),patch.object(d.resource,'setrlimit'),patch.object(d.os,'sched_setaffinity'),patch.object(d.subprocess,'check_output',return_value='{}'):
                result=d.publish(root,token)
            self.assertEqual(events[0],['test.mp4','VIDEO_INDEX.json']);self.assertTrue(events[1].startswith('verify-'))
            self.assertEqual(events[2],['READY.json']);self.assertTrue(events[3].startswith('verify-'))
            self.assertTrue(result['READY_uploaded_last']);self.assertEqual(result['status'],'PASS_SOURCE_CONFIRMATION_VIDEOS_FIXED_REVISION_READBACK')
            self.assertEqual(movie.read_bytes(),b'unit-test-not-real-video');self.assertFalse(result['complete_confirmation'])

    def test_new_bounded_video_hash_readback_not_model_or_history_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'test.mp4';path.write_bytes(b'fixture'*(1024**2))
            name='movies/PI0294-MP1-SOURCE-CONFIRM-fixture.mp4';prefix='fixed-owned-prefix'
            api=SimpleNamespace(list_repo_tree=lambda *a,**k:[SimpleNamespace(path=prefix+'/'+name,size=path.stat().st_size)])
            files={name:dict(bytes=path.stat().st_size,sha256=sha(path))};calls=[]
            def download(*args,**kwargs):calls.append(kwargs['revision']);return str(path)
            d.verify_videos(api,prefix,'a'*40,files,download,'test-not-credential')
            self.assertEqual(calls,['a'*40])
            for forbidden in ['weights.safetensors','../movie.mp4','movies/old-history.mp4','raw.png']:
                with self.subTest(forbidden=forbidden),self.assertRaises(PermissionError):
                    d.verify_videos(api,prefix,'a'*40,{forbidden:next(iter(files.values()))},download,'test-not-credential')
            files[name]['sha256']='f'*64
            with self.assertRaisesRegex(ValueError,'HASH'):d.verify_videos(api,prefix,'a'*40,files,download,'test-not-credential')


if __name__=='__main__':unittest.main()
