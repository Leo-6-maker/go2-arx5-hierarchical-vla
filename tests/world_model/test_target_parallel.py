import json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from scripts.pi0292.durable import OwnerLock
from go2arx5_vla.world_model.target_parallel import validate_layout,verify_inherited_owner,child_environment,wait_children
from go2arx5_vla.world_model.storage import digest

class ParallelOwner(unittest.TestCase):
    def test_peer_failure_stops_only_the_live_owned_slot(self):
        import time
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);live=root/'live';dead=root/'dead';live.mkdir();dead.mkdir()
            a=subprocess.Popen([sys.executable,'-c','raise SystemExit(1)']);a.wait()
            b=subprocess.Popen([sys.executable,'-c',"from pathlib import Path;import sys,time;p=Path(sys.argv[1]);end=time.monotonic()+5\nwhile not p.exists()and time.monotonic()<end:time.sleep(.01)\nraise SystemExit(0 if p.exists()else 2)",str(live/'STOP.json')])
            try:
                wait_children(dict(campaign_root=str(root),registered_root=str(root),deadline_epoch=time.time()+3),[(dict(registered_root=str(dead)),a),(dict(registered_root=str(live)),b)])
                self.assertEqual(b.wait(timeout=2),0);self.assertEqual(json.loads((live/'STOP.json').read_text())['reason'],'PEER_FAILED');self.assertFalse((dead/'STOP.json').exists())
            finally:
                if b.poll()is None:b.kill();b.wait()
    def test_child_environment_keeps_native_ros_paths_after_current_capsule(self):
        env=child_environment(dict(source_root='/fixture/current',native_environment=dict(PYTHONPATH='/fixture/ros',CUDA_VISIBLE_DEVICES='old',TMPDIR='/old'),tmp='/slot1/tmp',logs='/slot1/logs',cache='/slot1/cache'), [8,9])
        self.assertEqual(env['PYTHONPATH'],'/fixture/current/src:/fixture/current:/fixture/ros');self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'')
        self.assertEqual(env['TMPDIR'],'/slot1/tmp');self.assertEqual(env['ROS_LOG_DIR'],'/slot1/logs');self.assertEqual(env['XDG_CACHE_HOME'],'/slot1/cache')

    def test_contact_two_slot_requires_single_qualification_and_real_prior_batch(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);runs=root/'separate-data-disk';runs.mkdir();previous=root/'previous';previous.write_text('{}')
            proof=root/'prior';proof.write_text(json.dumps(dict(status='TWO_ENVIRONMENT_NATIVE_PILOT_PASSED',children=[dict(ticks=4103,shutdown_clean=True,exit_code=0)]*2)))
            batch=dict(schema='WM02_PILOT1_TWO_CONTACT_BATCH_V1',campaign_root=str(root),native_runs_root=str(runs),registered_root=str(runs/'batch'),source_sha='fixture',physical_owner_lock=str(root/'owner'),cpus=list(range(8)),slots=[],pilot1_binding='actual-window',pilot1_binding_sha256='pin',parallel_qualification=str(proof),parallel_qualification_sha256=digest(proof));children=[]
            for i in range(2):
                r=runs/str(i);r.mkdir();c=dict(registered_root=str(r),parallel_batch_spec=str(runs/'batch/EXECUTION_SPEC.json'),campaign_root=str(root),physical_owner_lock=batch['physical_owner_lock'],source_sha='fixture',ROS_DOMAIN_ID=220+i,cpus=list(range(i*4,i*4+4)),phase='collect-target-pilot1-contact',max_ticks=5703,pilot1_binding='actual-window',pilot1_binding_sha256='pin',previous_stage=str(previous));s=r/'SPEC';s.write_text(json.dumps(c));batch['slots'].append(dict(spec=str(s),spec_sha256=digest(s)));children.append(c)
            # This isolates the layout seam; contact.validate has independent real-file tests.
            with patch('go2arx5_vla.world_model.pilot1_contact.validate',return_value=5703)as validate:
                validate_layout(batch,children);self.assertEqual(validate.call_count,2)
                with self.assertRaises(PermissionError):validate_layout(dict(batch,parallel_qualification_sha256='wrong'),children)
                with self.assertRaises(PermissionError):validate_layout(batch,[children[0],dict(children[1],pilot1_binding='different')])
                with self.assertRaises(PermissionError):validate_layout(dict(batch,native_runs_root=str(root/'elsewhere')),children)
                validate.side_effect=PermissionError('single-scene-unqualified')
                with self.assertRaisesRegex(PermissionError,'single-scene-unqualified'):validate_layout(batch,children)
    def test_inherited_lock_remains_exclusive_until_last_child_closes(self):
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp)/'owner';lock=OwnerLock(path);fd=lock.handle.fileno()
            code="from go2arx5_vla.world_model.target_parallel import verify_inherited_owner;import sys;verify_inherited_owner(int(sys.argv[1]),sys.argv[2]);print('READY',flush=True);input()"
            p=subprocess.Popen([sys.executable,'-c',code,str(fd),str(path)],pass_fds=(fd,),stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
            try:
                self.assertEqual(p.stdout.readline().strip(),'READY');lock.close()
                with self.assertRaises(BlockingIOError):OwnerLock(path)
                p.communicate('\n',timeout=5);self.assertEqual(p.returncode,0)
                other=OwnerLock(path);other.close()
            finally:
                if p.poll()is None:p.kill();p.wait()
                lock.close()
    def test_path_match_without_held_lock_is_rejected(self):
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp)/'owner';other=Path(tmp)/'other'
            with path.open('a')as f,other.open('a')as g:
                with self.assertRaises(PermissionError):verify_inherited_owner(f.fileno(),path)
                with self.assertRaises(PermissionError):verify_inherited_owner(g.fileno(),path)
    def test_ros_cpu_or_identity_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);(root/'runs').mkdir();batch=dict(campaign_root=str(root),registered_root=str(root/'runs/batch'),source_sha='fixture',physical_owner_lock=str(root/'owner'),cpus=list(range(8)),slots=[]);children=[]
            for i in range(2):
                p=root/'runs'/str(i);p.mkdir();c=dict(registered_root=str(p),parallel_batch_spec=str(root/'runs/batch/EXECUTION_SPEC.json'),campaign_root=str(root),physical_owner_lock=batch['physical_owner_lock'],source_sha='fixture',ROS_DOMAIN_ID=220+i,cpus=list(range(i*4,i*4+4)),phase='cold-1',max_ticks=50);s=p/'EXECUTION_SPEC.json';s.write_text(json.dumps(c));batch['slots'].append(dict(spec=str(s),spec_sha256=digest(s)));children.append(c)
            validate_layout(batch,children)
            for key,value in [('ROS_DOMAIN_ID',220),('cpus',[0,1]),('source_sha','changed')]:
                changed=[children[0],dict(children[1],**{key:value})]
                with self.assertRaises(PermissionError):validate_layout(batch,changed)
if __name__=='__main__':unittest.main()
