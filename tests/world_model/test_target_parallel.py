import json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from scripts.pi0292.durable import OwnerLock
from go2arx5_vla.world_model.target_parallel import validate_layout,verify_inherited_owner,child_environment
from go2arx5_vla.world_model.storage import digest

class ParallelOwner(unittest.TestCase):
    def test_child_environment_keeps_native_ros_paths_after_current_capsule(self):
        env=child_environment(dict(source_root='/fixture/current',native_environment=dict(PYTHONPATH='/fixture/ros',CUDA_VISIBLE_DEVICES='old')), [8,9])
        self.assertEqual(env['PYTHONPATH'],'/fixture/current/src:/fixture/current:/fixture/ros');self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'')
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
