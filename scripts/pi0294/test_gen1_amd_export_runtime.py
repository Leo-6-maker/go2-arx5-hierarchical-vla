"""CPU role/IO tests. No actual PBS, official processor or Teacher evidence."""
from copy import deepcopy
import hashlib,json,os,subprocess,sys
from pathlib import Path
import pytest
from scripts.pi0294.generalization import amd_export_runtime as r,amd_export_accounting as accounting
from scripts.pi0294.pv1.export import materialize_rgb

def native_fixture():
 req=dict(request_id='fixture',submission_nonce='fixture-nonce')
 job=dict(job_state='R',project='CFP04-CF-007',Job_Name=accounting.JOB_NAME,
  Job_Owner='e1192548@stdct-login-01',exec_host='compute-fixture/4',
  Resource_List=dict(ncpus=1,ngpus=0,walltime='10:10:00',select='1:ncpus=1:ngpus=0'),
  exec_vnode='(compute-fixture:ncpus=1)',Variable_List=dict(GEN1_REQUEST_ID='fixture',GEN1_SUBMISSION_NONCE='fixture-nonce'))
 return job,req

def test_exact_zero_gpu_cpu_allocation():
 job,req=native_fixture();assert r.native_allocation(job,job_id='123.stdct-mgmt-02',hostname='compute-fixture',request=req)is job

@pytest.mark.parametrize('bad',['host','project','GPU','select_GPU','name','CPU','wall','nonce','job_id'])
def test_other_job_or_hardware_cannot_enter_export(bad):
 job,req=native_fixture();host='compute-fixture';jid='123.stdct-mgmt-02'
 if bad=='host':host='stdct-login-01'
 elif bad=='project':job['project']='CFP04-CF-009'
 elif bad=='GPU':job['Resource_List']['ngpus']=1
 elif bad=='select_GPU':job['exec_vnode']='(compute-fixture:ncpus=1:ngpus=1)'
 elif bad=='name':job['Job_Name']='other-human-work'
 elif bad=='CPU':job['Resource_List']['ncpus']=8
 elif bad=='wall':job['Resource_List']['walltime']='168:00:00'
 elif bad=='nonce':job['Variable_List']['GEN1_SUBMISSION_NONCE']='other'
 else:jid='not-an-actual-job'
 with pytest.raises(PermissionError):r.native_allocation(job,job_id=jid,hostname=host,request=req)

def image_files(tmp_path):
 folder=tmp_path/'registered-input';folder.mkdir();out=tmp_path/'output';out.mkdir();paths=[];hashes=[]
 for camera in('body_rgb','wrist_rgb'):
  p=folder/camera/'0000.png';p.parent.mkdir();p.write_bytes(b'IO-only-test-not-a-rendered-frame');p.chmod(0o400)
  paths.append(camera+'/0000.png');hashes.append(hashlib.sha256(p.read_bytes()).hexdigest())
 return folder,out,dict(id='GEN1-PILOT-aaaaaaaaaaaaaaaa-T1',study_split='G_TRAIN',split='G_TRAIN'),dict(rgb_paths=[paths],rgb_sha256=[hashes])

def test_registered_GEN_rgb_reuse_has_one_raw_namespace_no_inode_or_byte_change(tmp_path):
 folder,out,card,arrays=image_files(tmp_path);before={str(p):p.stat()for p in folder.rglob('*')if p.is_file()}
 raw=materialize_rgb(folder,out,card,arrays,lambda:None,reuse_registered_source_rgb=True)
 assert raw==folder and not(out/'raw').exists()and list(out.iterdir())==[]
 for path,old in before.items():
  now=Path(path).stat();assert (now.st_ino,now.st_size,now.st_mtime_ns,now.st_ctime_ns)==(old.st_ino,old.st_size,old.st_mtime_ns,old.st_ctime_ns)

def test_legacy_default_still_materializes_original_hardlinks(tmp_path):
 folder,out,card,arrays=image_files(tmp_path);card.update(id='old-PV1',split='TRAIN')
 raw=materialize_rgb(folder,out,card,arrays,lambda:None)
 assert raw==out/'raw'/card['id']
 assert (raw/'body_rgb/0000.png').stat().st_ino==(folder/'body_rgb/0000.png').stat().st_ino
 with pytest.raises(PermissionError):materialize_rgb(folder,out,card,arrays,lambda:None,reuse_registered_source_rgb=True)

def test_changed_frame_or_escape_never_reuses_registered_input(tmp_path):
 folder,out,card,arrays=image_files(tmp_path);arrays['rgb_sha256'][0][0]='0'*64
 with pytest.raises(ValueError):materialize_rgb(folder,out,card,arrays,lambda:None,reuse_registered_source_rgb=True)
 arrays['rgb_paths'][0][0]='../outside.png'
 with pytest.raises(ValueError):materialize_rgb(folder,out,card,arrays,lambda:None,reuse_registered_source_rgb=True)

def test_member_rejects_real_symlink_and_parent_escape(tmp_path):
 file=tmp_path/'regular';file.write_text('fixture');(tmp_path/'link').symlink_to(file)
 assert r.member(tmp_path,'regular')==file
 for path in('link','../regular',str(file)):
  with pytest.raises(PermissionError):r.member(tmp_path,path)

def test_unregistered_relative_entry_refuses_before_neural_or_physics_imports():
 program="from scripts.pi0294.generalization.amd_export_runtime import validate;import sys\nassert 'torch'not in sys.modules and 'mujoco'not in sys.modules\ntry:validate('not-registered')\nexcept PermissionError:pass\nelse:raise RuntimeError('entry incorrectly accepted')\nassert 'torch'not in sys.modules and 'mujoco'not in sys.modules"
 result=subprocess.run([sys.executable,'-c',program],capture_output=True,text=True,timeout=10)
 assert result.returncode==0,result.stderr
