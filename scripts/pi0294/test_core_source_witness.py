"""Kernel file custody and original remote-prefix tests; no SSH/native GO."""
import base64,json,os,stat,time
from pathlib import Path
import pytest
from scripts.pi0294.pv1 import core_source_witness as w,evaluation_core_ledger as ledger

@pytest.fixture
def capsule(tmp_path):
 root=tmp_path/'source';code=root/'code';code.mkdir(parents=True)
 files={'a.py':b'answer=7\n',w.HELPER:Path(w.__file__).read_bytes()}
 for name,data in files.items():
  p=code/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);p.chmod(0o400)
 seal=root/'SOURCE_SEAL.json';seal.write_bytes(w.raw(dict(source_sha='a'*40,files={n:w.sha(b)for n,b in files.items()})));seal.chmod(0o400)
 for path in sorted(code.rglob('*'),reverse=True):
  if path.is_dir():path.chmod(0o500)
 code.chmod(0o500);time.sleep(2.05);pin=w.sha(seal.read_bytes());certificate=w.raw(w.make(code,seal,expected_seal_sha256=pin))
 try:yield code,seal,pin,certificate
 finally:
  for path in [code,*code.rglob('*')]:
   if path.is_dir()and not path.is_symlink():path.chmod(0o700)

def verify(capsule):
 code,seal,pin,cb=capsule
 return w.verify(code,seal,cb,expected_certificate_sha256=w.sha(cb),expected_seal_sha256=pin)

def test_full_hash_then_actual_kernel_identity_no_model(capsule):
 proof=verify(capsule);assert proof['verified_files']==2 and proof['read_only']
 code,seal,pin,cb=capsule
 p=code/'a.py';p.chmod(0o600);p.write_bytes(b'answer=8\n');p.chmod(0o400)
 with pytest.raises(PermissionError):verify(capsule)

@pytest.mark.parametrize('change',['rewrite_restore_mtime','replace_identical','symlink','permission','directory','seal'])
def test_each_source_identity_change_rejected(capsule,change):
 code,seal,pin,cb=capsule;p=code/'a.py';s=p.stat()
 if change=='rewrite_restore_mtime':
  p.chmod(0o600);p.write_bytes(b'answer=8\n');os.utime(p,ns=(s.st_atime_ns,s.st_mtime_ns));p.chmod(0o400)
 elif change=='replace_identical':
  code.chmod(0o700);data=p.read_bytes();p.unlink();p.write_bytes(data);p.chmod(0o400);code.chmod(0o500)
 elif change=='symlink':
  code.chmod(0o700);p.unlink();p.symlink_to(seal);code.chmod(0o500)
 elif change=='permission':p.chmod(0o600)
 elif change=='directory':code.chmod(0o700)
 else:seal.chmod(0o600);seal.write_bytes(seal.read_bytes()+b' ');seal.chmod(0o400)
 with pytest.raises((PermissionError,FileNotFoundError)):verify(capsule)

def test_forged_certificate_or_missing_full_hash_not_accepted(capsule):
 code,seal,pin,cb=capsule;value=json.loads(cb);value['complete_full_hash_performed']=False;bad=w.raw(value)
 for data,expected in [(bad,w.sha(cb)),(bad,w.sha(bad))]:
  with pytest.raises(PermissionError):w.verify(code,seal,data,expected_certificate_sha256=expected,expected_seal_sha256=pin)

def test_certificate_is_explicit_new_request_not_legacy_flag():
 m={'execution':{'sealed_inputs':{}}};assert w.request_contract(m,{})is None
 with pytest.raises(PermissionError):w.request_contract(m,{w.FIELD:'b'*64,'evaluation_candidate':'C50'})
 m['execution']['sealed_inputs'][w.CERTIFICATE]='b'*64
 assert w.request_contract(m,{w.FIELD:'b'*64,'evaluation_candidate':'C50'})=='b'*64
 with pytest.raises(PermissionError):w.request_contract(m,{w.FIELD:'b'*64,'evaluation_candidate':'A_CONTROL'})
 with pytest.raises(PermissionError):w.request_contract(m,{})

def test_real_remote_source_prefix_requires_registered_certificate(capsule,tmp_path):
 code,seal,pin,cb=capsule;parent=tmp_path/'campaign';parent.mkdir();rid='fixture'
 row=dict(request_sha256='b'*64,source_sha='a'*40,remote_source_custody_sha256=w.sha(cb))
 path=parent/'HOPPER_RESOURCE_LEDGER.json';path.write_bytes(w.raw({'V5_CD_Core_allocations':{rid:row}}))
 before=path.read_bytes();binding=dict(remote_code=str(code),remote_seal=str(seal),remote_seal_sha256=pin,canonical_parent=str(parent))
 payload=dict(binding=binding,source_sha='a'*40,request_id=rid,request_sha256='b'*64,source_certificate_base64=base64.b64encode(cb).decode())
 prefix=ledger.REMOTE.split('sys.path[:0]=',1)[0]
 host="assert os.getuid()==22182 and os.environ.get('USER')=='e1192548' and not os.environ.get('PBS_JOBID')"
 assert host in prefix;prefix=prefix.replace(host,'assert not os.environ.get("PBS_JOBID")')
 namespace={'P':payload};exec(prefix,namespace);assert path.read_bytes()==before
 row['remote_source_custody_sha256']='0'*64;path.write_bytes(w.raw({'V5_CD_Core_allocations':{rid:row}}))
 with pytest.raises(AssertionError):exec(prefix,{'P':payload})

def test_new_certificate_keeps_original_once_debit_and_lost_reply_latch(capsule,tmp_path,monkeypatch):
 import sys,gzip
 capture=os.environ.get('ARX5_FIRST512_MEASUREMENT_CAPTURE')
 if not capture:pytest.skip('Captured complete GLOBAL required for original validator/onceIO subprocess test')
 captured=json.loads(Path(capture).read_bytes());g=json.loads(gzip.decompress(base64.b64decode(captured['GLOBAL_gzip_base64'])))
 from scripts.pi0294.pv1.partial_panel import FAILED
 r=g['V5_CD_Core_allocations'][FAILED]
 assert r['status']in('BOUND_QUALIFICATION','NATIVE_RUNNING')
 from scripts.pi0294.pv1 import mix_core_accounting as c,mix_accounting as a
 from scripts.pi0294.dec1.owner_transport import Journal,UnknownMutation
 code,seal,pin,oldcb=capsule
 # A fixture seal identifies this fixture's existing C/D source. Reaudit only
 # after its real filesystem timestamp has closed; no live capsule is edited.
 seal.chmod(0o600);doc=json.loads(seal.read_bytes());doc['source_sha']=r['source_sha'];seal.write_bytes(w.raw(doc));seal.chmod(0o400);time.sleep(2.05)
 pin=w.sha(seal.read_bytes());cb=w.raw(w.make(code,seal,expected_seal_sha256=pin))
 actor=os.getpid();start=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]);r.update(pid=actor,proc_start=start,native_handle=c.process_handle(r['boot_id'],actor,start))
 r[w.FIELD]=w.sha(cb);c.checked(g);oldq,oldloads,_=c.progress(r)
 parent=tmp_path/r['campaign_run_uuid'];parent.mkdir();path=parent/'HOPPER_RESOURCE_LEDGER.json';path.write_bytes(a.raw(g))
 root=tmp_path/r['child_run_uuid'];(root/'evidence').mkdir(parents=True);obj=ledger.Ledger.__new__(ledger.Ledger)
 obj.root=root;obj.actor_pid=actor;obj.actor_start=start;obj.request=r;obj.request_sha=r['request_sha256'];obj.native_handle=r['native_handle'];obj.sequence=0;obj.path=root/'evidence/SNAPSHOT.json';obj.accounting=c;obj.source_certificate=cb
 obj.binding=dict(remote_code=str(code),remote_seal=str(seal),remote_seal_sha256=pin,canonical_parent=str(parent));(root/'rpc').mkdir();obj.journal=Journal(root/'rpc',dict(run_uuid=root.name,job_id=obj.native_handle,source_sha=r['source_sha'],request_sha256=r['request_sha256']))
 host="assert os.getuid()==22182 and os.environ.get('USER')=='e1192548' and not os.environ.get('PBS_JOBID')";assert host in ledger.REMOTE
 monkeypatch.setattr(ledger,'REMOTE',ledger.REMOTE.replace(host,'assert not os.environ.get("PBS_JOBID")'))
 obj._argv=lambda:[sys.executable,'-c',"import sys;exec(compile(sys.stdin.read(),'<temp-once-program>','exec'))"]
 loaded=obj._rpc('DEBIT',kind='model_loads',count=1);assert loaded[c.FIELD][r['request_id']]['charged_model_loads']==oldloads+1
 obj._argv=lambda:[sys.executable,'-c',"import sys;exec(compile(sys.stdin.read(),'<temp-once-program>','exec'));sys.exit(7)"]
 with pytest.raises(Exception):obj._rpc('DEBIT',kind='generation_queries',count=1)
 after=path.read_bytes();assert json.loads(after)[c.FIELD][r['request_id']]['charged_query_attempts']==oldq+1
 assert obj.journal._state()['status']=='UNKNOWN'
 with pytest.raises(UnknownMutation):obj._rpc('DEBIT',kind='generation_queries',count=1)
 assert path.read_bytes()==after
