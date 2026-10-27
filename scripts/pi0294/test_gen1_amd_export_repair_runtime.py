"""Two actual temporary roots and full source/R0 reader, not native admission."""
import base64,copy,hashlib,io,json,os,sys,tarfile
from pathlib import Path
import pytest
from scripts.pi0294.generalization import amd_export_runtime as original,amd_export_repair_runtime as repair,amd_export_repair_accounting as accounting

B=Path(__file__).resolve().parents[4]/'work/private/v5-execution-20261025'
def raw(v):return(json.dumps(v,indent=2)+'\n').encode()
def put(p,b):p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
def digest(b):return hashlib.sha256(b).hexdigest()

@pytest.fixture
def data(tmp_path,monkeypatch):
 p=B/'GEN1_VANDA_OFFICIAL_EXPORT_PREPARED.private.json'
 if not p.exists():pytest.skip('private fixed packet/source overlay inputs unavailable')
 prepared=json.loads(p.read_bytes());r=tmp_path/'campaign/children/repair';old=r.with_name('old');code=r/'capsules/code';files={};seal=copy.deepcopy(prepared['seal'])
 with tarfile.open(fileobj=io.BytesIO(base64.b64decode(prepared['source_archive_base64'])))as archive:
  for name in seal['frozen15']:files[name]=archive.extractfile(name).read()
 runtime='scripts/pi0294/generalization/amd_export_runtime.py';files[runtime]=Path(original.__file__).read_bytes()
 for name,b in files.items():put(code/name,b)
 seal['files']={name:digest(b)for name,b in files.items()};seal['source_sha']='a'*40;put(r/'SOURCE_SEAL.json',raw(seal))
 monkeypatch.setattr(original,'__file__',str(code/runtime));monkeypatch.setattr(original,'ORIGINAL_PYTHON',sys.executable);monkeypatch.setattr(original,'python_environment',lambda *a,**k:None)
 for k,v in dict(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1').items():monkeypatch.setenv(k,v)
 plan=json.loads(base64.b64decode(prepared['small']['plans/GEN1_SAVED_EXPORT_PLAN.private.json']));members=[]
 for case in plan['cases']:
  folder=old/'inputs'/case['id'];folder.mkdir(parents=True);case['source_episode']=str(folder);audit=folder/'audit.json';audit.write_bytes(b'fixture-only-not-a-native-raw-audit');case['audit_file']=str(audit);members.append(dict(path=str(audit.relative_to(old)),bytes=audit.stat().st_size,sha256=original.sha(audit)))
 closed=json.loads(base64.b64decode(prepared['small']['plans/ALL_CAPTURE_CLOSED.private.json']));inputs=dict(official_source='fixture-upstream-not-executed',cosmos='fixture-cosmos-not-model');cosmos=tmp_path/'cosmos-config.json';cosmos.write_bytes(b'fixture-processor-input')
 official=dict(inputs,official_source_git_sha='51d4c89f72fda44cbf77285c6a8114b52676b8a1',cosmos_files=[dict(path=str(cosmos),resolved_path=str(cosmos.resolve()),bytes=cosmos.stat().st_size,sha256=original.sha(cosmos))])
 small={'INPUT_BINDINGS.private.json':raw(inputs),'plans/GEN1_SAVED_EXPORT_PLAN.private.json':raw(plan),'plans/ALL_CAPTURE_CLOSED.private.json':raw(closed),'plans/GEN1_V5_PREVIEW_V2.private.json':base64.b64decode(prepared['packet_base64']),'plans/statistics.json':base64.b64decode(prepared['small']['plans/statistics.json']),'plans/OFFICIAL_CPU_INPUT_IDENTITIES.private.json':raw(official)}
 row=dict(source_sha='a'*40,closed_capture_proof_sha256=digest(small['plans/ALL_CAPTURE_CLOSED.private.json']),raw_plan_sha256=digest(small['plans/GEN1_SAVED_EXPORT_PLAN.private.json']))
 small['plans/DELIVERED_RAW_MEMBERS.private.json']=raw(dict(status='PASS_ORIGINAL_CLOSED_SOURCE_BYTES_DELIVERED_FULL_SHA',closed_capture_proof_sha256=row['closed_capture_proof_sha256'],files=members))
 for name,b in small.items():put(r/name,b)
 m=dict(scope=accounting.VERSION,code=dict(capsule_relative='capsules/code',vla_commit='a'*40,source_seal_sha256=original.sha(r/'SOURCE_SEAL.json')),execution=dict(sealed_inputs={k:digest(b)for k,b in small.items()}))
 put(r.parent.parent/'HOPPER_RESOURCE_LEDGER.json',b'{}');monkeypatch.setattr(accounting,'rows',lambda g:[row]);monkeypatch.setattr(original.a,'validate',lambda g:{accounting.FIELD:dict(old_source_root=str(old))})
 return r,old,m,row

def test_full_original_reader_uses_closed_source_root_without_copy_or_relative_path_bug(data):
 r,old,m,row=data;before={str(p):p.read_bytes()for p in old.rglob('*')if p.is_file()}
 plan,packet,inputs=original.validate_source_inputs(r,m,row,input_source_root=old)
 assert len(plan['cases'])==11 and not(r/'inputs').exists()
 assert before=={str(p):p.read_bytes()for p in old.rglob('*')if p.is_file()}
 assert 'mujoco'not in sys.modules

@pytest.mark.parametrize('bad',['source','raw','extra','escape','wrong_root','R0'])
def test_source_raw_set_or_registered_root_mutation_rejects(data,bad):
 r,old,m,row=data
 if bad=='source':(r/'capsules/code/scripts/pi0292/train_adapter.py').write_bytes(b'changed')
 elif bad=='raw':next((old/'inputs').rglob('audit.json')).write_bytes(b'changed')
 elif bad=='extra':put(old/'inputs/hidden',b'unregistered')
 elif bad=='escape':next((old/'inputs').rglob('audit.json')).unlink();next((old/'inputs').iterdir()).joinpath('audit.json').symlink_to('/etc/hosts')
 elif bad=='wrong_root':old=old.with_name('anonymous-other-root')
 else:(r/'plans/statistics.json').write_bytes(b'changed')
 with pytest.raises((PermissionError,FileNotFoundError)):original.validate_source_inputs(r,m,row,input_source_root=old)

def test_aggregate_counts_both_roots_and_rejects_hidden_links(tmp_path,monkeypatch):
 a=tmp_path/'old';b=tmp_path/'new';a.mkdir();b.mkdir();(a/'retained').write_bytes(b'12345');(b/'numeric').write_bytes(b'1234567')
 assert repair.aggregate_output(b,a)['total_bytes']==12
 monkeypatch.setattr(accounting,'METADATA',1);monkeypatch.setitem(accounting.LIMITS,'shared_previous_paid_output_upper_bytes',10)
 with pytest.raises(PermissionError):repair.aggregate_output(b,a)
 (b/'numeric').unlink();(b/'link').symlink_to(a/'retained')
 with pytest.raises(PermissionError):repair.aggregate_output(b,a)

def test_actual_isolated_tool_child_preserves_parent_loader_and_pins_all_files(tmp_path,monkeypatch):
    git=tmp_path/'tools/git';git.parent.mkdir();git.write_text('#!/bin/sh\nprintf "git version 2.40.0\\n"\n');git.chmod(0o700)
    module=tmp_path/'module';module.write_text('fixture PATH-only module')
    monkeypatch.setattr(repair,'GIT',str(git));monkeypatch.setattr(repair,'MODULE',str(module))
    monkeypatch.setattr(repair,'GIT_SHA',repair.sha(git));monkeypatch.setattr(repair,'MODULE_SHA',repair.sha(module))
    monkeypatch.setenv('PATH',os.environ['PATH']);monkeypatch.setenv('GIT_OPTIONAL_LOCKS','1');monkeypatch.setenv('LD_LIBRARY_PATH','fixture-parent-loader')
    v=repair.tools(run_git=True);assert v['actual_git_executed']and os.environ['LD_LIBRARY_PATH']=='fixture-parent-loader'
    assert os.environ['PATH'].split(os.pathsep)[0]==str(git.parent)and os.environ['GIT_OPTIONAL_LOCKS']=='0'
    before=os.environ['PATH'];git.write_text('changed')
    with pytest.raises(PermissionError):repair.tools()
    assert os.environ['PATH']==before
