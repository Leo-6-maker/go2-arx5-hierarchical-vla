"""Actual CPU tensor collator fixture plus mocked reader, not official PASS."""
import json
from pathlib import Path
import pytest
from scripts.pi0294.fd1 import put,sha
from scripts.pi0294.data_cov1 import mixed_batch as m
from scripts.pi0294.test_data_cov1_mixed_batch import Processor

def fixture(tmp_path):
 bindings=[];roots={}
 for i in range(2):
  identifier='GEN1-PILOT-'+str(i)*16+'-T1';out=tmp_path/('parent'+str(i));out.mkdir();(out/'numeric').mkdir();p=out/'numeric'/(identifier+'.npz');p.write_bytes(b'IO-fixture-not-native-numeric');p.chmod(0o400);put(out/'statistics.json',dict(CPU_fixture=True))
  row=dict(id=identifier,admitted_for_training=True,status='PASS_SAVED_TEACHER_EXPORT_AND_OFFICIAL_CHAIN',codec=dict(status='PASS_FIXED_ORIGINAL_V3_0_NO_SILENT_CLIP'),split='TRAIN',study_split='G_TRAIN',nominal_clearance_m=.004,full_task_success=True,independent_raw_admitted=True,file='numeric/'+identifier+'.npz',sha256=sha(p),source_raw_manifest_sha256=str(i)*64,RGB_materialization='EXPLICIT_GEN1_REGISTERED_IMMUTABLE_SOURCE_INPUT')
  put(out/'EXPORT_RECEIPT.json',row);put(out/'DATASET_MANIFEST.json',dict(episodes=[row]));bindings.append(dict(id=identifier,export_root=str(out),export_receipt_sha256=sha(out/'EXPORT_RECEIPT.json')))
  raw=tmp_path/'registered-inputs'/identifier;raw.mkdir(parents=True);roots[identifier]=str(raw)
 return bindings,roots,sha(tmp_path/'parent0/statistics.json')

def test_all_six_anchors_per_parent_use_explicit_source_binding(monkeypatch,tmp_path):
 bindings,roots,stats=fixture(tmp_path);seen=[]
 class Reader:
  def __init__(self,root,*args,**kwargs):self.root=Path(root);seen.append(kwargs['raw_bindings'])
  def __len__(self):return 601
  def __getitem__(self,index):return dict(processed=(self.root.name,index))
 monkeypatch.setattr(m,'NativeDataset',Reader);monkeypatch.setattr(m,'STATS',stats);processor=Processor()
 result=m.qualify_all(bindings,processor,'fixture-source','fixture-cosmos',registered_GEN1_raw_roots=roots)
 assert seen==[{b['id']:roots[b['id']]}for b in bindings]
 assert result['distinct_parent_count']==2 and result['maximum_samples_per_batch']==12
 assert processor.calls[0]==processor.calls[1]and len(processor.calls[0])==12
 assert not any((Path(b['export_root'])/'raw').exists()for b in bindings)

def test_missing_mapping_or_old_population_cannot_use_new_rgb_mode(monkeypatch,tmp_path):
 bindings,roots,stats=fixture(tmp_path);seen=[];monkeypatch.setattr(m,'NativeDataset',lambda *a,**k:seen.append(k));monkeypatch.setattr(m,'STATS',stats)
 with pytest.raises(PermissionError):m.qualify_all(bindings,Processor(),'fixture','fixture',registered_GEN1_raw_roots={})
 path=Path(bindings[0]['export_root'])/'EXPORT_RECEIPT.json';row=json.loads(path.read_bytes());row['study_split']='TRAIN';path.unlink();put(path,row);bindings[0]['export_receipt_sha256']=sha(path)
 with pytest.raises(PermissionError):m.qualify_all(bindings,Processor(),'fixture','fixture',registered_GEN1_raw_roots=roots)
 assert not seen
