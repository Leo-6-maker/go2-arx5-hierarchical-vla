"""GEN bank/sampling contracts on CPU fixtures, no native exporter/model claim."""
from collections import Counter
from copy import deepcopy
from types import SimpleNamespace
import pytest
from scripts.pi0294.test_gen1_amd_export import P,receipts
from scripts.pi0294.generalization import amd_bank as b,amd_dataset as d,amd_capture as c
from scripts.pi0294.pv1.mix1024 import MixSampler
from scripts.pi0294.pv1.dataset import admit_new_support


def bank_fixture():
 captured,exported=receipts();cards={r['id']:r for i in(0,1)for r in c.cards(P.read_bytes(),i)}
 for row in exported:
  card=cards[row['id']];row.update(split='TRAIN',scene_group=card['group_id'],lineage_id=card['GEN1_lineage_id'],
   compatibility_TRAIN_field_is_not_old_PV1_population=True,nominal_clearance_m=.004,
   file='numeric/'+row['id']+'.npz',sha256='a'*64,source_card_sha256='b'*64,bytes=123,
   official_chain=dict(deterministic_repeat=True,full_641_two_camera_raw_RGB256_verified=True,model_forward=False))
 return b.manifest(P.read_bytes(),captured,exported)


def test_distinct_bank_recomputes_all_original_outcomes_and_cannot_be_old_PV1():
 v=bank_fixture();proof=b.validate(v,P.read_bytes());assert len(proof['admitted'])==12 and not proof['native_training_admission']
 assert len(v['capture_outcomes'])==12 and not v['old_COV1_population_changed']
 with pytest.raises(PermissionError):admit_new_support(v,'COVERAGE_FIRST')


@pytest.mark.parametrize('bad',['R0','DEV','V1','codec','parent','raw_reuse','lineage','scene_group','numeric','outcomes','injected','chain'])
def test_wrong_bank_cannot_enter_new_reader(bad):
 v=bank_fixture();row=v['episodes'][0]
 if bad=='R0':v['statistics_sha256']='f'*64
 elif bad=='DEV':row['study_split']='G_TEST'
 elif bad=='V1':row['visual_profile']='V1'
 elif bad=='codec':row['codec']['native_decode_unchanged']=False
 elif bad=='parent':row['id']='foreign'
 elif bad=='raw_reuse':row['source_raw_manifest_sha256']=v['episodes'][1]['source_raw_manifest_sha256']
 elif bad=='lineage':row['lineage_id']='another'
 elif bad=='scene_group':row['scene_group']='old-PV1-group'
 elif bad=='numeric':row['file']='../escape.npz'
 elif bad=='outcomes':v['capture_outcomes'].pop()
 elif bad=='injected':v['later_rows_hot_injected']=True
 else:row['official_chain']['model_forward']=True
 with pytest.raises((PermissionError,KeyError)):b.validate(v,P.read_bytes())


def fixture_reader():
 rows=[]
 for lane in ['original_T1','original_T2','C1_T1','new_T1','new_T2']:
  for parent in range(4):
   row=dict(id=lane+'-'+str(parent),split='TRAIN',task=lane[-2:],PV1_lane=lane,scene_group='GEN1-geometry-'+str(parent))
   rows.extend((row,anchor)for anchor in range(601))
 return SimpleNamespace(rows=rows,manifest_sha='f'*64)


def test_all_32768_selections_match_frozen_C50_math_with_explicit_GEN_state():
 data=fixture_reader();gen=d.Sampler(data);old=MixSampler(data,arm='C50');counts=Counter()
 for cursor in range(d.VISITS):
  selected=gen.selection_at(cursor);assert selected==old.selection_at(cursor);counts[selected['lane']]+=1
 assert counts==dict(original_T1=4096,original_T2=8192,C1_T1=8192,new_T1=4096,new_T2=8192)
 state=gen.state_dict(d.VISITS);assert state['training_identity']==b.IDENTITY and 'arm'not in state
 assert state['cursor']==32768 and state['algorithm'].startswith('GEN1_')
 with pytest.raises(PermissionError):gen.state_dict(1)
 resumed=d.Sampler(data,start=512*32)
 for cursor in range(512*32,512*32+64):assert resumed.selection_at(cursor)==gen.selection_at(cursor)


@pytest.mark.parametrize('bad',[dict(seed=20261020),dict(stop=4096*32),dict(start=-1)])
def test_other_seed_or_extra_training_rejected_before_iteration(bad):
 with pytest.raises(PermissionError):d.Sampler(fixture_reader(),**bad)
