"""Private fixed packet/captured accounting; all outcomes here are fixtures."""
import base64,json,os
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.generalization import amd_capture as c,amd_adoption as a

W=Path(os.environ.get('PI0294_PRIVATE_WORKSPACE',str(Path(__file__).resolve().parents[4])))
B=W/'work/private/v5-execution-20261025'
P=W/'work/private/science-next/wdr1-target-identity/v4a-core-eval-preparation-01/GEN1_V5_PREVIEW_V2_1792914629517993678.private.json'
if not P.exists():pytest.skip('private frozen input unavailable',allow_module_level=True)


def test_fixed_twelve_raw_ids_keep_gen_lineage_and_original_tasks():
    cards=c.cards(P.read_bytes(),0)+c.cards(P.read_bytes(),1)
    assert len(cards)==len({x['id']for x in cards})==12
    assert len({x['geometry_id']for x in cards})==6
    assert all(x['split']==x['study_split']=='G_TRAIN'and x['visual_profile']=='V0'
        and '/'not in x['id']and not x['group_id'].startswith('PI0294/PV1/')for x in cards)
    assert all(x['nominal_clearance_m']==.004 for x in cards)
    for i in(0,1):assert [x['task']for x in c.cards(P.read_bytes(),i)].count('T1')==3


@pytest.mark.parametrize('index',[True,-1,2])
def test_no_extra_worker_or_resampled_packet(index):
    with pytest.raises(PermissionError):c.cards(P.read_bytes(),index)
    with pytest.raises(PermissionError):c.cards(P.read_bytes()+b' ',0)


@pytest.fixture
def allocation():
    path=B/'GEN1_AMD_STATIC01_GLOBAL_STATIC24_TERMINAL.private.json'
    if not path.exists():pytest.skip('actual static closure not captured')
    g=json.loads(base64.b64decode(json.loads(path.read_bytes())['value']['GLOBAL_base64']))
    workers={str(i):dict(root='/registered/coordinator/children/gen1-amd-teacher-'+str(i).zfill(2),
        run_uuid='gen1-amd-teacher-'+str(i).zfill(2),manifest_sha256=str(i)*64,
        attempt_ids=[x['id']for x in c.cards(P.read_bytes(),i)])for i in(0,1)}
    return g,workers


@pytest.mark.parametrize('bad',[None,'replay','missing_id','wrong_id','bad_static','late'])
def test_captured_static_boundary_and_exact_twelve_first_attempts(allocation,bad):
    g,workers=allocation;now=1793045000.
    if bad=='replay':g[a.FIELD]['teacher_pilot']={}
    elif bad=='missing_id':workers['0']['attempt_ids'].pop()
    elif bad=='wrong_id':workers['1']['attempt_ids'][0]='PI0294/PV1/TRAIN/FAKE'
    elif bad=='bad_static':g[a.FIELD]['static_previews']['actual_integrations']=1
    elif bad=='late':now=g[a.FIELD]['stop_new_training_epoch']-18.25*3600
    call=lambda:a.capture_allocation(g,packet_bytes=P.read_bytes(),workers=workers,static_closure_sha256='a'*64,now_epoch=now)
    if bad is None:
        v=call();assert v['row']['episode_starts']==12 and v['row']['engine_starts']==1008
        assert v['row']['per_cell_task_shadow_constructors']==81 and v['row']['per_cell_readonly_audit_constructors']==3
        assert v['row']['physics_ticks']==384000 and v['row']['Core_CPU_s']==43200
        assert not v['new_GPU']and not v['new_PBS']and v['row']['extra_counter_probes']==0
    else:
        with pytest.raises(PermissionError):call()
