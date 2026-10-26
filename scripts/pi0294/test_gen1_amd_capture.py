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


def test_terminal_cpu_accounting_uses_original_meter_without_duplicate_keyword(tmp_path):
    """Actual original durable writer, never a native physics fixture."""
    import ast,time
    from types import SimpleNamespace
    from scripts.pi0294.pv1.capture import Meter
    tree=ast.parse(Path(c.__file__).read_text())
    terminal=[node for node in ast.walk(tree)if isinstance(node,ast.Call)
        and isinstance(node.func,ast.Attribute)and node.func.attr=='progress'
        and any(k.arg=='source_sha'for k in node.keywords)]
    assert len(terminal)==1 and 'worker_CPU_s'not in {k.arg for k in terminal[0].keywords}
    meter=SimpleNamespace(root=tmp_path,ledger=dict(attempted_cells=['fixed-first-attempt']),
        ticks=700,live_ticks=300,shadow_ticks=400,starts=7,started=time.process_time())
    Meter.progress(meter,status='TERMINAL_KNOWN',CPU_s=12.5,source_sha='cpu-fixture')
    value=json.loads((tmp_path/'RESOURCE_LEDGER.json').read_bytes())
    assert value['CPU_s']==12.5 and value['worker_CPU_s']>=0
    assert value['ticks']==700 and value['live_ticks']==300 and value['episode_starts']==1


@pytest.mark.parametrize('phase',['task','negative','audit'])
def test_actual_original_meter_is_charged_for_all_three_finite_phases(phase):
    import ast,time
    from scripts.pi0294.pv1 import capture as original
    # Execute the actual nested class, with no constructor/model/physics import.
    node=next(x for x in ast.walk(ast.parse(Path(c.__file__).read_text()))
              if isinstance(x,ast.ClassDef)and x.name=='Meter')
    namespace=dict(original=original,a=a,time=time)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),
                 '<actual-cold-counter-class>','exec'),namespace)
    m=namespace['Meter'].__new__(namespace['Meter'])
    m.cell='fixture';m.cell_starts=m.starts=m.cell_audit_starts=m.audit_starts=0
    m.cell_negative_starts=m.negative_starts=0;m.negative_phase=m.audit_phase=False
    m.m=dict(limits=dict(engine_starts=528));m.guard=lambda:None
    m.start()  # one real-task constructor
    m.negative_phase=True
    for _ in range(4):m.start()
    if phase=='negative':
        with pytest.raises(PermissionError):m.start()
    m.negative_phase=False
    for _ in range(80):m.start()  # original eighty prefix clones
    if phase=='task':
        with pytest.raises(PermissionError):m.start()
    m.audit_phase=True
    for _ in range(3):m.start()
    if phase=='audit':
        with pytest.raises(PermissionError):m.start()
    assert m.starts==88 and m.negative_starts==4 and m.audit_starts==3
    assert c.COUNTER_POLICY=='GEN1_ORIGINAL_TASK81_NEGATIVE4_RAWAUDIT3_V2'


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
