"""Real temporary JSON read views; no native audit/model/physics qualification."""
from copy import deepcopy
import json,os
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts.pi0294.generalization import amd_seed_audit as s,amd_capture as c

W=Path(os.environ.get('PI0294_PRIVATE_WORKSPACE',str(Path(__file__).resolve().parents[4])))
P=W/'work/private/science-next/wdr1-target-identity/v4a-core-eval-preparation-01/GEN1_V5_PREVIEW_V2_1792914629517993678.private.json'
if not P.exists():pytest.skip('private frozen packet unavailable',allow_module_level=True)


@pytest.fixture
def files(tmp_path):
    packet=P.read_bytes();card=c.cards(packet,0)[0];folder=tmp_path/'episodes'/card['id'];folder.mkdir(parents=True)
    (folder/'CARD.json').write_text(json.dumps(card));(tmp_path/'ATTEMPT_PLAN.json').write_text(json.dumps(dict(attempts={card['id']:dict(logical_id=card['id'],card=card)})))
    (folder/'samples.json').write_text('[{"actual_label":"unchanged"}]')
    reader=lambda p:json.loads(Path(p).read_bytes());return tmp_path,card,packet,SimpleNamespace(read=reader)


def test_legacy_seed_comparison_passes_without_changing_a_single_original_byte(files):
    root,card,packet,base=files;before={p:p.read_bytes()for p in root.rglob('*')if p.is_file()};read=base.read
    with pytest.raises(KeyError):base.read(root/'episodes'/card['id']/'CARD.json')['seed']
    with s.legacy_identity_read_view(base,root,card,packet)as proof:
        value=base.read(root/'episodes'/card['id']/'CARD.json');binding=base.read(root/'ATTEMPT_PLAN.json')['attempts'][card['id']]
        assert value['seed']==binding['card']['seed']==s.geometry_seed(packet,card)
        assert {k:v for k,v in value.items()if k!='seed'}==card
        assert base.read(root/'episodes'/card['id']/'samples.json')==[dict(actual_label='unchanged')]
        assert not proof['original_files_changed']and not proof['policy_or_supervision_changed']
    assert base.read is read and {p:p.read_bytes()for p in before}==before


@pytest.mark.parametrize('bad',['altered_geometry','different_task','heldout','packet','attempt_binding','nested'])
def test_identity_view_cannot_relabel_other_data_or_hide_a_changed_binding(files,bad):
    root,card,packet,base=files;original=base.read
    if bad=='altered_geometry':card=deepcopy(card);card['source_xy'][0]+=.01
    elif bad=='different_task':card=deepcopy(card);card['task']='T2'
    elif bad=='heldout':card=deepcopy(card);card['study_split']='G_TEST'
    elif bad=='packet':packet+=b' '
    elif bad=='attempt_binding':
        p=root/'ATTEMPT_PLAN.json';value=json.loads(p.read_bytes());value['attempts'][card['id']]['logical_id']='other';p.write_text(json.dumps(value))
    elif bad=='nested':base._GEN1_identity_view_active=True
    with pytest.raises(PermissionError):
        with s.legacy_identity_read_view(base,root,card,packet):pass
    assert base.read is original


def test_numeric_audit_exception_restores_original_reader(files):
    root,card,packet,base=files;original=base.read
    with pytest.raises(RuntimeError):
        with s.legacy_identity_read_view(base,root,card,packet):raise RuntimeError('original numeric rejection')
    assert base.read is original and not hasattr(base,'_GEN1_identity_view_active')
