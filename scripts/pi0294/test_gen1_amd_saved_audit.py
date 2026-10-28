"""Actual temp counter/writer plus original wrapper, not native model evidence."""
import json
import pytest
from scripts.pi0294.generalization.amd_saved_audit import Counter
from scripts.pi0294.pv1.capture import counted_model


def test_original_counter_wrapper_preserves_model_and_stops_before_fifth_call(tmp_path):
    (tmp_path/'RESOURCE_LEDGER.json').write_text(json.dumps(dict(status='RESERVED',engine_starts=0,unknown_effects=[])))
    guards=[];native_calls=[];meter=Counter(tmp_path,lambda:guards.append('guard'))
    class NativeModel:
        @staticmethod
        def from_xml_string(xml):native_calls.append(xml);return NativeModel()
    wrapped=counted_model(NativeModel,meter)
    for _ in range(4):assert type(wrapped.from_xml_string('fixture-not-physical-XML'))is NativeModel
    with pytest.raises(PermissionError):wrapped.from_xml_string('must-not-call')
    assert len(native_calls)==4 and len(guards)==5
    value=json.loads((tmp_path/'RESOURCE_LEDGER.json').read_bytes())
    assert value['engine_starts']==4 and value['status']=='RUNNING'and value['unknown_effects']==[]


@pytest.mark.parametrize('bad',[dict(status='RUNNING',engine_starts=0,unknown_effects=[]),
    dict(status='RESERVED',engine_starts=1,unknown_effects=[]),
    dict(status='RESERVED',engine_starts=0,unknown_effects=['unknown'])])
def test_saved_audit_counter_refuses_replayed_or_unknown_role_before_native_call(tmp_path,bad):
    p=tmp_path/'RESOURCE_LEDGER.json';p.write_text(json.dumps(bad));before=p.read_bytes()
    with pytest.raises(PermissionError):Counter(tmp_path,lambda:None)
    assert p.read_bytes()==before
