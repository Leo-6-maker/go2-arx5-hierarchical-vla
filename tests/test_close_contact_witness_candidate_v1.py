from pathlib import Path
import pytest
from go2arx5_vla.world_model.close_contact_witness_candidate_v1 import publish


def test_receiver_witness_rejects_wrong_clock_and_unverified_reference(tmp_path):
    e=dict(version='MOBILE_ACCEPTED_BOUNDARY_050_V1',exact_receiver_knot=True,
           native_step=5703,native_time_s=11.406,accepted_pose7=[0,0,.64,0,0,0,1],accepted_wire_sha256='a'*64)
    output=tmp_path/'witness';publish(output,e);original=output.read_bytes()
    with pytest.raises(ValueError):publish(output,dict(e,native_time_s=11.408))
    with pytest.raises(ValueError):publish(output,dict(e,exact_receiver_knot=False))
    assert output.read_bytes()==original
