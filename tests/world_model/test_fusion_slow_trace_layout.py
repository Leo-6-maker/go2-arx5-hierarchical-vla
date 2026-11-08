import struct
import pytest

from go2arx5_vla.world_model.init_trace_reader import read_trace, RECORD


@pytest.mark.parametrize('capacity',[262144,524288])
def test_supported_slow_layout_retains_the_same_record_abi_and_sequence(tmp_path,capacity):
    path=tmp_path/'trace'
    row=RECORD.pack(0,1,0.,17,0,0,0,0,0,0,*([0.]*96))
    path.write_bytes(b'WM02IT01'+struct.pack('<II',RECORD.size,capacity)+row)
    rows=read_trace(path)
    assert len(rows)==1 and rows[0]['kind']==17 and rows[0]['ticket']==0


def test_neighbouring_unregistered_trace_capacity_is_still_rejected(tmp_path):
    path=tmp_path/'trace'
    path.write_bytes(b'WM02IT01'+struct.pack('<II',RECORD.size,262145))
    with pytest.raises(ValueError,match='Unknown trace layout'):read_trace(path)
