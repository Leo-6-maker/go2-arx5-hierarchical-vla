"""Run the actual C++ receiver state machine in a tiny CPU executable with no simulator."""

import os
import struct
import subprocess
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
EPOCH = b"7" * 64
PACK = struct.Struct("<64s8sQddd")
HARNESS = r"""
#include "student_receiver.hpp"
#include <iostream>
int main(int argc,char** argv) {
 if(argc!=2)return 2;
 student_grip::Session s;
 double x[2]={.042,.042};
 s.reset(argv[1],std::string(64,'7').c_str(),6503,x);
 unsigned long long tick;
 while(std::cin>>tick){
   const bool ok=s.receive(tick,.002*tick,x,x);
   if(!ok)s.stop("RECEIVER_REJECT_STOP",tick,x,x);
   std::cout<<int(ok)<<std::endl;
 }
}
"""


@pytest.fixture(scope="module")
def binary(tmp_path_factory):
    d = tmp_path_factory.mktemp("receiver-build")
    src = d / "test.cc"
    src.write_text(HARNESS)
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-O2",
            "-I",
            str(ROOT / "scripts/sla_hierarchical_v16"),
            str(src),
            "-o",
            str(d / "test"),
        ],
        check=True,
    )
    return d / "test"


def packet(seq=0, t=5.006, width=0.042, epoch=EPOCH):
    return PACK.pack(epoch, b"GRTGRIP1", seq, t, width / 0.044, t + 0.399)


@pytest.mark.parametrize(
    "variant",
    [
        "ok",
        "wrong_epoch",
        "wrong_sequence",
        "too_fast",
        "changed_duplicate",
        "expired",
        "unanchored",
        "missing_after",
        "stale_time",
        "malformed",
    ],
)
def test_compiled_receipts(binary, tmp_path, variant):
    with subprocess.Popen(
        [str(binary), str(tmp_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    ) as proc:

        def call(t, data):
            f = tmp_path / "command.bin"
            if data is not None:
                f.write_bytes(data)
                os.chmod(f, 0o600)
            elif f.exists():
                f.unlink()
            proc.stdin.write(str(t) + "\n")
            proc.stdin.flush()
            return proc.stdout.readline().strip() == "1"

        assert call(0, None)
        if variant == "unanchored":
            assert not call(2503, packet(width=0.040))
        else:
            assert call(2503, packet())
            choice = {
                "ok": packet(1, 5.056, 0.040),
                "wrong_epoch": packet(1, 5.056, epoch=b"1" * 64),
                "wrong_sequence": packet(2, 5.056),
                "too_fast": packet(1, 5.056, 0.0399),
                "changed_duplicate": packet(0, 5.006, 0.041),
                "expired": packet(1, 4.0),
                "missing_after": None,
                "stale_time": packet(1, 5.0),
                "malformed": b"bad",
            }
            assert call(2528, choice[variant]) == (variant == "ok")
        proc.stdin.close()
        assert proc.wait(timeout=4) == 0
    rows = (tmp_path / "receiver.csv").read_text().splitlines()
    assert rows[0].startswith("PHYSICAL_RESET,")
    if variant != "ok":
        assert rows[-1].startswith("RECEIVER_REJECT_STOP,")
