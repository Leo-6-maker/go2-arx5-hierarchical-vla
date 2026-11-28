"""Verify packet continuity before a Recovery teacher ships its first atom."""

import json
import hashlib
from pathlib import Path
import struct


PACKET = struct.Struct("<8sQddd")
DT = .002
ORIGINAL_EARLY_ENTRY = "f14aa14ce9f4f4400ccc274598e15b45cd40a4b6a2041217c0e43da62effaf56"


def previous_sequence(samples, packet, native_step, native_time_s):
    """Bind the next command to the last actual 50 ms acknowledgement and wire."""
    with samples.open("rb") as stream:
        stream.seek(max(0, samples.stat().st_size-16384))
        last = json.loads(stream.read().splitlines()[-1])
    magic, sequence, source, width, until = PACKET.unpack(packet.read_bytes())
    command = last["gripper_command"]
    if (magic != b"GRTGRIP1" or sequence != command["sequence"]
            or last["native_start_step"] != native_step-25 or last["native_end_step"] != native_step
            or last["ticks"] != 25 or not last["safety_valid"] or source != command["source_native_s"]
            or abs(native_time_s-source-25*DT)>1.e-8 or abs(until-source-.4)>1.e-8
            or abs(width-last["installed_EEF10"][9])>1.e-12):
        raise ValueError("The live gripper wire disagrees with the last acknowledged complete atom")
    return sequence


def open_recovery_sequence(adapter, native_step, native_time_s):
    from go2arx5_vla.world_model.native_live_control_v3 import live_path

    return previous_sequence(Path(adapter.output)/"MOBILE_SAMPLES.private.jsonl",
                             live_path(adapter,"gripper.bin"),native_step,native_time_s) + 1


def stage_continuous(source, destination):
    content = source.read_bytes()
    if hashlib.sha256(content).hexdigest() != ORIGINAL_EARLY_ENTRY:
        raise ValueError("The Recovery entry differs from the closed early pair")
    code = content.decode()
    edits = (
        ('    reference = adapter.reference_history[-1]["pose"].copy()\n'
         '    write(adapter.output/"RECOVERY-HANDOFF-INTENT.private.json", dict(prepared["event"],\n',
         '    from v7_recovery_sequence import open_recovery_sequence\n'
         '    reference = adapter.reference_history[-1]["pose"].copy()\n'
         '    first_sequence = None\n'
         '    if prepared["event"]["trigger"] == "EARLY_UNINSTALLED_G7":\n'
         '        first_sequence = open_recovery_sequence(adapter, prepared["event"]["native_step"],\n'
         '                                               prepared["event"]["native_time_s"])\n'
         '    write(adapter.output/"RECOVERY-HANDOFF-INTENT.private.json", dict(prepared["event"],\n'
         '          first_teacher_gripper_sequence=first_sequence,\n'),
        ('                sample, _ = consume(adapter, preview, sequence=128+query*10+i)\n',
         '                sequence = 128 + query*10 + i\n'
         '                if first_sequence is not None:\n'
         '                    sequence = first_sequence + consumed_atoms\n'
         '                sample, _ = consume(adapter, preview, sequence=sequence)\n'),
    )
    for old,new in edits:
        if code.count(old) != 1:
            raise ValueError("The original handoff seam changed")
        code = code.replace(old,new)
    compile(code,str(destination),"exec")
    with destination.open("x") as f:
        f.write(code)
