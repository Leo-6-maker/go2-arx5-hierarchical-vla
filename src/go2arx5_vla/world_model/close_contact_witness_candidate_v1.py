"""Future-profile witness writer; not connected to any existing native runner."""
import math
import os
from pathlib import Path
import re


def publish(path, evidence):
    if (evidence['version'] != 'MOBILE_ACCEPTED_BOUNDARY_050_V1'
            or evidence['exact_receiver_knot'] is not True):
        raise ValueError('EXACT_EXISTING_RECEIVER_BOUNDARY_REQUIRED')
    tick, seconds = evidence['native_step'], evidence['native_time_s']
    z, wire = evidence['accepted_pose7'][2], evidence['accepted_wire_sha256']
    if (type(tick) is not int or tick < 5703 or not math.isfinite(seconds)
            or not math.isfinite(z) or abs(seconds-.002*tick) > 1e-8
            or not re.fullmatch('[0-9a-f]{64}', wire)):
        raise ValueError('EXACT_NATIVE_CLOCK_AND_WIRE_ID_REQUIRED')
    path = Path(path)
    data = f'SLA_CLOSE_EXACT_RECEIVER_BOUNDARY_V1\n{tick} {seconds:.17g} {z:.17g} {wire}\n'.encode()
    temporary = path.with_name(path.name + '.pending.' + str(os.getpid()))
    with temporary.open('xb') as handle:
        handle.write(data); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)
    return dict(path=str(path),native_step=tick,reference_is_receiver_not_selected_policy=True,
                candidate_is_installed=False,independent_admission=False)
