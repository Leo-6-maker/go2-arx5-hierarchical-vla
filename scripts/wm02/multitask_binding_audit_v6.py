"""Cold compilation and recorded-state geometry audit for the two-object ABI."""
import ctypes
from collections import Counter
from itertools import product
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import time
import traceback

import numpy as np

from scripts.pi0292.durable import write
from go2arx5_vla.world_model.storage import digest


def pinned(binding):
    path = Path(binding['path'])
    if digest(path) != binding['sha256']:
        raise ValueError('BOUND_MULTITASK_AUDIT_INPUT_CHANGED:' + path.name)
    return path


def compile_candidate(build, output, name, edit):
    original = pinned(build['source'])
    pinned(build['library'])
    pinned(build['trace_header'])
    pinned(build['trace_library'])
    pinned(build['compiler_commands'])
    for header in build.get('additional_headers',()):
        pinned(header)
    argv = list(build['argv'])
    source = output / (name+'.cpp')
    library = output / (name+'.so')
    source.write_bytes(edit(original.read_bytes()))
    argv[argv.index(str(original))] = str(source)
    if argv[-2:] != ['-o', build['library']['path']]:
        raise ValueError('ARCHIVED_OBSERVER_COMPILER_OUTPUT_MISMATCH')
    argv[-1] = str(library)
    with (output / (name+'-compile.log')).open('x') as log:
        subprocess.run(argv, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=90)
    loaded = ctypes.CDLL(str(library), mode=ctypes.RTLD_LOCAL)
    return loaded, {'source_sha256': digest(source), 'library_sha256': digest(library),
                    'frozen_parent_sha256': digest(original)}


def build_observer(spec, output):
    from go2arx5_vla.world_model.multitask_contact_v6 import observer_source

    build = dict(spec['observer_build'],argv=spec['observer_build']['observer_argv'])
    loaded, evidence = compile_candidate(build,output,'observer',observer_source)
    pair = loaded.WM02MultitaskTaskPairV6
    pair.argtypes = [ctypes.c_char_p] * 3
    pair.restype = ctypes.c_int
    if pair(b'floor', b'FL', b'unknown') != -1:
        raise ValueError('INVALID_ROLE_MUST_FAIL_CLOSED')
    return pair, evidence


def check_compiled_pairs(model, pair):
    from go2arx5_vla.world_model.multitask_contact_v6 import observer_permitted
    from scripts.pi0294.task_contacts import original_allowed

    names = list({model.geom(i).name or None for i in range(model.ngeom)})
    encode = lambda x: None if x is None else x.encode()
    comparisons = 0
    for a, b, color in product(names, names, ('red', 'blue')):
        role_pair = pair(encode(a), encode(b), encode(color))
        for depth in (-.00401, -.004, .0):
            actual_rule = (original_allowed(a,b) or role_pair == 1) and not (
                role_pair == 1 and depth < -.004
            )
            if actual_rule != observer_permitted(a,b,depth,color):
                raise ValueError(f'COMPILED_ROLE_PREDICATE_MISMATCH:{a}:{b}:{color}:{depth}')
            comparisons += 1
    return comparisons


def recorded_prefix(path):
    # Hash only the read prefix; the query0 packet has its separate original pin.
    from scripts.pi0294.task_state import parse_state

    accumulated = hashlib.sha256()
    with path.open('rb') as stream:
        for expected in range(1, 5704):
            length = stream.read(4)
            if len(length) != 4:
                raise ValueError('ORIGINAL_INITIALIZATION_PREFIX_TRUNCATED')
            size, = struct.unpack('<I', length)
            if not 40 <= size <= 2 * 1024**2:
                raise ValueError('ORIGINAL_PREFIX_PACKET_LENGTH')
            raw = stream.read(size)
            if len(raw) != size:
                raise ValueError('ORIGINAL_PREFIX_PACKET_TRUNCATED')
            accumulated.update(length)
            accumulated.update(raw)
            packet = parse_state(raw)
            if packet['step'] != expected or abs(packet['time_s'] - .002 * expected) > 1e-8:
                raise ValueError('ORIGINAL_INITIALIZATION_CLOCK_GAP')
            yield packet, hashlib.sha256(raw).hexdigest(), accumulated if expected == 5703 else None


def audit_family(item, pair, folder):
    import mujoco
    from PIL import Image
    from go2arx5_vla.world_model.multitask_camera_v6 import TwoObjectCameraSession
    from go2arx5_vla.world_model.multitask_contact_v6 import observer_permitted
    from go2arx5_vla.world_model.multitask_state_v6 import TwoObjectState, parse_state
    from go2arx5_vla.world_model.native_replay import forbid_integration

    model_path = pinned(item['model'])
    model = mujoco.MjModel.from_binary_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    fixed_boxes = data.qpos[27:].copy()
    geometry = {c: Counter() for c in ('red', 'blue')}
    first = {c: None for c in geometry}
    comparisons = check_compiled_pairs(model, pair)
    with forbid_integration(mujoco) as calls:
        for packet, raw_sha, prefix_hash in recorded_prefix(Path(item['recorded_journal'])):
            data.qpos[:27] = packet['qpos'][:27]
            data.qpos[27:] = fixed_boxes
            data.qvel[:26] = packet['qvel'][:26]
            data.qvel[26:] = 0
            data.ctrl[:] = packet['ctrl']
            data.time = packet['time_s']
            mujoco.mj_forward(model, data)
            for contact in data.contact:
                a, b = (model.geom(i).name or None for i in (contact.geom1, contact.geom2))
                for color, counts in geometry.items():
                    if observer_permitted(a,b,contact.dist,color):
                        continue
                    names = sorted((str(a),str(b)))
                    counts[' / '.join(names)] += 1
                    if first[color] is None:
                        first[color] = dict(step=packet['step'], names=names, distance_m=float(contact.dist))
        if raw_sha != item['query0_state_sha256'] or prefix_hash is None:
            raise ValueError('ORIGINAL_QUERY0_PACKET_SHA_MISMATCH')
        # Clone the 41/38/20 packet's actual wire structure with no fabricated
        # contact forces. Geometry contacts above are recomputed by mj_forward.
        raw = struct.pack('<8sQdIII', b'GRTSTATE', 5703, packet['time_s'], 41, 38, 20)
        raw += b''.join(np.asarray(a, dtype='<f8').tobytes() for a in (data.qpos,data.qvel,data.ctrl))
        raw += struct.pack('<I', 0)
        clone = parse_state(raw)
        states = {color: TwoObjectState(model,color).update(clone) for color in geometry}
        if not np.array_equal(states['red']['tcp'],states['blue']['tcp']):
            raise ValueError('TASK_SELECTION_CHANGED_OWN_TCP')
        camera = TwoObjectCameraSession(model_path)
        try:
            images, body, _, calibration = camera.render_memory(clone)
            masks = camera.render_copy.visible_masks(clone)
            for key, rgb in [('body_rgb',body),('wrist_rgb',images['wrist_rgb'])]:
                if rgb.shape != (256,256,3) or rgb.dtype != np.uint8:
                    raise ValueError('V6_BODY_WRIST_RGB_ABI')
                Image.fromarray(rgb).save(folder / f'{key}.private.png')
        finally:
            camera.close()
        if calls:
            raise RuntimeError('GEOMETRY_AUDIT_ATTEMPTED_PHYSICS')
    return dict(family=item['family'], scene_sha256=item['model']['sha256'],
        compiled_pair_checks=comparisons, reference_prefix_sha256=prefix_hash.hexdigest(),
        reference_query0_sha256=raw_sha, geometric_snapshots=5703,
        contact_geometry_failures=geometry, first_geometry_failure=first,
        visible_pixels=masks, wrist_bytes_sha256=hashlib.sha256(images['wrist_rgb'].tobytes()).hexdigest(),
        body_bytes_sha256=hashlib.sha256(body.tobytes()).hexdigest(),
        RGB_calibration_time_s=calibration['wrist_rgb']['timestamp_s'],
        geometry_audit_only=True, recorded_WBC_state_restored=False)


def main(spec):
    import mujoco
    from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close

    root = Path(spec['registered_root'])
    start, cpu = time.monotonic(),time.process_time()
    rows = []
    value = dict(source_sha=spec['source_sha'], native_dispatch_qualified=False,
                 new_native_requests=0,new_integrations=0,new_updates=0,new_model_generations=0)
    try:
        if (spec['physics'] or spec['new_physical_requests_budget'] or
                mujoco.__version__ != '3.2.7' or mujoco.get_mjcb_control() is not None):
            raise PermissionError('PASSIVE_MUJOCO_327_AUDIT_ONLY')
        pair, value['observer_candidate'] = build_observer(spec, root)
        for item in spec['families']:
            folder = root / item['family']
            folder.mkdir()
            rows.append(audit_family(item,pair,folder))
            write(root/'PROGRESS.private.json',dict(completed=len(rows),new_integrations=0))
        value.update(exit_code=0,status='COLD_OBSERVER_PACKET_AND_GEOMETRY_AUDIT_COMPLETE')
    except BaseException:
        value.update(exit_code=1,status='COLD_BINDING_TECHNICAL_FAILURE',first_failure=traceback.format_exc())
    value.update(records=rows, wall_s=time.monotonic()-start, process_CPU_s=time.process_time()-cpu)
    write(root/'RESULT.private.json',value)
    close(spec,value['exit_code'])
    return value['exit_code']
