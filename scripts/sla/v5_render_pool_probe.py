"""Measure isolated cold scene states sharing one EGL renderer without integration."""

import argparse
from contextlib import closing
import ctypes
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import mujoco as mj
from mujoco import egl as mjegl
import numpy as np
from OpenGL import EGL, GL

from go2arx5_vla.world_model.native_replay import forbid_integration


POPULATIONS = (1, 2, 4, 8, 16, 32, 64, 128)
OBSERVATIONS_PER_PASS = 128
CAMERAS = ("body_rgb", "wrist_rgb")
SCENE = "2eced69ad7ebd1992a4a5105f2ca0bf9c53ec3cca2eb52c69342b4b4dbeab21b"


def egl_device():
    query = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_ssize_t))
    display_device = query(EGL.eglGetProcAddress(b"eglQueryDisplayAttribEXT"))
    device_attribute = query(EGL.eglGetProcAddress(b"eglQueryDeviceAttribEXT"))
    handle = ctypes.c_ssize_t()
    ordinal = ctypes.c_ssize_t(-1)
    if not (display_device(ctypes.cast(mjegl.EGL_DISPLAY, ctypes.c_void_p), 0x322C, ctypes.byref(handle))
            and device_attribute(handle.value, 0x323A, ctypes.byref(ordinal))):
        raise RuntimeError("NVIDIA EGL did not identify its CUDA device")
    return ordinal.value


def run(scene, output, expected):
    if hashlib.sha256(scene.read_bytes()).hexdigest() != SCENE:
        raise ValueError("Exact cold qualification scene required")
    model = mj.MjModel.from_binary_path(str(scene))
    geom = model.geom("task_cube_geom")
    body = int(geom.bodyid[0])
    joint = int(model.body_jntadr[body])
    if model.jnt_type[joint] != mj.mjtJoint.mjJNT_FREE:
        raise ValueError("Expected an independently movable test body")
    offset = int(model.jnt_qposadr[joint])
    first = {}
    results = []

    with forbid_integration(mj) as integrations, closing(mj.Renderer(model, height=256, width=256)) as renderer:
        device = egl_device()
        if device != expected or os.environ["MUJOCO_EGL_DEVICE_ID"] != str(expected):
            raise ValueError("Actual EGL device differs from registered role")
        vendor = GL.glGetString(GL.GL_VENDOR).decode()
        name = GL.glGetString(GL.GL_RENDERER).decode()
        for worlds in POPULATIONS:
            start = time.monotonic()
            states = [mj.MjData(model) for _ in range(worlds)]
            for i, data in enumerate(states):
                data.qpos[offset:offset + 2] += ((i % 16 - 7.5) * .002, (i // 16 - 3.5) * .002)
                mj.mj_forward(model, data)
            create = time.monotonic() - start
            baselines = [data.qpos.copy() for data in states]
            for _ in range(3):
                for camera in CAMERAS:
                    renderer.update_scene(states[0], camera=camera)
                    renderer.render()
            passes = []
            changed_frames = []
            fingerprints = set()
            for _ in range(2):
                t = time.monotonic()
                for i in range(OBSERVATIONS_PER_PASS):
                    env = i % worlds
                    views = []
                    for camera in CAMERAS:
                        renderer.update_scene(states[env], camera=camera)
                        views.append(renderer.render().copy())
                    rgb = np.stack(views)
                    fingerprint = hashlib.sha256(rgb.tobytes()).hexdigest()
                    fingerprints.add(fingerprint)
                    if env not in first:
                        first[env] = (fingerprint, rgb.copy())
                    if fingerprint != first[env][0]:
                        diff = np.abs(rgb.astype(np.int16) - first[env][1])
                        changed_frames.append((int(np.count_nonzero(diff)), int(diff.max())))
                passes.append(time.monotonic() - t)
            if any(data.time != 0 or not np.array_equal(data.qpos, baseline)
                   for data, baseline in zip(states, baselines, strict=True)):
                raise ValueError("Cold benchmark changed live simulation state")
            positions = [tuple(data.geom(geom.id).xpos[:2]) for data in states]
            if len(set(positions)) != worlds:
                raise ValueError("Cold world copies share their scene state")
            unique = len({first[env][0] for env in range(worlds)})
            if unique != worlds:
                raise ValueError("The actual camera pair did not expose all distinct cold inputs")
            results.append(dict(worlds=worlds, distinct_geometry_states=len(set(positions)), create_s=create,
                observations=2 * OBSERVATIONS_PER_PASS,
                observations_per_s=2 * OBSERVATIONS_PER_PASS / sum(passes),
                seconds=passes, rss_high_water_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                rgb_hashes=len(fingerprints), changed_same_state_frames=len(changed_frames),
                largest_changed_channel_count=max((n for n, _ in changed_frames), default=0),
                largest_channel_delta=max((n for _, n in changed_frames), default=0)))
        if integrations:
            raise RuntimeError("Cold probe performed native integration")
    with output.open("x") as handle:
        json.dump(dict(scene_sha256=SCENE, registered_device=expected, actual_egl_device=device,
            vendor=vendor, renderer=name, results=results, worlds_isolated_before_integration=True,
            native_ticks=0, native_starts=0, controller_scored=False, optimizer_updates=0,
            renderers=1, cameras=CAMERAS, distinct_reference_rgb_pairs=len({sha for sha, _ in first.values()})),
            handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("expected", type=int)
    run(**vars(parser.parse_args()))
