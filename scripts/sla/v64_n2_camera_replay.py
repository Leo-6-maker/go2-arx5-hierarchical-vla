"""Render the real N1 camera packets on one independently selected EGL device."""

import argparse
from contextlib import closing
import ctypes
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import time

import mujoco
import numpy as np
from OpenGL import GL

from go2arx5_vla.world_model.multitask_camera_v6 import TwoObjectCameraSession
from go2arx5_vla.world_model.multitask_state_v6 import packets
from go2arx5_vla.world_model.native_replay import forbid_integration
from go2arx5_vla.world_model.storage import digest
from v5_render_pool_probe import egl_device


REFERENCE = "709d793d544765fd2fcd51f10ee1f956784b2ccd293dc61e34b9724cf7328039"


def pci_device(ordinal):
    cuda = ctypes.CDLL("libcuda.so.1")
    device, bus = ctypes.c_int(), ctypes.create_string_buffer(32)
    if (cuda.cuInit(0) or cuda.cuDeviceGet(ctypes.byref(device), ordinal)
            or cuda.cuDeviceGetPCIBusId(bus, len(bus), device)):
        raise RuntimeError("Unable to bind actual EGL device to PCI identity")
    # CUDA and NVML format the PCI domain with different leading zero counts.
    def normalized(value):
        domain, bus, slot = value.lower().split(":")
        return f"{int(domain, 16):x}:{bus}:{slot}"

    identity = normalized(bus.value.decode())
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,pci.bus_id",
                                   "--format=csv,noheader"], text=True, timeout=5)
    matches = [uuid for uuid, pci in (line.strip().split(", ") for line in raw.splitlines())
               if normalized(pci) == identity]
    uuid, = matches
    return uuid, identity


def probe(case, output, *, gpu_uuid=None):
    result = case / "RESULT.private.json"
    if digest(result) != REFERENCE or json.loads((case / "control/LEASE.json").read_bytes())["status"] != "CLOSED":
        raise ValueError("The exact closed N1 observation card is required")
    detail = json.loads(result.read_bytes())["detail"]
    spec = json.loads((case / "EXECUTION_SPEC.json").read_bytes())
    model = Path(spec["model_path"])
    if digest(model) != spec["frozen_files"][str(model)]:
        raise ValueError("The archived scene differs from its recorded native bytes")
    native = case / "native-01/state.bin.journal"
    expected = {row["native_step"]: row for row in detail["RGB_pairs"]}
    if list(expected) != list(range(5703, 10703, 250)):
        raise ValueError("Unexpected camera record clock")
    rendered, frames, identity = [], [], None
    with forbid_integration(mujoco) as integrations, closing(TwoObjectCameraSession(model)) as session:
        for packet in packets(native):
            tick = packet["step"]
            if tick not in expected:
                continue
            start = time.monotonic()
            rgb, body, _, calibration = session.render_memory(packet)
            elapsed = time.monotonic() - start
            if identity is None:
                vendor = GL.glGetString(GL.GL_VENDOR).decode()
                renderer = GL.glGetString(GL.GL_RENDERER).decode()
                identity = dict(vendor=vendor, renderer=renderer)
                if gpu_uuid is not None:
                    egl_cuda = egl_device()
                    uuid, pci = pci_device(egl_cuda)
                    if uuid != gpu_uuid or "NVIDIA" not in vendor:
                        raise RuntimeError("The observed EGL device differs from its assigned GPU")
                    identity.update(observed_egl_cuda=egl_cuda, uuid=uuid, pci=pci)
                elif "llvmpipe" not in renderer:
                    raise RuntimeError("The original software renderer was not recovered")
            wrist = rgb["wrist_rgb"]
            if any(image.shape != (256, 256, 3) or image.dtype != np.uint8 for image in (body, wrist)):
                raise ValueError("The native RGB ABI changed")
            hashes = {name: sha256(array.tobytes()).hexdigest()
                      for name, array in (("body", body), ("wrist", wrist))}
            if (abs(calibration["wrist_rgb"]["timestamp_s"] - expected[tick]["rgb_time_s"]) > 1e-8
                    or abs(calibration["body_rgb"]["timestamp_s"] - expected[tick]["rgb_time_s"]) > 1e-8):
                raise ValueError("RGB diverged from the archived native observation clock")
            frames.append(dict(tick=tick, hashes=hashes, calibration=calibration, seconds=elapsed,
                               matches_original={name: value == expected[tick][name] for name, value in hashes.items()}))
            rendered.append(np.stack((body, wrist)))
    if integrations or [row["tick"] for row in frames] != list(expected):
        raise ValueError("Native replay advanced or missed a declared camera packet")
    if gpu_uuid is None and not all(all(row["matches_original"].values()) for row in frames):
        raise ValueError("Software replay differs from the original recorded RGB")
    output.mkdir(mode=0o700)
    images = output / "RGB.npz"
    with images.open("xb") as stream:
        np.savez_compressed(stream, frames=np.stack(rendered))
    with (output / "RESULT.json").open("x") as stream:
        json.dump(dict(identity=identity, original_case=REFERENCE, scene_sha256=digest(model),
                       native_journal_sha256=digest(native), images_sha256=digest(images), frames=frames,
                       new_native_requests=0, new_integrations=0, model_forwards=0), stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gpu-uuid")
    probe(**vars(parser.parse_args()))
