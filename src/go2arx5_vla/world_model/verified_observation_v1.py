"""One immutable RGB observation at an explicit file/byte receive boundary.

No future-frame prefetch, model, camera or clock API. The memory path accepts
the exact delivered NPZ bytes, not an unchecked mutable camera-buffer alias.
"""
import hashlib
from io import BytesIO
from pathlib import Path
from types import MappingProxyType
import zipfile

import numpy as np

KEYS = ('state_eef10', 'body_rgb', 'wrist_rgb', 'T_policy_from_mount',
        'policy_from_world', 'accepted_reference_TCP')
MAX_OBSERVATION_BYTES = 4 * 1024**2


def immutable_array(value):
    value = np.ascontiguousarray(value)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


class VerifiedObservation:
    def __init__(self, pin, raw):
        if len(raw) > MAX_OBSERVATION_BYTES:
            raise MemoryError('BOUNDED_OBSERVATION_BYTES_REQUIRED')
        raw = bytes(raw)
        if hashlib.sha256(raw).hexdigest() != pin['sha256']:
            raise ValueError('UNIFIED_INPUT_BYTES_CHANGED')
        # Bound decompression too; a small compressed file is not a memory bound.
        with zipfile.ZipFile(BytesIO(raw)) as archive:
            if sum(m.file_size for m in archive.infolist()) > MAX_OBSERVATION_BYTES:
                raise MemoryError('BOUNDED_DECODED_OBSERVATION_REQUIRED')
        with np.load(BytesIO(raw), allow_pickle=False) as z:
            values = {key: immutable_array(z[key]) for key in KEYS}
        for key in ('body_rgb', 'wrist_rgb'):
            a = values[key]
            if a.dtype != np.uint8 or a.ndim != 3 or a.shape[-1] != 3:
                raise ValueError('ORDINARY_UINT8_BODY_AND_WRIST_RGB_REQUIRED')
        if values['state_eef10'].shape != (10,) or not all(np.isfinite(a).all() for a in values.values()):
            raise ValueError('FINITE_CAPTURED_QUERY_REQUIRED')
        self._pin = dict(pin)
        self.serialized = raw
        self._values = MappingProxyType(values)
        self.nbytes = sum(a.nbytes for a in values.values())

    @classmethod
    def from_file(cls, pin):
        # One open/read/hash/decode of the SAME bytes, avoiding hash/load TOCTOU.
        with Path(pin['path']).open('rb') as handle:
            raw = handle.read(MAX_OBSERVATION_BYTES + 1)
        return cls(pin, raw)

    def values_for(self, pin):
        if dict(pin) != self._pin:
            raise ValueError('VERIFIED_OBSERVATION_IDENTITY_CHANGED')
        return self._values
