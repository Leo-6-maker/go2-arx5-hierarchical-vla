"""Read-only native snapshot exporter; stdlib-only so it can run over existing SSH.

This observer neither imports MuJoCo/ROS nor opens any control/lease file for write.
It may observe a CLOSED run, which is always labelled SEALED, never LIVE.
"""
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import time

SCHEMA = 'WM02_NATIVE_LIVE_SNAPSHOT_V1'
MAX_STATE = 2 * 1024 * 1024


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def header(raw):
    if not 728 <= len(raw) <= MAX_STATE:
        raise ValueError('INVALID_NATIVE_PACKET_LENGTH')
    magic, step, stamp, nq, nv, nu = struct.unpack_from('<8sQdIII', raw)
    count, = struct.unpack_from('<I', raw, 724)
    if (magic, nq, nv, nu) != (b'GRTSTATE', 34, 32, 20) or len(raw) != 728 + 160 * count:
        raise ValueError('NATIVE_PACKET_ABI')
    if not math.isfinite(stamp) or abs(stamp - step * .002) > 1e-8:
        raise ValueError('ORIGINAL_NATIVE_CLOCK_REQUIRED')
    if not all(math.isfinite(v) for v in struct.unpack_from('<86d', raw, 36)):
        raise ValueError('NONFINITE_NATIVE_STATE')
    for i in range(count):
        if not all(math.isfinite(v) for v in struct.unpack_from('<19d', raw, 728 + 160 * i + 8)):
            raise ValueError('NONFINITE_NATIVE_CONTACT')
    return step, stamp


class NativeSource:
    def __init__(self, config):
        self.cpu_start = time.process_time()
        self.config = config
        pin = config['execution_spec']
        self.spec_path = Path(pin['path'])
        if digest(self.spec_path) != pin['sha256']:
            raise ValueError('EXECUTION_SPEC_CHANGED')
        self.spec = json.loads(self.spec_path.read_text())
        self.root = Path(self.spec['registered_root'])
        self.output = Path(config['native_output'])
        if (self.spec_path.resolve() != (self.root / 'EXECUTION_SPEC.json').resolve()
                or self.output.resolve() != (self.root / 'native-01').resolve()
                or self.root.name != config['episode_id']
                or self.spec['source_sha'] != config['source_sha']):
            raise ValueError('EXACT_NATIVE_RUN_BINDING_REQUIRED')
        for key, pin_key in (('model_path', 'scene_sha256'), ('native_binary', 'native_binary_sha256')):
            if digest(self.spec[key]) != config[pin_key]:
                raise ValueError('NATIVE_ASSET_CHANGED:' + key)
        self.identity = {k: config[k] for k in ('episode_id', 'source_sha', 'scene_sha256', 'native_binary_sha256')}
        self.identity['spec_sha256'] = pin['sha256']
        self.previous = None
        self.sequence = 0
        self.command_signature = None
        self.command = {}

    def command_status(self):
        path = self.output / 'COMMANDS.private.jsonl'
        try:
            stat = path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            if signature == self.command_signature:
                return self.command
            with path.open('rb') as stream:
                offset = max(0, stat.st_size - 256 * 1024)
                stream.seek(offset)
                lines = stream.read(256 * 1024).splitlines()
            if offset:
                lines = lines[1:]
            rows = []
            for line in lines:
                try:
                    rows.append(json.loads(line))
                except (ValueError, UnicodeDecodeError):
                    continue  # Last append can be incomplete; it is never a command acknowledgement.
            installed = {r.get('wire_sha256') for r in rows if r.get('phase') == 'installed'}
            proposed = [r for r in rows if r.get('phase') == 'proposed' and r.get('wire_sha256') in installed]
            actual = [r for r in rows if 'achieved' in r.get('phase', '')]
            self.command = {'stage': actual[-1].get('stage', actual[-1]['phase']) if actual else self.spec['phase']}
            if proposed:
                # This is the installed horizon endpoint, explicitly not the current executed atom.
                self.command['installed_horizon_goal_m'] = proposed[-1]['action_EEF10'][-1][:3]
            self.command_signature = signature
        except FileNotFoundError:
            pass
        return self.command

    def snapshot(self):
        now = time.time()
        self.sequence += 1
        try:
            lease = json.loads(Path(self.spec['lease']).read_text())
            if lease.get('spec_sha256') != self.identity['spec_sha256'] or lease.get('run_uuid') != self.root.name:
                raise ValueError('WORKER_LEASE_IDENTITY_CHANGED')
            lease_status = lease.get('status', 'UNKNOWN')
        except (FileNotFoundError, json.JSONDecodeError):
            lease_status = 'UNKNOWN'
        if lease_status == 'ACTIVE' and now >= self.spec['deadline_epoch']:
            lease_status = 'EXPIRED'
        message = dict(schema=SCHEMA, **self.identity, sequence=self.sequence,
                       lease_status=lease_status, source_epoch_s=now, command=self.command_status(),
                       observer_pid=os.getpid(), observer_cpu_s=time.process_time() - self.cpu_start)
        try:
            with (self.output / 'state.bin').open('rb') as stream:
                stat = os.fstat(stream.fileno())
                raw = stream.read(MAX_STATE + 1)
        except FileNotFoundError:
            return dict(message, packet=None)
        step, stamp = header(raw)
        sha = hashlib.sha256(raw).hexdigest()
        if self.previous is not None:
            old_step, old_sha = self.previous
            if step < old_step or (step == old_step and sha != old_sha):
                raise ValueError('NATIVE_STATE_REGRESSED_OR_CHANGED_AT_SAME_TICK')
        self.previous = step, sha
        return dict(message, step=step, sim_time_s=stamp, state_age_s=max(0., now - stat.st_mtime),
                    packet_sha256=sha, packet=base64.b64encode(raw).decode('ascii'))


def emit(config, *, max_wall=300., fps=15., stream=None):
    import sys
    if not 1 <= max_wall <= 86400 or not 1 <= fps <= 30:
        raise ValueError('FINITE_OBSERVER_ENDPOINT_AND_RATE_REQUIRED')
    stream = sys.stdout if stream is None else stream
    source = NativeSource(config)
    started = time.monotonic()
    count = 0
    try:
        while time.monotonic() - started < max_wall:
            tick = time.monotonic()
            stream.write(json.dumps(source.snapshot(), allow_nan=False, separators=(',', ':')) + '\n')
            stream.flush()
            count += 1
            time.sleep(max(0., 1 / fps - (time.monotonic() - tick)))
    except BrokenPipeError:
        pass
    return count
