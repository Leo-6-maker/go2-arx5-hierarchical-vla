"""Finite binary-pipe companion for the current admitted target worker.

No native imports or process start on import. This library is NOT admission or
a launcher. The existing owner must validate its current manifest/lease and
once-reserved complete resources before calling it, and supply that original
resource/identity guard. An SDK failure/unknown never restarts or resends.
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import struct
import subprocess
import threading
import time
import uuid

from .target_sdk_constraint_wire import request, bind_samples, MOBILE_SDK_SHA256, FROZEN_TASK_SHA256

CLIENT_TASK_SHA256 = 'd0c485bc0d4a4d420c2fcd621103358718f99980e4cde3b6c464e189b2896ad3'
CORE_SDK_SHA256 = 'a8c1c3e422a64ad806b0f3457695052df938fcbe6cc34f09fb25c09d5f0b9220'
SDK_ROLE = 'DEC1_FROZEN_TARGET_SDK_CLIENT_QUALIFICATION_V1'
TASK_ROLE = 'DEC1_FROZEN_TARGET_TASK_V1'
T0, STOP_NEW, STOP_PHYSICS, HARD_CLOSE = 1792348196, 1792866596, 1792924196, 1792952996
CAMPAIGN = '20261018T182956Z-localpc-pv1'


def need(ok, why):
    if not ok: raise PermissionError('DEC1_SDK_CLIENT_' + why)


def digest(raw): return hashlib.sha256(raw).hexdigest()


def _pin(path, expected):
    p = Path(path)
    need(p.is_absolute() and p.is_file() and not p.is_symlink(), 'EXPLICIT_PINNED_FILE')
    need(p.stat().st_size <= 64*1024**2, 'BOUNDED_PINNED_FILE')
    actual = hashlib.sha256()
    with p.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024**2), b''): actual.update(chunk)
    need(type(expected) is str and len(expected) == 64 and actual.hexdigest() == expected,
         'PINNED_FILE_BYTES_CHANGED')
    return p


@dataclass(frozen=True)
class SdkClientPins:
    executable: str
    executable_sha256: str
    client_task: str
    urdf: str
    urdf_sha256: str
    mobile_archive: str
    core_archive: str
    private_readonly_cache: str
    cache_members: tuple  # (relative member, original sealed sha256)
    frames: tuple         # actual frozen base / EE / EE1 overrides; '-' means unchanged
    owner_manifest: str
    owner_manifest_sha256: str
    owner_run_uuid: str
    owner_source_sha: str
    owner_request_id: str
    initialization_mode: str


def validate_pins(pins):
    need(isinstance(pins, SdkClientPins), 'SEALED_CLIENT_INPUTS')
    binary = _pin(pins.executable, pins.executable_sha256)
    need(os.access(binary, os.X_OK), 'COMPILED_CLIENT_REQUIRED')
    need(pins.initialization_mode in ('ORIGINAL_SDK_INITIALIZE','SEALED_CACHE_LOAD'), 'EXPLICIT_SEALED_INITIALIZATION_MODE')
    _pin(pins.client_task, FROZEN_TASK_SHA256 if pins.initialization_mode=='ORIGINAL_SDK_INITIALIZE' else CLIENT_TASK_SHA256)
    _pin(pins.urdf, pins.urdf_sha256)
    _pin(pins.mobile_archive, MOBILE_SDK_SHA256)
    _pin(pins.core_archive, CORE_SDK_SHA256)
    cache = Path(pins.private_readonly_cache)
    need(cache.is_absolute() and cache.is_dir() and not cache.is_symlink(), 'EXPLICIT_PRIVATE_CACHE')
    if pins.initialization_mode=='ORIGINAL_SDK_INITIALIZE':
        root=Path(pins.owner_manifest).parent.resolve()
        need(cache.resolve().is_relative_to(root) and cache.resolve()!=root
             and not any(cache.iterdir()) and cache.stat().st_mode & 0o222
             and pins.cache_members==(), 'NEW_EMPTY_REGISTERED_CACHE_ONLY')
    else:
        _validate_readonly_cache(cache,pins.cache_members)
    need(type(pins.frames) is tuple and len(pins.frames) == 3
         and all(type(s) is str and 0 < len(s) <= 256 and '\x00' not in s for s in pins.frames),
         'EXPLICIT_ORIGINAL_FRAME_OVERRIDES')
    return pins


def _validate_readonly_cache(cache,members):
    need(type(members) is tuple and 0 < len(members) <= 256, 'SEALED_AD_CACHE_MEMBERS')
    names = set(); total = 0
    for name, expected in members:
        member = Path(name)
        need(type(name) is str and not member.is_absolute() and '..' not in member.parts
             and name not in names, 'NO_CACHE_ESCAPE_OR_DUPLICATE')
        names.add(name); total += (cache/member).stat().st_size
        need(total <= 256*1024**2, 'BOUNDED_AD_CACHE_BYTES')
        _pin(cache/member, expected)
    need(total <= 256*1024**2, 'BOUNDED_AD_CACHE_BYTES')
    found = set()
    for p in (cache, *cache.rglob('*')):
        need(not p.is_symlink() and not (p.stat().st_mode & 0o222), 'READONLY_PRIVATE_CACHE_ONLY')
        if p.is_file(): found.add(str(p.relative_to(cache)))
        else: need(p.is_dir(), 'NO_SPECIAL_CACHE_FILES')
    need(found == names, 'EXACT_CACHE_INVENTORY')


class SdkClient:
    """One owned subprocess, one thread, <=80 native policy-knot requests.

    A returned BoundSdkSamples still has no prefix-safety/physics authority.
    The original worker owns full-state guards, original quota and STOP.
    """
    def __init__(self, folder, pins, *, guard, maximum_requests, timeout_s=30, startup_timeout_s=120):
        self.pid, self.thread = os.getpid(), threading.get_ident()
        self.folder = Path(folder); self.guard = guard; self.pins = pins
        self.process = self.log = None; self.fd = None
        self.failed = self.closed = False; self.sequence = 0
        need(callable(guard), 'CURRENT_ORIGINAL_RESOURCE_AND_IDENTITY_GUARD')
        need(isinstance(pins, SdkClientPins), 'SEALED_CLIENT_INPUTS')
        self._guard()
        need(time.time() < STOP_NEW, 'NO_NEW_CONSTRUCTOR_AFTER_ORIGINAL_CUTOFF')
        validate_pins(pins)
        need(type(maximum_requests) is int and 0 < maximum_requests <= 80, 'ORIGINAL_FINITE_QUERY_CAP')
        need(type(timeout_s) in (int, float) and 0 < timeout_s <= 30
             and type(startup_timeout_s) in (int, float) and 0 < startup_timeout_s <= 120, 'BOUNDED_CLIENT_IO')
        need(self.folder.is_dir() and not self.folder.is_symlink(), 'EXISTING_REGISTERED_OUTPUT')
        self.maximum, self.timeout = maximum_requests, timeout_s
        self._put('SDK_CLIENT_INTENT.json', dict(once=True, no_automatic_restart=True,
            pins=pins.__dict__, maximum_requests=maximum_requests, pid=self.pid, thread=self.thread))
        # An old INTENT makes a second construction fail BEFORE Popen, even if
        # no start/ready receipt survived. Never infer no constructor effects.
        self.log = (self.folder/'SDK_CLIENT.log').open('xb', buffering=0)
        self.fd, response = os.pipe(); os.set_blocking(self.fd, False)
        try:
            self._guard()
            need(time.time() < STOP_NEW, 'FRESH_NO_NEW_CONSTRUCTOR_AFTER_ORIGINAL_CUTOFF')
            argv = [pins.executable, pins.client_task, pins.private_readonly_cache, pins.urdf,
                    *pins.frames, str(response), str(maximum_requests),pins.initialization_mode]
            env=dict(os.environ)
            # The SDK companion must not inherit the sim clock/actuation
            # interposers used by a full-body task subprocess.
            env.pop('LD_PRELOAD',None);env.pop('LD_AUDIT',None);env.pop('LD_LIBRARY_PATH',None)
            env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            self.process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=self.log, stderr=self.log,
                                            pass_fds=(response,), bufsize=0, start_new_session=True,env=env)
            os.close(response); response = None
            os.set_blocking(self.process.stdin.fileno(), False)
            self._put('SDK_CLIENT_STARTED.json', dict(pid=self.process.pid,
                proc_start=Path('/proc/%d/stat' % self.process.pid).read_text().rsplit(')', 1)[1].split()[19],
                argv=argv, constructor_effects_not_inferred_from_exit=True,
                simulation_preloads_removed=True,CUDA_VISIBLE_DEVICES=''))
            ready = self._read(20, time.monotonic()+startup_timeout_s)
            need(ready == b'D1SDKI01'+struct.pack('<III', 12, 12, 3), 'ACTUAL_ORIGINAL_SDK_READY')
            self._guard()
            self._put('SDK_CLIENT_READY.json', dict(pid=self.process.pid, model_type=3, state_dim=12, input_dim=12,
                native_prefix_safety_qualified=False, physics_advance_authorized=False))
        except BaseException:
            self.failed = True
            try: self.close()
            finally:
                if response is not None: os.close(response)
            raise

    def _guard(self, closeout=False):
        need(os.getpid() == self.pid and threading.get_ident() == self.thread, 'SAME_OWNER_PROCESS_AND_THREAD')
        self.guard(closeout=closeout)
        p = self.pins
        manifest_path = _pin(p.owner_manifest, p.owner_manifest_sha256)
        need(self.folder.resolve().is_relative_to(manifest_path.parent.resolve())
             and self.folder.resolve() != manifest_path.parent.resolve(), 'OUTPUT_INSIDE_CURRENT_REGISTERED_CHILD')
        m = json.loads(manifest_path.read_bytes())
        need(m['schema'] == 'PI0294_PV1_CPU_CHILD_RUN_MANIFEST_V1'
             and m['scope'] in (SDK_ROLE, TASK_ROLE) and m['run_uuid'] == p.owner_run_uuid
             and m['code']['vla_commit'] == p.owner_source_sha
             and len(p.owner_source_sha) == 40 and all(c in '0123456789abcdef' for c in p.owner_source_sha),
             'CURRENT_DEC1_CHILD_NOT_EXPIRED_R4_OR_CLOSED_COUNTER')
        need(str(uuid.UUID(p.owner_request_id)) == p.owner_request_id, 'EXPLICIT_ONCE_REQUEST_ID')
        q_path = _pin(manifest_path.parent/'REQUEST.json', m['execution']['sealed_inputs']['REQUEST.json'])
        q = json.loads(q_path.read_bytes())
        need(q['request_id'] == p.owner_request_id and q['child_run_uuid'] == p.owner_run_uuid
             and q['scope'] == m['scope'] and q['campaign_run_uuid'] == CAMPAIGN
             and q['original_T0_epoch'] == m['limits']['PV1_T0_epoch'] == T0
             and m['limits']['PV1_hard_close_epoch'] == HARD_CLOSE
             and q['deadline_epoch'] == m['limits']['operation_deadline_epoch'] <= STOP_PHYSICS,
             'ORIGINAL_SCOPE_CLOCK_AND_CURRENT_REQUEST_BINDING')
        need(m['execution']['sdk_initialization_mode']==q['sdk_initialization_mode']==p.initialization_mode,
             'NO_IMPLICIT_CACHE_COMPILATION_MODE_SWITCH')
        if p.initialization_mode=='ORIGINAL_SDK_INITIALIZE':
            need(m['scope']==SDK_ROLE and all(type(q[k]) is int and q[k]==0
                 for k in ('GR00T_queries','physics_ticks','policy_episodes')), 'COLD_CPU_QUALIFICATION_ONLY_NO_TASK_COMPILATION')
        need(closeout or time.time() < q['deadline_epoch'], 'CURRENT_TARGET_OPERATION_DEADLINE')
        # The existing standard entry/resource guard, not these bytes or this
        # library, proves adopted authority, actual reservation and host/profile.

    def _put(self, name, value):
        if type(value) is not bytes:
            value = dict(owner_run_uuid=self.pins.owner_run_uuid,
                         owner_manifest_sha256=self.pins.owner_manifest_sha256,
                         owner_request_id=self.pins.owner_request_id, **value)
        raw = value if type(value) is bytes else (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()
        with (self.folder/name).open('xb') as handle:
            handle.write(raw); handle.flush(); os.fsync(handle.fileno())

    def _wait(self, fd, event, deadline):
        self._guard()
        need(time.monotonic() < deadline, 'IO_UNKNOWN_NO_RESEND')
        with selectors.DefaultSelector() as selector:
            selector.register(fd, event)
            return selector.select(min(.05, max(0., deadline-time.monotonic())))

    def _read(self, count, deadline):
        need(0 <= count <= 4*1024**2, 'BOUNDED_REPLY')
        raw = bytearray()
        while len(raw) < count:
            if not self._wait(self.fd, selectors.EVENT_READ, deadline): continue
            try: chunk = os.read(self.fd, count-len(raw))
            except BlockingIOError: continue
            need(bool(chunk), 'TRUNCATED_OR_UNKNOWN_CLIENT_REPLY')
            raw.extend(chunk)
        return bytes(raw)

    def exchange(self, prediction):
        self._guard()
        need(not self.failed and not self.closed and self.sequence < self.maximum, 'LATCHED_OR_FINITE_CLIENT_LIMIT')
        q = request(prediction); tag = 'SDK_REQUEST_%03d' % self.sequence
        body = bytearray(b'D1SDKC01'+struct.pack('<I', len(q['times'])))
        for t, x, u in zip(q['times'], q['states'], q['inputs']): body.extend(struct.pack('<25d', t, *x, *u))
        # All local validation precedes the durable once intent. Any error
        # after this boundary latches permanently, including an unknown write.
        self._put(tag+'_INTENT.json', dict(request=q, binary_sha256=digest(body), no_automatic_retry=True))
        try:
            deadline = time.monotonic()+self.timeout; sent = 0
            while sent < len(body):
                fd = self.process.stdin.fileno()
                if not self._wait(fd, selectors.EVENT_WRITE, deadline): continue
                try: n = os.write(fd, body[sent:])
                except BlockingIOError: continue
                need(n > 0, 'REQUEST_WRITE_UNKNOWN_NO_RESEND'); sent += n
            header = self._read(12, deadline)
            need(header == b'D1SDKR01'+struct.pack('<I', len(q['times'])), 'EXACT_RESPONSE_COUNT_AND_VERSION')
            binary = bytearray(header); samples = []
            for _ in q['times']:
                raw = self._read(20, deadline); binary.extend(raw)
                t, cost, width = struct.unpack('<ddI', raw)
                need(0 < width <= 4096 and len(binary)+width*8 <= 4*1024**2, 'BOUNDED_ORIGINAL_CONSTRAINT_VECTOR')
                raw = self._read(width*8, deadline); binary.extend(raw)
                samples.append(dict(time=t, jointLimitsCost=cost, selfCollision=list(struct.unpack('<%dd' % width, raw))))
            reply = json.dumps(dict(request=q, samples=samples), allow_nan=False).encode()
            value = bind_samples(prediction, reply, expected_reply_sha256=digest(reply))
            self._guard()
            self._put(tag+'_RAW_REPLY.bin', bytes(binary))
            self._put(tag+'_REPLY.json', reply)
            self._put(tag+'_CONFIRMED.json', dict(reply_sha256=digest(reply), raw_reply_sha256=digest(binary),
                                                 native_prefix_safety_qualified=False, physics_advance_authorized=False))
            self.sequence += 1
            return value
        except BaseException:
            self.failed = True
            raise

    def close(self):
        if self.closed: return
        # Cleanup is allowed after a task deadline. The original owner still
        # has to settle actual/unknown constructor and CPU effects separately.
        need(os.getpid() == self.pid and threading.get_ident() == self.thread, 'SAME_OWNER_CLEANUP')
        guard_error = None
        try: self.guard(closeout=True)
        except BaseException as error: guard_error = error; self.failed = True
        try:
            if self.process is not None:
                self.process.stdin.close()
                try: rc = self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGTERM)
                    try: rc = self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(self.process.pid, signal.SIGKILL); rc = self.process.wait(timeout=5)
                post_error=None
                if rc==0 and not self.failed and guard_error is None:
                    try:
                        if self.pins.initialization_mode=='ORIGINAL_SDK_INITIALIZE': self._seal_cache()
                        else: _validate_readonly_cache(Path(self.pins.private_readonly_cache),self.pins.cache_members)
                    except BaseException as error:
                        self.failed=True;post_error=error
                self._put('SDK_CLIENT_EXIT.json', dict(pid=self.process.pid, returncode=rc,
                    failed_or_unknown=self.failed or rc != 0, requests_confirmed=self.sequence,
                    cache_validation_or_seal_failed=post_error is not None,
                    lease_closed=False, constructor_and_CPU_settled=False))
                if post_error is not None: raise post_error
        finally:
            if self.fd is not None: os.close(self.fd); self.fd = None
            if self.log is not None: self.log.close()
            self.closed = True
        if guard_error is not None: raise guard_error

    def _seal_cache(self):
        # Only AFTER the original SDK has exited naturally: do not change its
        # working-directory permissions while its destructor may still run.
        cache=Path(self.pins.private_readonly_cache);members=[];total=0
        paths=list(cache.rglob('*'))
        need(0<len(paths)<=1024,'BOUNDED_REALIZED_CACHE_TREE')
        for p in paths:
            need(not p.is_symlink() and (p.is_file() or p.is_dir()),'NO_SPECIAL_GENERATED_CACHE_MEMBER')
            if p.is_file():
                total+=p.stat().st_size
                need(total<=256*1024**2 and p.stat().st_size<=64*1024**2,'BOUNDED_REALIZED_CACHE_BYTES')
                members.append((str(p.relative_to(cache)),digest(p.read_bytes())))
        need(0<len(members)<=256,'REALIZED_CACHE_FILES_REQUIRED')
        for p in paths:
            if p.is_file(): p.chmod(0o444)
        for p in sorted([cache,*[x for x in paths if x.is_dir()]],key=lambda x:len(x.parts),reverse=True):p.chmod(0o555)
        _validate_readonly_cache(cache,tuple(members))
        self._put('SDK_CLIENT_CACHE_SEALED.json',dict(
            source='EXPLICIT_ORIGINAL_SDK_INITIALIZATION_AFTER_NATURAL_EXIT',
            original_task_sha256=FROZEN_TASK_SHA256,urdf_sha256=self.pins.urdf_sha256,
            mobile_archive_sha256=MOBILE_SDK_SHA256,core_archive_sha256=CORE_SDK_SHA256,
            frames=self.pins.frames,members=members,total_bytes=total,
            resource_settlement_or_lease_closure=False,native_prefix_safety_qualified=False))
