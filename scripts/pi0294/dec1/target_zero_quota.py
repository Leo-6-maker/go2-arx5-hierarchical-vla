"""Zero-integration quota/STOP boundary for current target native IO checks.

This component neither registers nor admits work. Its caller must first use
the current standard entry, actual paired eight-dimension reservation, lease
and full remaining-work forecast. It must never use the expired R4 runner.
Only a new, dedicated zero-physics request can arm quota zero. No task quota,
prefix safety verdict, model query or task success is supplied by this class.
The native worker must independently deny every integration entry point.
"""
import hashlib
import math
from pathlib import Path

from .target_paused_stop import OwnedTargetQuota, need, pause_document

ROLE = 'DEC1_FROZEN_TARGET_NATIVE_IO_QUALIFICATION_V1'
SCHEMA = 'DEC1_FROZEN_TARGET_ZERO_PHYSICS_NATIVE_IO_REQUEST_V1'
DIMS = ('CoreLab_CPU_s', 'local_CPU_s', 'episode_starts', 'physics_ticks',
        'constructors', 'CoreLab_output_bytes', 'local_output_bytes', 'transfer_bytes')


class OwnedTargetZeroQuota(OwnedTargetQuota):
    """A one-shot zero arm and the original frozen-clock STOP, never advance.

    The sealed request is checked again on every owner guard invocation. STOP
    still needs fresh actual native pause ACK plus byte-identical full state;
    writing zero alone is not proof that an existing process is paused.
    """
    def __init__(self, folder, *, request_path, request_sha256, clock_path,
                 episode_id, executor_id, run_uuid, manifest_sha256, guard, spin):
        need(callable(guard), 'EXISTING_OWNER_GUARD_AND_SPIN')
        request_path = Path(request_path)
        need(isinstance(request_sha256, str) and len(request_sha256) == 64
             and all(c in '0123456789abcdef' for c in request_sha256), 'ZERO_REQUEST_HASH')

        def current_guard(*, closeout):
            guard(closeout=closeout)
            need(not request_path.is_symlink() and request_path.is_file()
                 and request_path.stat().st_size <= 32768, 'ZERO_BOUNDED_SEALED_REQUEST')
            raw = request_path.read_bytes()
            need(len(raw) <= 32768 and hashlib.sha256(raw).hexdigest() == request_sha256,
                 'ZERO_REQUEST_BYTES_CHANGED')
            q = pause_document(raw)
            need(q['schema'] == SCHEMA and q['scope'] == ROLE
                 and q['child_run_uuid'] == run_uuid, 'ZERO_CURRENT_REQUEST_ROLE_AND_RUN')
            need(all(type(q[k]) is int and q[k] == 0 for k in
                 ('GPU_submissions', 'training_updates', 'GR00T_queries',
                  'policy_episodes', 'physics_ticks')), 'ZERO_NO_NEURAL_OR_PHYSICS_PERMISSION')
            v = q['requested']
            need(set(v) == set(DIMS)
                 and all(type(v[k]) in (int, float) and math.isfinite(v[k]) and v[k] >= 0 for k in DIMS)
                 and all(type(v[k]) is int for k in DIMS if not k.endswith('CPU_s'))
                 and v['episode_starts'] == v['physics_ticks'] == 0,
                 'ZERO_EXPLICIT_EIGHT_DIMENSIONS_AND_NO_PHYSICS_HOLD')

        self._initialize(folder, clock_path=clock_path, episode_id=episode_id,
            executor_id=executor_id, run_uuid=run_uuid, manifest_sha256=manifest_sha256,
            guard=current_guard, spin=spin, maximum_native_step=0,
            owner_extra=dict(scope=ROLE, request_sha256=request_sha256,
                             physics_advance_authorized=False))

    def arm(self):
        """Once, before process launch; unknown publication is never retried."""
        need(not self.latched and not self.unknown, 'LATCHED_NO_MORE_QUOTA')
        try:
            self._owner()
            self.guard(closeout=False)
            need(self.last is None and self.sequence == 0, 'ZERO_ARM_ONCE_ONLY')
            self._publish_quota(0)
        except BaseException:
            self.latched = self.unknown = True
            raise

    def quota(self, step):
        self.latched = True
        need(False, 'ZERO_NEVER_ADVANCE_NATIVE_QUOTA')
