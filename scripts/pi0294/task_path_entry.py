"""Cold binding of the existing standard entry for the approved S1 child.

Pure package preparation and read-only path checks. No new scheduler, registry,
resource reservation, process start, environment installation or physics. The
original enter.sh/doctor.py execute unchanged in the new run's sealed layout.
"""
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re

APPROVAL_SHA256 = '8a55153f728c0431116e2f9932f5286f6a0647d362252991712541638b27484f'
ENTRY_FILES = {
    'enter.sh': '0c7973e0a3bc0e585e7ab96cfcc9033532a98f56e9c9410882282a291bb7082f',
    'doctor.py': '013e7cea41740b10fdd213e30f6d65b0fbcaee295cce44b624fc61e23fe6d861',
    'ENVIRONMENTS.json': '117eb51e8a9cbb4fa60115f23f2e29d5a7df463c31ca792ddfded7659c32b6b6',
}
PROFILE = '4080-simulation'
HOST = 'corelab-4080s'
OWNER = 'EXISTING_LOCAL_PC'
PATH_ROLES = frozenset(('code', 'entry', 'doctor', 'profile', 'python', 'libraries',
    'model', 'assets', 'data', 'cwd', 'tmp', 'cache', 'control', 'lease', 'journal',
    'heartbeat', 'logs', 'output'))
LOCAL_FILESYSTEMS = frozenset(('ext4', 'xfs', 'btrfs', 'zfs', 'overlay', 'tmpfs'))
DURABLE_ROLES = frozenset(('control', 'lease', 'journal', 'heartbeat', 'logs', 'output'))


def need(ok, reason):
    if not ok:
        raise PermissionError('S1_TASK_PATH_ENTRY_'+reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def authority(raw, *, now_epoch):
    """Exact direct human approval; never reopen an expired historical scope."""
    need(type(raw) is bytes and sha(raw) == APPROVAL_SHA256, 'EXACT_HUMAN_APPROVAL_BYTES')
    value = json.loads(raw)
    need(type(now_epoch) in (int, float) and math.isfinite(now_epoch), 'FINITE_CURRENT_CLOCK')
    start = datetime.fromisoformat(value['adoption_recorded_at_UTC']).timestamp()
    end = datetime.fromisoformat(value['window_end_UTC']).timestamp()
    old_close = datetime.fromisoformat(value['old_hard_deadline_SGT']).timestamp()
    need(start <= now_epoch < end <= min(start+24*3600, old_close), 'APPROVED_FINITE_WINDOW')
    need(value['answer'] == '批准该 S1 有限开发范围' and value['approved_scope_only'] is True
         and value['ten_day_extra_scope_or_extension_adopted'] is False
         and value['closed_legacy_series_reopened'] is False, 'S1_ONLY_NO_LEGACY_OR_TD10_EXPANSION')
    proposal = value['approved_proposal']
    expected = dict(starts_max=4, short_task_s_each_max=20, conditional_full_task_s_each=32,
        warmup_s_each_max=5, integrations_all_live_shadow_warmup_and_tail_max=62012,
        RGB_pairs_max=2084, CPU_core_seconds_total_max=14400, CPU_concurrent_max=4,
        Core_new_output_and_staging_GiB_max=8, PC_small_evidence_GiB_max=1,
        new_transfer_GiB_max=8, GR00T_queries_max=0, training_updates_max=0, PBS_max=0)
    need(all(type(proposal.get(k)) is int and proposal[k] == v for k, v in expected.items()),
         'EXACT_APPROVED_S1_FINITE_CAPS')
    return value


def prepare_standard_entry(approval_bytes, files, *, run_root, source_sha, now_epoch):
    """Return an unwritten layout using the original three entry files verbatim.

    The original doctor resolves cwd relative to its own location. Putting the
    immutable capsule at entry/code/vla makes that existing behavior node local.
    The old profile's source field remains provenance; resolve-only does not
    open it. Actual module/library origins still require node_local_paths.
    """
    a = authority(approval_bytes, now_epoch=now_epoch)
    need(set(files) == set(ENTRY_FILES) and all(type(v) is bytes for v in files.values()),
         'THREE_ORIGINAL_ENTRY_MEMBERS')
    need(all(sha(files[k]) == pin for k, pin in ENTRY_FILES.items()),
         'ORIGINAL_ENTRY_MEMBER_CHANGED')
    root = PurePosixPath(run_root)
    need(root.is_absolute() and str(root) == str(run_root) and '..' not in root.parts
         and root.parent.name == 'PI0294' and root.name.startswith('20261029T')
         and root.name.endswith('-standing-task-path-s1'), 'NEW_REGISTERED_RUN_LAYOUT_ONLY')
    need(type(source_sha) is str and re.fullmatch('[0-9a-f]{40}', source_sha), 'FULL_FIXED_SOURCE_SHA')
    profiles = json.loads(files['ENVIRONMENTS.json'])['profiles']
    rows = [r for r in profiles if r['name'] == PROFILE]
    need(len(rows) == 1 and rows[0]['hostnames'] == [HOST]
         and PurePosixPath(rows[0]['python']).is_absolute(), 'ORIGINAL_SIMULATION_PROFILE')
    entry = root/'entry/infrastructure'
    return dict(schema='S1_ORIGINAL_STANDARD_ENTRY_COLD_LAYOUT_PROPOSAL_V1',
        run_uuid=root.name, owner=OWNER, profile=PROFILE, host=HOST,
        source_sha=source_sha, approval_sha256=APPROVAL_SHA256,
        deadline_epoch=datetime.fromisoformat(a['window_end_UTC']).timestamp(),
        capsule_relative='entry/code/vla',
        entry_path=str(entry/'enter.sh'), doctor_path=str(entry/'doctor.py'),
        entry_cwd=str(root/'entry/code/vla'), profile_python=rows[0]['python'],
        members={str(entry/k): dict(sha256=ENTRY_FILES[k], bytes=len(v)) for k, v in files.items()},
        directories=[str(root/x) for x in ('entry/infrastructure', 'entry/code/vla',
            'control', 'tmp', 'cache', 'logs', 'outputs')],
        existing_entry_implementation_changed=False,
        original_registry_registration_required=True,
        registry_or_lease_created=False, resources_reserved=False,
        native_admission=False, launch_allowed=False)


def registered_snapshot(raw, *, expected_sha256, run_root, role):
    """Verify a sealed readback of the one original canonical registry.

    This snapshot cannot register anything. The original owner's actual onceIO
    registration/readback must happen before these bytes can contain this run.
    """
    need(type(raw) is bytes and sha(raw) == expected_sha256, 'EXACT_CANONICAL_REGISTRY_READBACK')
    root = PurePosixPath(run_root)
    need(root.is_absolute() and str(root) == str(run_root) and '..' not in root.parts
         and type(role) is str and role, 'EXPLICIT_REGISTERED_RUN_AND_ROLE')
    rows = [r for r in json.loads(raw)['entries'] if r.get('role') == role]
    need(len(rows) == 1 and rows[0]['path'] == str(root)
         and rows[0].get('run_uuid') == root.name and rows[0].get('executor_node') == HOST,
         'ACTUAL_UNIQUE_ORIGINAL_REGISTRY_ENTRY')
    return rows[0]


def _mounts(raw):
    result = []
    for line in raw.decode().splitlines():
        fields, after = line.split(' - ', 1)
        mount = fields.split()[4]
        mount = re.sub(r'\\(040|011|012|134)', lambda m: chr(int(m.group(1), 8)), mount)
        result.append((Path(mount), after.split()[0]))
    return result


def _filesystem(path, mounts):
    matches = [(root, fs) for root, fs in mounts if path.is_relative_to(root)]
    need(bool(matches), 'ACTUAL_MOUNT_MISSING')
    return max(matches, key=lambda r: len(r[0].parts))[1]


def node_local_paths(paths):
    """Read-only full runtime dependency audit, never a launch permission.

    Uses actual resolved paths and the kernel mount table, not path prefixes or
    a local cache symlink. Every required output/control directory/file must
    already exist. Does not create or repair missing dependencies.
    """
    need(set(paths) == PATH_ROLES, 'COMPLETE_RUNTIME_DEPENDENCY_ROLES')
    mounts = _mounts(Path('/proc/self/mountinfo').read_bytes())
    facts = []
    for role, names in paths.items():
        need(type(names) is list and names, 'NONEMPTY_RUNTIME_PATHS_'+role)
        for name in names:
            need(type(name) is str and Path(name).is_absolute() and '..' not in Path(name).parts,
                 'ABSOLUTE_RUNTIME_PATH_'+role)
            path = Path(name).resolve(strict=True)
            fs = _filesystem(path, mounts)
            need(fs in LOCAL_FILESYSTEMS and not (role in DURABLE_ROLES and fs == 'tmpfs'),
                 'NONLOCAL_OR_VOLATILE_RUNTIME_PATH_'+role)
            st = path.stat()
            facts.append(dict(role=role, configured=name, resolved=str(path),
                filesystem=fs, device=st.st_dev, inode=st.st_ino, bytes=st.st_size))
    return dict(status='ACTUAL_NODE_LOCAL_RUNTIME_PATHS_VERIFIED_NOT_ADMISSION',
        paths=facts, resources_reserved=False, launch_allowed=False)
