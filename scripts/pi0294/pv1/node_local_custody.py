"""Read-only custody proof for a future cold registered Core successor.

The original canonical registry remains the authority. Immutable before/after
captures and its actual onceIO receipt avoid a live NFS registry dependency.
This helper grants no scope, lease, resources, model call or runtime admission.
Existing cold-successor and scientific validation must still admit the run.
"""
import hashlib,json,os,re,stat
from pathlib import Path,PurePosixPath

DESCRIPTOR='plans/NODE_LOCAL_REGISTRATION_CUSTODY.private.json'
ASSET_RELOCATION='plans/NODE_LOCAL_ASSET_RELOCATION.private.json'
SCHEMA='CORE_ORIGINAL_REGISTRATION_NODE_LOCAL_CUSTODY_V1'


def need(value,reason):
    if not value:raise PermissionError('CORE_LOCAL_CUSTODY_'+reason)


def sha(body):return hashlib.sha256(body).hexdigest()


def absolute(name):
    need(type(name)is str and PurePosixPath(name).is_absolute()
        and str(PurePosixPath(name))==name and '..'not in PurePosixPath(name).parts,
        'EXACT_ABSOLUTE_PATH')
    return PurePosixPath(name)


def registration_bytes(descriptor,before_bytes,after_bytes,intent_bytes,receipt_bytes,*,binding,manifest):
    """Verify actual append/CAS provenance, never register or read NFS."""
    d=json.loads(descriptor);root=absolute(binding['run_root']);registry=absolute(binding['registry'])
    need(d['schema']==SCHEMA and d['run_root']==str(root)and d['run_uuid']==root.name
        and binding['run_uuid']==manifest['run_uuid']==root.name
        and d['role']==binding['role']==manifest['execution']['private_path_role']
        and d['original_registry']==str(registry)and d['snapshot_is_new_authority']is False
        and d['native_admission']is False,'EXACT_COLD_ROLE_AND_ONE_ORIGINAL_AUTHORITY')
    need(re.fullmatch('[0-9a-f]{40}',d['source_sha'])is not None
        and d['source_sha']==manifest['code']['vla_commit'],'FULL_OWN_SOURCE_IDENTITY')
    before,after,intent,receipt=map(json.loads,(before_bytes,after_bytes,intent_bytes,receipt_bytes))
    need(d['registry_before_sha256']==sha(before_bytes)and d['registry_after_sha256']==sha(after_bytes)
        and d['registration_intent_sha256']==sha(intent_bytes)and d['registration_receipt_sha256']==sha(receipt_bytes),
        'EXACT_ORIGINAL_CAS_MEMBER_BYTES')
    need(set(before)==set(after)and all(before[k]==after[k]for k in before if k!='entries')
        and after['entries'][:len(before['entries'])]==before['entries'],
        'ALL_PREVIOUS_REGISTRY_METADATA_AND_ENTRIES_RETAINED')
    added=after['entries'][len(before['entries']):]
    need(type(d['appended_roles'])is list and 1<=len(d['appended_roles'])<=3
        and len(set(d['appended_roles']))==len(d['appended_roles'])
        and [r['role']for r in added]==d['appended_roles']and d['role']in d['appended_roles'],
        'ONE_EXACT_ROLE_APPEND_OPERATION_NOT_A_REPLACEMENT_REGISTRY')
    need(not any(r.get('role')in d['appended_roles']for r in before['entries']),
        'NO_PREEXISTING_ROLE_REWRITE')
    rows=[r for r in added if r['role']==d['role']]
    need(len(rows)==1,'UNIQUE_NEW_ORIGINAL_REGISTRY_ROLE')
    row=rows[0];lease=absolute(binding['lease'])
    need(lease.is_relative_to(root/'control')and row['path']==str(root)and row['run_uuid']==root.name
        and row['executor_node']=='corelab-4080s'and row['owner']=='EXISTING_LOCAL_PC'
        and row['source_sha']==d['source_sha']and row['local_lease']==str(lease),
        'REGISTERED_NODE_LOCAL_MUTABLE_LEASE_NOT_HISTORICAL_NFS_LEASE')
    key=d['registration_request_id']
    need(type(key)is str and 0<len(key)<=160 and re.fullmatch('[A-Za-z0-9_-]+',key)is not None,
        'EXACT_ORIGINAL_REGISTRATION_REQUEST')
    need(intent['schema']=='DEC1_SOURCE_CPU_ONCE_METADATA_INTENT_V1'
        and receipt['schema']=='DEC1_SOURCE_CPU_ONCE_METADATA_RECEIPT_V1'
        and intent['target']==receipt['target']==str(registry)
        and intent['request_id']==receipt['request_id']==key
        and intent['before_sha256']==receipt['before_sha256']==sha(before_bytes)
        and intent['proposed_sha256']==receipt['after_sha256']==sha(after_bytes)
        and intent['temporary']==str(registry.with_name(registry.name+'.source-once-'+key))
        and receipt['intent_sha256']==sha(intent_bytes)
        and receipt['status']=='CONFIRMED_SINGLE_METADATA_WRITE_NOT_CROSS_SITE_ADMISSION'
        and intent['native_dispatch_admitted']is receipt['native_dispatch_admitted']is False,
        'ACTUAL_ORIGINAL_ONCEIO_CAS_RECEIPT_NOT_A_PASS_FLAG')
    return dict(status='ORIGINAL_REGISTRATION_BYTES_AND_LOCAL_LEASE_REFERENCE_VERIFIED',
        role=d['role'],registry_snapshot_sha256=sha(after_bytes),local_lease=str(lease),
        live_canonical_registry_read=False,new_registry_authority=False,native_admission=False)


def registered_role(root,manifest,binding):
    """Read only sealed node-local provenance; never follow binding.registry."""
    root=Path(root)
    need(str(root)==binding['run_root'],'OWN_REGISTERED_ROOT')
    sealed=manifest['execution']['sealed_inputs']
    def read(name):
        p=PurePosixPath(name)
        need(not p.is_absolute()and '..'not in p.parts and str(p)==name and name in sealed,
            'PRESEALED_LOCAL_PROVENANCE_MEMBER')
        file=root/name
        need(not any(x.is_symlink()for x in(file,*file.parents)),'NO_PROVENANCE_SYMLINK')
        local_path(file)
        info=file.stat()
        need(stat.S_ISREG(info.st_mode)and info.st_uid==os.getuid()and not info.st_mode&0o222,
            'OWNED_IMMUTABLE_PROVENANCE_FILE')
        b=file.read_bytes();need(sha(b)==sealed[name],'IMMUTABLE_LOCAL_PROVENANCE_BYTES')
        return b
    descriptor=read(DESCRIPTOR);d=json.loads(descriptor)
    result=registration_bytes(descriptor,*(read(d[k])for k in('before_capture_relative','after_capture_relative',
        'intent_capture_relative','receipt_capture_relative')),binding=binding,manifest=manifest)
    return result


def registration(root,manifest,binding,*,legacy_read):
    """Select sealed custody only for an explicitly prepared immutable role."""
    if DESCRIPTOR in manifest['execution']['sealed_inputs']:
        return registered_role(root,manifest,binding)
    rows=[r for r in legacy_read(binding['registry'])['entries']if r.get('role')==binding['role']]
    need(len(rows)==1 and rows[0]['path']==str(root)and rows[0]['run_uuid']==Path(root).name,
        'ACTUAL_UNIQUE_LEGACY_REGISTRATION')
    return dict(live_canonical_registry_read=True,new_registry_authority=False,native_admission=False)


def immutable_local_member(root,manifest,name):
    """A sealed owned file on an existing local filesystem; no mount actions."""
    root=Path(root);relative=PurePosixPath(name)
    need(not relative.is_absolute()and '..'not in relative.parts and str(relative)==name,
        'EXACT_RELATIVE_LOCAL_MEMBER')
    p=root/name
    need(not any(x.is_symlink()for x in(p,*p.parents)),'NO_LOCAL_MEMBER_SYMLINK')
    local_path(p)
    info=p.stat();need(stat.S_ISREG(info.st_mode)and info.st_uid==os.getuid()
        and not info.st_mode&0o222,'IMMUTABLE_OWN_LOCAL_MEMBER')
    pins=manifest['execution']['sealed_inputs'];b=p.read_bytes()
    need(name in pins and sha(b)==pins[name],'PRESEALED_LOCAL_MEMBER_BYTES')
    return b


def local_path(path):
    """Reject NFS/unknown filesystems, using only kernel mount metadata."""
    p=Path(path);absolute(str(p))
    need(not any(x.is_symlink()for x in(p,*p.parents)),'NO_LOCAL_PATH_SYMLINK')
    matches=[]
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        before,after=line.split(' - ',1);mount=before.split()[4]
        # mountinfo escapes whitespace and backslashes as octal sequences.
        mount=re.sub(r'\\([0-7]{3})',lambda x:chr(int(x.group(1),8)),mount)
        if str(p)==mount or str(p).startswith(mount.rstrip('/')+'/'):
            matches.append((len(mount),after.split()[0]))
    need(bool(matches)and max(matches)[1]in('ext4','xfs','btrfs','zfs'),
        'EXISTING_NODE_LOCAL_FILESYSTEM_REQUIRED')
    return p


def native_assets(root,manifest,*,expected_original_sha256):
    """Verify all original asset bytes at explicit aliases for a new role."""
    original_name='plans/ASSET_IDENTITY.private.json';root=Path(root)
    original=(root/original_name).read_bytes()
    need(manifest['execution']['sealed_inputs'].get(original_name)==expected_original_sha256
        and sha(original)==expected_original_sha256,'ORIGINAL_ASSET_IDENTITY')
    pins=json.loads(original)['absolute_files']
    if DESCRIPTOR in manifest['execution']['sealed_inputs']:
        aliases=asset_relocations(original,immutable_local_member(root,manifest,ASSET_RELOCATION),
            expected_original_sha256=expected_original_sha256)
        for name,pin in pins.items():
            p=local_path(aliases[name]);need(p.is_file()and sha(p.read_bytes())==pin,'LOCAL_NATIVE_ASSET_BYTES')
    else:
        need(ASSET_RELOCATION not in manifest['execution']['sealed_inputs'],'NO_ORPHAN_ASSET_RELOCATION')
        need(len(pins)==26 and all(sha(Path(p).read_bytes())==h for p,h in pins.items()),
            'ALL_LEGACY_NATIVE_ASSET_BYTES')


def entry_directory(root,manifest,binding):
    """The new role pins copies of the unchanged entry, doctor and env table."""
    execution=manifest['execution'];relative=execution.get('node_local_entry_relative')
    if DESCRIPTOR not in execution['sealed_inputs']:
        need(relative is None,'NO_ORPHAN_LOCAL_ENTRY')
        return Path(binding['registry']).parent
    need(type(relative)is str and relative,'EXPLICIT_PRESEALED_LOCAL_ENTRY')
    for name in('enter.sh','doctor.py','ENVIRONMENTS.json'):
        b=immutable_local_member(root,manifest,str(PurePosixPath(relative)/name))
        if name!='ENVIRONMENTS.json':
            need(sha(b)==execution['entry_sha256'if name=='enter.sh'else'doctor_sha256'],
                'UNCHANGED_STANDARD_ENTRY_AND_DOCTOR')
    return Path(root)/relative


def asset_relocations(original_bytes,relocation_bytes,*,expected_original_sha256):
    """Explicit immutable byte aliases; no monkeypatch, bind mount or copy."""
    need(sha(original_bytes)==expected_original_sha256,'ORIGINAL_FROZEN_ASSET_IDENTITY_BYTES')
    original=json.loads(original_bytes);relocation=json.loads(relocation_bytes)
    need(relocation['schema']=='CORE_FIXED_ASSET_BYTE_RELOCATION_V1'
        and relocation['original_identity_sha256']==expected_original_sha256
        and set(relocation['aliases'])==set(original['absolute_files'])
        and len(original['absolute_files'])==26,'ALL_TWENTY_SIX_ORIGINAL_ASSET_MEMBERS')
    result={}
    for old,pin in original['absolute_files'].items():
        absolute(old);alias=relocation['aliases'][old]
        absolute(alias['local_path'])
        need(alias['sha256']==pin,'ORIGINAL_FILE_BYTES_NOT_A_NEW_ROBOT_ASSET')
        result[old]=alias['local_path']
    return result
