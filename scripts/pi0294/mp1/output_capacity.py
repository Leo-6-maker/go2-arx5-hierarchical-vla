"""Evidence-bound output estimates; never widen the adopted MP1 resource cap."""
from pathlib import Path

from scripts.pi0292.durable import read
from scripts.pi0294.native_dataset import sha

GIB=1024**3
AUTHORITY='c8cdf60e41e31da5796b44a064ef2f22d594a491'
RUNTIME='03646938ece5c6fa5f09b0fd00732f8159a51468'


def checked_estimate(root,path,management_manifest):
    root=Path(root).absolute();path=Path(path).absolute()
    if (path.is_symlink() or not path.resolve().is_relative_to((root/'plans').resolve())
            or path.stat().st_mode&0o222):
        raise PermissionError('IMMUTABLE_OWN_OUTPUT_ESTIMATE_BINDING')
    q=read(path);stage=read(root/'MP1_STAGE_BINDING.json');original=root/'RUN_MANIFEST.json';launch=read(original)
    if (q['schema']!='MP1_EVIDENCED_SOURCE_OUTPUT_ESTIMATE_REBINDING_V1'
            or q['authority_release']!=AUTHORITY or q['parent_run']!=root.parent.parent.name
            or q['source_child']!=root.name or root.name!='source-pair-01'
            or q['original_T0_epoch']!=read(root.parent.parent/'MP1_ADOPTION_AND_INCREMENTAL_BUDGET.json')['T0_epoch']
            or stage['schema']!='MP1_SOURCE_SHARED_STAGE_V1'
            or q['old_estimate_bytes']!=140*GIB or q['protocol_peak_limit_bytes']!=400*GIB
            or type(q['new_estimate_bytes']) is not int or not 140*GIB<q['new_estimate_bytes']<=400*GIB
            or any(q[k] is not False for k in ['new_scientific_budget','scientific_inputs_changed','deletion_required'])
            or q['launch_source_sha']!=RUNTIME or launch['code']['vla_commit']!=RUNTIME
            or sha(original)!=q['original_run_manifest_sha256']):
        raise PermissionError('ONLY_EVIDENCED_ESTIMATE_WITHIN_ORIGINAL_SOURCE_MP1_CAP')
    if {k:v for k,v in launch.items() if k!='code'}!={k:v for k,v in management_manifest.items() if k!='code'}:
        raise PermissionError('ORIGINAL_LAUNCH_SCIENTIFIC_MANIFEST_UNCHANGED')
    management_keys={'vla_commit','capsule_relative','source_seal_relative','source_seal_sha256'}
    if ({k:v for k,v in launch['code'].items() if k not in management_keys}
            !={k:v for k,v in management_manifest['code'].items() if k not in management_keys}):
        raise PermissionError('ORIGINAL_UPSTREAM_AND_FROZEN_NUMERICS_UNCHANGED')
    code=root/launch['code']['capsule_relative'];seal=root/launch['code']['source_seal_relative']
    if (not code.resolve().is_relative_to(root.resolve()) or code.is_symlink()
            or sha(seal)!=launch['code']['source_seal_sha256']):
        raise PermissionError('ORIGINAL_LAUNCH_CAPSULE_AND_SEAL')
    sealed=read(seal)
    if sealed['source_sha']!=RUNTIME:raise PermissionError('ORIGINAL_LAUNCH_SOURCE_SHA')
    for name,digest in sealed['files'].items():
        p=code/name
        if p.is_symlink() or not p.resolve().is_relative_to(code.resolve()) or sha(p)!=digest:
            raise PermissionError('ORIGINAL_LAUNCH_SOURCE_MEMBER_CHANGED')
    for name,digest in launch['execution']['sealed_inputs'].items():
        p=root/name
        if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()) or sha(p)!=digest:
            raise PermissionError('ORIGINAL_LAUNCH_SCIENTIFIC_INPUT_CHANGED')
    evidence=q['checkpoint_size_evidence'];seen=set();sizes=[]
    for item in evidence:
        key=(item['branch'],item['step'])
        if key in seen or item['branch'] not in ['S-BC','S-CORR'] or type(item['step']) is not int or not 0<item['step']<=2048:
            raise PermissionError('UNIQUE_EXISTING_OWN_CHECKPOINT_SIZE_EVIDENCE')
        seen.add(key);folder=root/'training'/item['branch']/('checkpoint-'+str(item['step']));p=folder/'CHECKPOINT_COMPLETE.json'
        if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()) or sha(p)!=item['manifest_sha256']:
            raise PermissionError('OUTPUT_ESTIMATE_CHECKPOINT_MANIFEST_CHANGED')
        complete=read(p)
        if complete['global_step']!=item['step'] or complete['full_resume'] is not True:
            raise PermissionError('ONLY_COMPLETE_NATIVE_CHECKPOINT_SIZE_EVIDENCE')
        size=0
        for name in complete['files']:
            member=folder/name
            if member.is_symlink() or not member.resolve().is_relative_to(folder.resolve()):
                raise PermissionError('OUTPUT_SIZE_MEMBER_ESCAPE')
            size+=member.stat().st_size
        if size!=item['logical_member_bytes']:raise PermissionError('OUTPUT_SIZE_EVIDENCE_CHANGED')
        sizes.append(size)
    if (not {(b,s) for b,s in [('S-BC',512),('S-BC',2048),('S-CORR',512)]}.issubset(seen)
            or q['maximum_total_complete_checkpoint_slots']!=7 or not len(sizes)<=7
            or q['noncheckpoint_margin_bytes']<2*GIB
            or q['new_estimate_bytes']<7*max(sizes)+q['noncheckpoint_margin_bytes']):
        raise PermissionError('CONSERVATIVE_CHECKPOINT_PROJECTION_AND_NONCHECKPOINT_MARGIN')
    return q['new_estimate_bytes'],launch
