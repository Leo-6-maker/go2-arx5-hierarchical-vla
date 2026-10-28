"""Retain all raw evidence inside the original colocated output partitions."""
from pathlib import Path
import os

from . import mix_vanda_cohost as storage,mix_accounting as a


def need(ok,why):
    if not ok:raise PermissionError('V5_VANDA_EVAL_OUTPUT_'+why)


def reference_view(root,plan):
    """Only the exact ten immutable input names may be excluded as inputs."""
    root=Path(root);view=root/'model_views'
    need(view.is_dir() and not view.is_symlink(),'BOUND_INPUT_VIEW_PARENT')
    actual={str(p.relative_to(view)) for p in view.rglob('*') if p.is_file() or p.is_symlink()}
    expected={plan['arm']+'/'+n for n in [*plan['files'],'CHECKPOINT_COMPLETE.json']}
    need(actual==expected,'NO_OUTPUT_OR_EXTRA_PAYLOAD_HIDDEN_IN_EXCLUDED_MODEL_INPUTS')
    from .mix_vanda_reference import metadata
    for name,value in plan['files'].items():
        p=view/plan['arm']/name
        need(p.is_symlink() and p.resolve()==Path(value['path']) and metadata(value['path'])['bytes']==value['bytes'],
             'EXACT_UNMODIFIED_ORIGINAL_REFERENCE_INPUT')
    need(a.sha((view/plan['arm']/'CHECKPOINT_COMPLETE.json').read_bytes())==plan['checkpoint_manifest_sha256'],'ORIGINAL_COMPLETE_INPUT_BYTES')
    return view


def cosmos_view(root):
    """One original official-backbone input alias, never an output escape."""
    import json
    root=Path(root);folder=root/'model_inputs'
    if not folder.exists():return None
    need(folder.is_dir()and not folder.is_symlink()and (folder/'nvidia').is_dir()
         and not(folder/'nvidia').is_symlink(),'REAL_COSMOS_INPUT_NAMESPACE')
    path=root/'NEURAL_BINDINGS.private.json';manifest=root/'RUN_MANIFEST.json'
    need(path.is_file()and not path.is_symlink()and not path.stat().st_mode&0o222,'SEALED_COSMOS_BINDINGS')
    m=json.loads(manifest.read_bytes());need(m['execution']['sealed_inputs'].get(path.name)==a.sha(path.read_bytes()),'EXACT_COSMOS_BINDING_BYTES')
    from .mix_vanda_eval_identity import cold_profile
    need(cold_profile(m),'ONLY_NEW_TYPED_MODEL_INPUT_ALIAS')
    actual={str(p.relative_to(folder))for p in folder.rglob('*')}
    need(actual=={'nvidia','nvidia/Cosmos-Reason2-2B'},'NO_HIDDEN_OUTPUT_IN_COSMOS_INPUT_ROLE')
    alias=folder/'nvidia/Cosmos-Reason2-2B';bindings=json.loads(path.read_bytes())
    need(alias.is_symlink()and alias.lstat().st_uid==os.getuid()
         and alias.resolve(strict=True)==Path(bindings['cosmos']).resolve(strict=True),'EXACT_EXISTING_PINNED_BACKBONE_INPUT')
    return folder


def enforce(neural,physical,owner,reference_plan):
    neural,physical,owner=map(Path,(neural,physical,owner));contract=storage.storage_contract()
    need(neural.parent.name=='neural-children' and physical.parent.name=='children' and owner.parent.name=='control'
         and neural.name==physical.name and neural.parent.parent==physical.parent.parent==owner.parent.parent,
         'ACTUAL_DISTINCT_SAME_CAMPAIGN_OUTPUT_ROLES')
    view=reference_view(neural,reference_plan)
    backbone=cosmos_view(neural)
    episodes=physical/'episodes';need(episodes.is_dir() and not episodes.is_symlink(),'REAL_ORIGINAL_EPISODE_NAMESPACE')
    folders=list(episodes.iterdir());need(len(folders)<=26 and all(p.is_dir() and not p.is_symlink() for p in folders),'ORIGINAL_REGULAR_AND_REPLACEMENT_BOUND')
    physical_rows={p.name:storage.enforce_output(p,contract['physical_per_attempt_bytes']) for p in folders}
    query=[];label=reference_plan['arm']+'-DEV-N294101'
    for p in (neural/'evidence'/label/'resident',owner/'resident'):
        if p.exists():
            need(not p.is_symlink(),'REAL_ADDITIONAL_QUERY_AND_OWNER_JOURNAL_NAMESPACE')
            query.append((p,storage.tree_bytes(p)))
    query_upper=contract['query_calls']*contract['additional_per_query_bytes']
    need(sum(max(v['logical_bytes'],v['allocated_bytes']) for _,v in query)<=query_upper,'ADDITIONAL_QUERY_OUTPUT_OVERFLOW_PRESERVE_ALL')
    remaining=[]
    for root,excluded in [(neural,[view,*([backbone]if backbone else[]),*[p for p,_ in query if p.is_relative_to(neural)]]),
                          (physical,[episodes]),(owner,[p for p,_ in query if p.is_relative_to(owner)])]:
        remaining.append(storage.tree_bytes(root,exclude=excluded))
    # Finite audit-tool delivery is retained alongside this panel, including
    # its exclusive input chunks. It cannot disappear from peak accounting
    # merely because it is outside the three original worker roots.
    codec=physical/'plans/VANDA_AUDIT_CODEC_TOOLCHAIN.private.json'
    if codec.is_file():
        import json
        metadata=json.loads(codec.read_bytes());aux=metadata.get('delivery_preparation_root')
        if aux is not None:
            root=Path(aux)
            need(root.parent==owner.parent and root.name=='v5-cd-c50-vanda-codec-cold-01'
                 and root.is_dir() and not root.is_symlink(),'EXACT_RETAINED_FINITE_AUDIT_TOOL_INPUT_ROOT')
            remaining.append(storage.tree_bytes(root))
    other=sum(max(v['logical_bytes'],v['allocated_bytes']) for v in remaining)
    need(other<=contract['other_artifacts_bytes'],'OTHER_RETAINED_OUTPUT_OVERFLOW_PRESERVE_ALL')
    return dict(schema='V5_CD_COLOCATED_ACTUAL_RETAINED_OUTPUT_READBACK_V1',
        physical_attempts=physical_rows,additional_query_namespaces={str(p.relative_to(neural.parent.parent)):v for p,v in query},
        other_artifacts_bytes=other,full_contract=contract,raw_deleted_or_truncated=False,original_model_payload_bytes_copied=0)
