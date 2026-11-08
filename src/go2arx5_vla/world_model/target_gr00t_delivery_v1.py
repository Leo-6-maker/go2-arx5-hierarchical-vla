"""Parent-bound GR00T inference package, without an external inference adapter.

The existing owner admits compute and transfers private artifacts. This module
only copies a completed checkpoint's learned action parameters and Processor,
then installs them into ordinary upstream GR00T tensors. Optimizer archives
remain at the training site; all frozen parent files must already be present.
"""
import json, shutil
from pathlib import Path

from . import target_gr00t_profile_v1 as profile
from .storage import digest, write_json

SCHEMA = 'WM02_TARGET_GR00T_PARENT_BOUND_INFERENCE_V1'


def _member(root, name):
    name = Path(name)
    profile.need(not name.is_absolute() and bool(name.parts)
                 and all(p not in ('.', '..') for p in name.parts), 'PACKAGE_RELATIVE_MEMBER')
    path = Path(root)/name
    profile.need(not path.is_symlink() and path.resolve().is_relative_to(Path(root).resolve())
                 and path.is_file(), 'PACKAGE_REGULAR_MEMBER:'+str(name))
    return path


def create(checkpoint, destination, spec, restore, *, guard):
    """Seal a production inference package only after actual full-state restore."""
    cp, out = Path(checkpoint), Path(destination)
    saved = json.loads((cp/'CHECKPOINT_COMPLETE.json').read_bytes())
    p = profile.validate(spec)
    profile.need(p['mode']=='FORMAL' and saved['global_step']==p['endpoint']==2000
                 and saved['generation_id']==spec['generation_id'] and saved['recipe']==p,
                 'FORMAL_ENDPOINT_NOT_DIAGNOSTIC_DELIVERY')
    profile.need(restore.get('checkpoint_sha256')==digest(cp/'CHECKPOINT_COMPLETE.json')
                 and all(restore.get(k) is True for k in ('model_parameters_restored',
                     'optimizer_restored','scheduler_restored','RNG_restored',
                     'fixed_batch_generation_restored')),
                 'ACTUAL_COMPLETE_RESTORE_BEFORE_PACKAGE')
    out.mkdir(exist_ok=False)
    for name, pin in saved['files'].items():
        if name not in ('target_parameters.pt', 'PARAMETER_CONTRACT.json', 'statistics.json',
                        'DATASET_MANIFEST.json') and not name.startswith('processor/'):
            continue
        guard(); source = _member(cp, name)
        profile.need(digest(source)==pin, 'PACKAGE_CHECKPOINT_MEMBER:'+name)
        target = out/name; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    for name in ('target_parameters.pt','PARAMETER_CONTRACT.json','statistics.json',
                 'DATASET_MANIFEST.json'):
        _member(out,name)
    profile.need((out/'processor').is_dir() and any((out/'processor').iterdir()),
                 'ACTUAL_SAVED_TARGET_PROCESSOR')
    write_json(out/'RESTORE_ACTUAL.json',restore)
    files={str(f.relative_to(out)):digest(f)for f in out.rglob('*')if f.is_file()}
    result=dict(schema=SCHEMA,profile=profile.PROFILE,generation_id=spec['generation_id'],
        global_step=saved['global_step'],checkpoint_sha256=digest(cp/'CHECKPOINT_COMPLETE.json'),
        parent_manifest_sha256=profile.PARENT,parent_files=spec['parent']['files'],
        upstream_sha=profile.UPSTREAM,recipe=p,source_sha=spec['source_sha'],files=files,
        action_horizon=40,action_period_s=.05,physics_dt_s=.002,
        representation='ORDINARY_UPSTREAM_INTERNAL_ACTION_PARAMETERS_REQUIRED_PARENT_PLUS_DELTA',
        baseline_changed=False,visual_language_backbone_frozen=True,
        external_inference_adapter=False,Core_DEV='NOT_RUN',
        reconstruction='REQUIRES_ACTUAL_INDEPENDENT_MODEL_LOAD_AND_MATCHED_GENERATION')
    write_json(out/'PACKAGE_MANIFEST.json',result)
    guard(); return dict(path=str(out),sha256=digest(out/'PACKAGE_MANIFEST.json'))


def verify(package, pin, parent_files):
    root=Path(package)
    profile.need(digest(_member(root,'PACKAGE_MANIFEST.json'))==pin,'EXACT_INFERENCE_PACKAGE')
    m=json.loads((root/'PACKAGE_MANIFEST.json').read_bytes());p=m['recipe']
    expected=profile.recipe(p['dataset_sha256'],p['statistics_sha256'],mode='FORMAL',
        variant=p['variant'],endpoint=2000,microbatch=p['microbatch'])
    profile.need(m['schema']==SCHEMA and m['parent_manifest_sha256']==profile.PARENT
        and m['parent_files']==parent_files and m['upstream_sha']==profile.UPSTREAM
        and m['global_step']==2000 and p==expected and m['external_inference_adapter']is False
        and m['action_horizon']==40 and m['action_period_s']==.05 and m['physics_dt_s']==.002,
        'EXACT_PARENT_AND_PRODUCTION_MODEL_CONTRACT')
    for name,pin in m['files'].items():
        profile.need(digest(_member(root,name))==pin,'INFERENCE_PACKAGE_MEMBER:'+name)
    profile.need(digest(root/'statistics.json')==p['statistics_sha256']
                 and digest(root/'DATASET_MANIFEST.json')==p['dataset_sha256'],
                 'TARGET_PROCESSOR_DATA_IDENTITIES')
    return m


def install(model, package, pin, parent_files):
    """Use an already admitted, strictly loaded ordinary parent, before CUDA.

    Every shape/name/dtype is checked before the first copy. Only the selected
    embodiment row of category tables and declared shared action tensors change.
    No wrapper is installed; Processor decoding remains upstream's single call.
    """
    import torch
    from scripts.pi0294.generalization.minimal_adapter import LEAVES,TargetCategoryLinear
    root=Path(package);manifest=verify(root,pin,parent_files)
    profile.need(not any(isinstance(m,TargetCategoryLinear)for m in model.modules()),
                 'ORDINARY_ORIGINAL_PARENT_REQUIRED')
    contract=json.loads((root/'PARAMETER_CONTRACT.json').read_bytes())
    rows=contract['trainable'];names=tuple(r['name']for r in rows)
    virtual=[];target_names=[]
    for name,_ in model.named_parameters():
        path,_,leaf=name.rpartition('.')
        if path in LEAVES and leaf in ('W','b'):
            name=path+'.target_'+leaf;target_names.append(name)
        virtual.append(name)
    profile.need(contract['variant']==manifest['recipe']['variant']
        and contract['target_category']==10
        and names==profile.whitelist(tuple(virtual),tuple(target_names),contract['variant']),
        'PACKAGE_EXACT_INTERNAL_ACTION_WHITELIST')
    state=torch.load(root/'target_parameters.pt',map_location='cpu',weights_only=True)
    profile.need(set(state)==set(names),'EXACT_ACTION_DELTA_NAMES')
    copies=[]
    for row in rows:
        name=row['name'];path,_,leaf=name.rpartition('.')
        if path in LEAVES and leaf in ('target_W','target_b'):
            table=model.get_parameter(path+'.'+leaf.removeprefix('target_'))
            profile.need(table.shape[0]>10,'ORIGINAL_TARGET_EMBODIMENT_ROW')
            target=table[10]
        else:target=model.get_parameter(name)
        value=state[name]
        profile.need(value.shape==target.shape and list(value.shape)==row['shape']
            and value.numel()==row['numel'] and str(value.dtype)==row['dtype']
            and value.dtype==target.dtype and torch.isfinite(value).all(),
            'EXACT_FINITE_ACTION_PARAMETER:'+name)
        copies.append((target,value))
    with torch.no_grad():
        for target,value in copies:target.copy_(value.to(target.device))
    model.requires_grad_(False);model.eval()
    return manifest


def compare_generations(first, second, *, tolerance=1e-6):
    """Predeclared component tolerance, not selected after observing the result."""
    import numpy as np
    a,b=Path(first),Path(second)
    one=json.loads((a/'RESULT.json').read_bytes())['records']
    two=json.loads((b/'RESULT.json').read_bytes())['records']
    profile.need(len(one)==len(two)==4,'FOUR_MATCHED_PACKAGE_RECONSTRUCTION_PROBES')
    maxima={k:0. for k in ('physical_H40','normalized')}
    for x,y in zip(one,two):
        profile.need(all(x[k]==y[k]for k in ('episode','split','anchor','seed','noise')),
                     'PACKAGE_SAME_REAL_INPUT_AND_NOISE')
        with np.load(a/(x['episode']+'.npz'),allow_pickle=False)as za, np.load(b/(y['episode']+'.npz'),allow_pickle=False)as zb:
            for key in maxima:
                profile.need(za[key].shape==zb[key].shape and np.isfinite(za[key]).all()
                    and np.isfinite(zb[key]).all(),'FINITE_MATCHED_PACKAGE_OUTPUT_SHAPE')
                maxima[key]=max(maxima[key],float(np.max(np.abs(za[key]-zb[key]))))
    profile.need(all(v<=tolerance for v in maxima.values()),'PACKAGE_RECONSTRUCTION_PREDICTION_MISMATCH')
    return dict(status='PASS_ACTUAL_INDEPENDENT_PARENT_DELTA_RECONSTRUCTION',
        actual_matched_queries=4,absolute_component_tolerance=tolerance,max_abs_error=maxima,
        Core_reconstruction='NOT_RUN',native_task_success='NOT_MEASURED_BY_RECONSTRUCTION')
