"""Extract only six inherited training-only projector leaves on CPU."""
import json
from pathlib import Path
import sys
import numpy as np
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from scripts.pi0292.durable import write


def run(spec):
    import flax.nnx as nnx
    from flax.traverse_util import flatten_dict
    import jax
    import orbax.checkpoint as ocp
    from openpi.models.projectors import AlignProjector
    from types import SimpleNamespace
    if any(d.platform != 'cpu' for d in jax.devices()):
        raise ValueError('CPU_ONLY_PROJECTOR_ASSET_EXTRACTION_REQUIRED')
    parent = read_pin(spec['parent_manifest'])
    if not parent['all_files_verified'] or parent['revision'] != '5bbdf3012ab643554aa2eb83ddeae9ace3dce476':
        raise ValueError('VERIFIED_SAME_SLA_BASE_PARENT_REQUIRED')
    params = Path(spec['parent_params'])
    if params.resolve() != (Path(spec['parent_manifest']['path']).parent/'model/params').resolve():
        raise ValueError('EXACT_PARENT_PATH_REQUIRED')
    with ocp.PyTreeCheckpointer() as checkpoint:
        metadata = checkpoint.metadata(params)
        item = {'params': {'align_projector': metadata['params']['align_projector']}}
        args = jax.tree.map(lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray), item)
        # Orbax0.11.13 uses the transformations path for partial restoration.
        # An empty map retains matching keys in this explicitly reduced item.
        loaded = checkpoint.restore(params, args=ocp.args.PyTreeRestore(item=item, restore_args=args, transforms={}))
    values = flatten_dict(loaded['params']['align_projector'])
    if not all(p[-1] == 'value' for p in values):
        raise ValueError('ORIGINAL_NNX_PARENT_PROJECTOR_TREE_REQUIRED')
    values = {p[:-1]:v for p,v in values.items()}
    config = SimpleNamespace(vggt_dim=1024, vla_adapt_layers_align=False, vggt_adapt_layers_align=False,
                             iSF_zscore=False, use_vlm_norm=True)
    abstract = nnx.eval_shape(lambda: AlignProjector(2048, config, nnx.Rngs(20261112)))
    expected = flatten_dict(nnx.state(abstract).to_pure_dict())
    if set(values) != set(expected) or len(values) != 6:
        raise ValueError('EXACT_SIX_INHERITED_PROJECTOR_LEAVES_REQUIRED')
    root = Path(spec['registered_root'])/'projector'
    root.mkdir(exist_ok=False)
    rows = []
    for index, (path,value) in enumerate(sorted(values.items())):
        if value.shape != expected[path].shape or not np.isfinite(value).all():
            raise ValueError('INHERITED_PROJECTOR_TEMPLATE_OR_FINITE_CHECK_FAILED')
        a = np.asarray(value,dtype=np.float32)
        file = root/f'{index:02d}.npy'
        np.save(file,a,allow_pickle=False)
        rows.append(dict(path=list(path),file=file.name,shape=list(a.shape),dtype=str(a.dtype),
                         original_dtype=str(value.dtype),sha256=sha(file),numel=a.size))
    manifest = dict(schema='SLA_INHERITED_ALIGN_PROJECTOR_ASSET_V1',parent_manifest=spec['parent_manifest'],
                    arrays=rows,random_initialization=False,full_SLA_parent_model_loads=0,
                    teacher_model_loads=0,projector_leaves_restored=6,optimizer_updates=0,physical_requests=0)
    write(root/'MANIFEST.private.json',manifest)
    return dict(schema='SLA_INHERITED_PROJECTOR_ASSET_ACTUAL_V1',passes=True,
                projector=dict(path=str(root),sha256=sha(root/'MANIFEST.private.json')),
                inherited_leaves=6,inherited_numel=sum(r['numel']for r in rows),
                full_SLA_model_loads=0,teacher_loads=0,optimizer_updates=0,physical_requests=0)


if __name__ == '__main__':
    spec=json.loads(Path(sys.argv[1]).read_bytes())
    write(Path(spec['registered_root'])/'RESULT.private.json',run(spec))
