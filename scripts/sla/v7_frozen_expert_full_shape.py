"""Read real endpoint shapes and trace optimizer initialization without model values or updates."""

import hashlib
import json
from pathlib import Path
import sys
from tempfile import mkdtemp

from v7_frozen_expert_optimizer import make_frozen_expert_adamw


def run(work):
    import jax
    from flax import nnx
    import numpy as np
    import orbax.checkpoint as ocp

    assert jax.default_backend() == "cpu"
    root = Path("/mnt/local/hdd1/UserData/liuyu-nfs/Projects/GR00T-WBC/WM02/v7-b6-p250.dWnpqo")
    proof = root / "CORE-PROVIDERS/ACTUAL.json"
    assert hashlib.sha256(proof.read_bytes()).hexdigest() == "c22d2bf4dd6c05411cea0d3b0874254c366bc2b0d4a1aec6e44074151ada7771"
    path = root / "LAMBDA_ZERO/STEP0250"
    with ocp.PyTreeCheckpointer() as reader:
        metadata = reader.metadata(path).tree
    # snapshot() stores pure parameter dictionaries, while live learner leaves
    # are NNX Param states. Rebuild that real variable ABI before tracing Adam;
    # using plain arrays would omit every moment's serialized /value node.
    parameters = {group:nnx.State(jax.tree.map(
        lambda m:nnx.Param(jax.ShapeDtypeStruct(m.shape,m.dtype)).to_state(),values))
        for group,values in metadata["parameters"].items()}
    initialize,_ = make_frozen_expert_adamw(beta1=.9,beta2=.999,eps=1e-8,weight_decay=.01,clip_norm=1.)
    states = jax.eval_shape(initialize,parameters)

    def signature(tree):
        def component(key):
            for name in ("key","name","idx"):
                if hasattr(key,name):return str(getattr(key,name))
            raise ValueError("Unknown pytree path")
        return {"/".join(component(k) for k in keys):dict(shape=list(value.shape),dtype=str(value.dtype))
                for keys,value in jax.tree_util.tree_flatten_with_path(tree)[0]}

    actual,original = signature(states),signature(metadata["optimizer"])
    assert actual == original,"The complete original optimizer state ABI changed"
    counts = {key:len(jax.tree.leaves(value)) for key,value in parameters.items()}
    elements = {key:sum(int(np.prod(x.shape)) for x in jax.tree.leaves(value)) for key,value in parameters.items()}
    out = Path(mkdtemp(prefix="v7-frozen-expert-full-shape.",dir=work))/"ACTUAL.json"
    result = dict(status="ACTUAL_FULL_PARAMETER_SHAPES_AND_OPTIMIZER_STATE_ABI_PASS_ZERO_UPDATES",
        endpoint_metadata_source=str(path),complete_core_restore_proof_sha256=hashlib.sha256(proof.read_bytes()).hexdigest(),
        parameter_leaves=counts,parameter_elements=elements,optimizer_leaf_shapes=actual,
        original_optimizer_state_ABI_equal=True,traced_only_initializer=True,
        model_weight_values_read=False,actual_model_forwards=0,actual_model_optimizer_updates=0,
        full_model_gradient_or_H200_update_qualified=False,physics_integrations=0)
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(out,hashlib.sha256(out.read_bytes()).hexdigest(),flush=True)
    print(json.dumps({k:v for k,v in result.items() if k!="optimizer_leaf_shapes"}),flush=True)


if __name__ == "__main__":
    run(Path(sys.argv[1]))
