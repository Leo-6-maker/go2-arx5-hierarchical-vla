"""Recover all source-domain weights and optionally layer the exact V3 Go2 endpoint."""

import hashlib
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from v3_main_place_learn import endpoint_files
from v4_main250_hopper import foundation_paths


def load_parent(custody, archive, archive_sha256, reference, *, v3_endpoint=None):
    import jax
    from flax.traverse_util import flatten_dict, unflatten_dict
    from go2arx5_vla.sla_adapter import scale20k_model, scale20k_package
    from go2arx5_vla.sla_adapter.scale20k_contract import canonical_hash

    complete = read(pin(archive / "COMPLETE.json", archive_sha256))
    anchor = read(reference / "CHECKPOINT.json")
    require(complete["progress"]["phase_step"] in (3800, 4300)
            and len(complete["arrays"]) == 140, "The parent must be an available original complete endpoint")
    require(len({r["key"] for r in complete["arrays"]}) == 140, "Parent array names are ambiguous")
    for row in complete["arrays"]:
        member = archive / row["file"]
        require(member.resolve(strict=True).is_relative_to(archive.resolve(strict=True)),
                "Parent member left its archived directory")
        tensor = np.load(pin(member, row["sha256"]), allow_pickle=False, mmap_mode="r")
        require(list(tensor.shape) == row["shape"] and str(tensor.dtype) == row["dtype"],
                "Parent optimizer, RNG, or parameter tensor identity differs")

    manifest, params = foundation_paths(custody)
    graphs, template, frozen, identity = scale20k_model.load(manifest, params)
    require(identity["frozen_fingerprint"] == anchor["frozen_model"]
            and canonical_hash(identity["inventory"]) == anchor["partition_sha256"],
            "Original frozen parent or parameter scope changed")
    records = [row for row in complete["arrays"] if row["key"].startswith("['parameters']")]
    parameters = scale20k_package.read_weights(archive, records, template)
    handoff = None
    if v3_endpoint is not None:
        import orbax.checkpoint as ocp
        from v3_main_place_learn import MAIN

        require(archive_sha256 == MAIN, "A V3 update cannot be overlaid on an unrelated parent")
        terminal_path = v3_endpoint / "TRAIN_TERMINAL.json"
        terminal = read(terminal_path)
        path = Path(terminal["endpoint"]).resolve(strict=True)
        require(terminal["formal_endpoint"] and terminal["completed_updates"] == 250
                and path.parent == v3_endpoint.resolve(strict=True) and path.name == "STEP0250"
                and endpoint_files(path) == terminal["files"], "The registered exact V3 raw endpoint is required")
        # Its optimizer is kept in custody. V4 always initializes a new optimizer.
        with ocp.PyTreeCheckpointer() as saver:
            metadata = saver.metadata(path).tree
            active_args = jax.tree.map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), metadata["active"])
            state = saver.restore(path, args=ocp.args.PyTreeRestore(
                item={"active": metadata["active"], "step": metadata["step"]},
                restore_args={"active": active_args, "step": ocp.RestoreArgs(restore_type=int)},
                transforms={}))
        active = flatten_dict(state["active"])
        by_path = {tuple(row["path"].split("/")): row["group"] for row in identity["inventory"]}
        mission = {k for k,v in by_path.items() if v in {"action_expert", "domain_template"}}
        require(state["step"] == 250 and set(active) == mission and len(active) == 19,
                "Wrong V3 reusable action-side partition")
        for group, scope in (("shared", "action_expert"), ("target", "domain_template")):
            old = flatten_dict(parameters[group].to_pure_dict())
            for key in active:
                if by_path[key] == scope:
                    value = np.asarray(active[key])
                    require(value.shape == old[key].shape and value.dtype == old[key].dtype
                            and np.isfinite(value).all(), "Incompatible V3 tensor")
                    old[key] = jax.device_put(value)
            parameters[group].replace_by_pure_dict(unflatten_dict(old))
        handoff = sha(terminal_path)
    row_hash = hashlib.sha256("".join(r["sha256"] for r in records).encode()).hexdigest()
    return graphs, parameters, frozen, dict(parent_sha256=archive_sha256,
        source_parameter_archive_sha256=row_hash, inherited_v3_terminal_sha256=handoff,
        inherited_progress=complete["progress"], verified_array_count=140, new_optimizer=True)
