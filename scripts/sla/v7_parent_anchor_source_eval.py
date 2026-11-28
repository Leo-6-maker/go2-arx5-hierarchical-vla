"""Measure a cold-restored B6 arm on the unchanged independent source retention bank."""

import argparse
from contextlib import closing
import hashlib
import os
from pathlib import Path
import sys
import time

import numpy as np

from m38_gpu_preflight_v1 import bind_imports, check_gpu, pin, read, require, sha
from scripts.pi0292.durable import OwnerLock, write
from v62_source_dev_eval import (DEV_ROSTER, PAIRED_R4_DEV, R4_READBACK, fingerprint, load_inputs,
                                 restore_groups, summarize)
from v7_core_endpoint import PLAN, R4
from v7_parent_anchor_endpoint import ARMS
from v7_feedback_eval import check_pair


def run(worker, retained, roster, dual, single, video, baseline, training, providers, pair_sha, out, arm):
    import flax.nnx as nnx
    from flax.traverse_util import flatten_dict, unflatten_dict
    import jax
    import jax.numpy as jnp

    spec = read(worker)
    bind_imports(spec)
    from go2arx5_vla.sla_adapter import batch_v2, full_h40_v6, scale20k_model
    from go2arx5_vla.sla_adapter.model_v1 import Tokenizer
    from go2arx5_vla.sla_adapter.scale20k_contract import canonical_hash
    from go2arx5_vla.sla_adapter.scale20k_training import host_batch
    from openpi.models import pi0
    from scripts.sla.core_hopper_native_scope import accepted
    from scripts.sla.mfree_c20_vectorfield_v3 import extracted_fields
    from v3_main_place_learn import endpoint_files
    from v65_recovery_endpoint import r4_provider
    from v6_dual_course import initialize_single

    require(arm in ARMS,"Invalid fixed arm")
    audit = providers / "ACTUAL.json"
    pair = read(pin(audit, pair_sha))
    require(pair["scope"] == "V7_B6_SHARED_L2SP_MATCHED_V1" and pair["complete_training_endpoint"]
            and pair["step"] in (100,250,500) and set(pair["arms"]) == set(ARMS),
            "Require the independently restored matched B6 pair")
    step = pair["step"]
    proof = pair["arms"][arm]
    seal = read(pin(training/arm/f"STEP{step:04d}_SEALED.json",proof["endpoint_sha256"]))
    checkpoint = training/arm/f"STEP{step:04d}"
    require(seal["files"] == endpoint_files(checkpoint),"Full source optimizer checkpoint changed")
    r4 = read(pin(retained/"R4_STEP0200_PROVIDER_READBACK.json",R4_READBACK))
    r4_endpoint = Path(r4["endpoint"])
    r4_seal = read(pin(r4_endpoint.parent/"STEP0200_SEALED.json",R4))
    require(r4_seal["files"] == endpoint_files(r4_endpoint),"Inherited Ville IO bytes changed")
    r4_history = read(pin(baseline,PAIRED_R4_DEV))
    require(r4_history["roster"]["sha256"] == DEV_ROSTER and r4_history["model_forwards"] == 84
            and r4_history["physics_ticks"] == r4_history["optimizer_updates"] == 0,
            "Reuse only the original source baseline with actual CUDA evidence")
    original = {name:{item["index"]:item for item in v["rows"]} for name,v in r4_history["domains"].items()}
    data = load_inputs(dual,single,video,roster)
    _,_,receiver = accepted(spec)
    provider = read(pin(providers/arm/"CHECKPOINT.json",proof["provider_sha256"]))

    with closing(OwnerLock(Path(spec["neural_owner_lock"]))):
        require(sys.executable == spec["python"] and os.uname().nodename == spec["host"] and os.getuid() == spec["uid"]
                and all(os.environ.get(key) == spec["gpu_environment"][key] for key in
                        ("CUDA_VISIBLE_DEVICES","JAX_PLATFORMS","XLA_FLAGS","XLA_PYTHON_CLIENT_MEM_FRACTION")),
                "Use the same Corelab CUDA numerical and ownership binding")
        gpu = check_gpu(spec["GPU_uuid"],spec["allocator_limit_bytes"],spec["minimum_host_available_bytes"])
        require(jax.default_backend() == "gpu" and len(jax.local_devices()) == 1,"Need the original leased GPU")
        out.mkdir(mode=0o700)
        jax.config.update("jax_compilation_cache_dir",str(out/"jit-cache"))
        write(out/"START.json",dict(plan=PLAN,arm=arm,pair_sha256=sha(audit),gpu=gpu,max_forwards=42))
        asset = receiver["model"]
        graphs,template,frozen,identity = scale20k_model.load(asset["parent_manifest"],asset["parent_params"])
        require(asset["parent_manifest"]["sha256"] == provider["foundation_sha256"]
                and identity["frozen_fingerprint"] == provider["frozen_model"]
                and canonical_hash(identity["inventory"]) == provider["partition_sha256"],"Frozen source foundation changed")
        tokenizer = Tokenizer(str(pin(receiver["tokenizer"]["path"],provider["tokenizer_sha256"])))
        _,_,lora,digest = r4_provider(retained/"R4_STEP0200_PROVIDER")
        require(digest == provider["parent_lora_sha256"] == seal["lora_sha256"],"Frozen original LoRA differs")
        parameters = restore_groups(checkpoint,("shared","robodojo"),step)
        inherited = {key[1:]:array for key,array in lora.items()}
        shared = flatten_dict(parameters["shared"])
        require(not inherited.keys() & shared.keys(),"Frozen parameters reached the formal optimizer")
        states = {}
        for name,value in {"shared":unflatten_dict(shared|inherited),"robodojo":parameters["robodojo"],
                           "target":restore_groups(r4_endpoint,("target",),200)["target"]}.items():
            state = jax.tree.map(lambda x:x,template[name])
            state.replace_by_pure_dict(value)
            states[name] = state
        fields,_ = extracted_fields(batch_v2,pi0)
        output = {}
        limit = time.monotonic()+1800
        for domain,(reader,rows) in data.items():
            graph_role = "target" if domain == "single" else domain
            private = initialize_single(states["target"]) if domain == "single" else states["robodojo"]
            bound = nnx.State.merge(states["shared"],private)
            sample = full_h40_v6.make_sampler(graphs[graph_role],fields,rematerialize=False,precision="highest")
            measured = []
            for row in rows:
                require(time.monotonic() < limit and not (out/"STOP.json").exists(),"Source evaluation lease ended")
                index,horizon,dimensions = (row[key] for key in ("index","horizon","dimensions"))
                require(reader.rows[index] == (row["episode"],row["anchor"])
                        and fingerprint(reader[index],horizon) == row["hashes"],"Source frame or label changed")
                obs,actions = host_batch(reader,tokenizer,graph_role,[index],[index],training=False)
                noise = jax.random.normal(jax.random.PRNGKey(row["evaluation_noise_seed"]),
                                          (1,horizon,32),dtype=jnp.float32)
                digest = hashlib.sha256(np.asarray(noise).tobytes()).hexdigest()
                prior = original[domain][index]
                require(prior["actual_noise_sha256"] == digest and all(prior[key] == val for key,val in row.items()),
                        "Actual source evaluation noise or row changed")
                prefix = out/f"{domain}-{index:07d}"
                write(prefix.with_suffix(".ENTERED.json"),dict(row,actual_noise_sha256=digest))
                prediction = np.asarray(sample(bound,frozen,obs,noise))
                require(prediction.shape == (1,horizon,32) and np.isfinite(prediction).all(),"Invalid native-domain output")
                values = dict(row,actual_noise_sha256=digest,mae=dict(r4_init=prior["mae"]["r4_init"],
                     v7=float(np.abs(prediction[0,:,:dimensions]-np.asarray(actions)[0,:,:dimensions]).mean())),
                     prediction_sha256=dict(r4_init=prior["prediction_sha256"]["r4_init"],
                                            v7=hashlib.sha256(prediction.tobytes()).hexdigest()))
                write(prefix.with_suffix(".RETURNED.json"),values)
                measured.append(values)
            output[domain] = dict(summary=summarize(measured,"v7"),rows=measured)
        require([len(output[key]["rows"]) for key in ("robodojo","single")] == [24,18],
                "Missing original source retention rows")
        change = {name:d["summary"]["v7_mae"]/d["summary"]["r4_init_mae"]-1 for name,d in output.items()}
        write(out/"RESULT.json",dict(plan=PLAN,scope="V7_B6_SOURCE_RETENTION",step=step,arm=arm,roster_sha256=DEV_ROSTER,checkpoint=provider["checkpoint_sha256"],
              pair_sha256=pair_sha,model_forwards=42,physics_ticks=0,optimizer_updates=0,domains=output,
              relative_mae=change,retention_alert_domains=[name for name,value in change.items() if value>.1],
              single_role="R4_FIXED_INITIAL_VILLE_IO_SHARED_REPRESENTATION_ONLY"))


if __name__ == "__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ("worker","retained","roster","dual","single","video","baseline","training","providers","out"):
        cli.add_argument(name,type=Path)
    cli.add_argument("pair_sha")
    cli.add_argument("arm",choices=ARMS)
    run(**vars(cli.parse_args()))
