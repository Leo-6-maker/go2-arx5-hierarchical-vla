"""Measure where the original G7 max penalty sends gradient on real TRAIN outputs."""

import argparse
from collections import Counter
import hashlib
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write
from v7_d0_roster import HISTORY


R4 = "3eae531b9a0bea5cf2da8f3e19ebf77bb3ac152d9db0cd96a57b49210f7245e6"
STATISTICS = "f28d50e6eeea09b332d4bd159eca4333bbe23b185f95f1d336d45311e364ab3e"


def archived(site):
    from go2arx5_vla.sla_adapter.dense_g7_v6 import previous_command_context

    for wave, (_, terminal) in enumerate(HISTORY, 1):
        results = read(pin(site/f"RECOVERY-TRAIN-SEALED-{wave:02d}/RESULT.json",terminal))
        require(results["complete_reconciliation"] and results["actual_training_updates"] == 0,
                "The R4 TRAIN source did not close under its original owner")
        for entry in results["cases"]:
            case = site/f"RECOVERY-TRAIN-LIVE-{wave:02d}"/entry["id"]
            outcome = read(pin(case/"RESULT.private.json",entry["result"]["sha256"]))
            spec = read(case/"EXECUTION_SPEC.json")
            require(spec["collection_split"] == "TRAIN" and spec["evaluation_split"] == "TRAIN_RECOVERY"
                    and outcome["shutdown_clean"] and outcome["model_generations"] == entry["forwards"]
                    and read(case/"control/LEASE.json")["status"] == "CLOSED",
                    "Only closed original R4 TRAIN rollouts enter this read-side diagnostic")
            seen = 0
            for folder in sorted((case/"native-01/g7-queries").iterdir()):
                model = folder/"MODEL_RETURN.private.json"
                if not model.is_file():
                    continue
                result = read(model)["response"]
                action = pin(folder/"model-response/ACTION.npz",result["action"]["sha256"])
                require(result["comparison_checkpoint_sha256"] == R4 and result["new_generations_this_request"] == 1,
                        "Wrong frozen model or repeated generation identity")
                meta = read(folder/"INPUT_META.private.json")
                previous = previous_command_context(meta["previous_reference"])
                with np.load(action,allow_pickle=False) as arrays:
                    normalized = arrays["normalized"]
                require(normalized.shape == (1,40,32) and normalized.dtype == np.float32
                        and np.isfinite(normalized).all(), "Each witness must preserve the original full prediction")
                status = folder/"QUERY_ACTUAL.private.json"
                rejected = read(status).get("rejection") if status.is_file() else None
                yield entry["id"], int(folder.name), sha(action), normalized[0], previous, rejected
                seen += 1
            require(seen == entry["forwards"], "The diagnostic lost a paid original R4 forward")


def diagnostics(outputs, previous, stats):
    import jax
    import jax.numpy as jnp
    from go2arx5_vla.sla_adapter.dense_g7_v6 import GRIP_MARGIN, GRIP_LIMIT, grip_speed_loss

    lo, hi = (jnp.asarray(stats["actions"][key],dtype=jnp.float32) for key in ("q01","q99"))

    def original(normalized, prior):
        value, _ = grip_speed_loss(normalized[None],lo,hi,prior[None],xp=jnp,sigmoid=jax.nn.sigmoid)
        return value

    value, gradient = jax.jit(jax.vmap(jax.value_and_grad(original)))(
        jnp.asarray(outputs),jnp.asarray(previous,dtype=jnp.float32))
    latent = (jnp.asarray(outputs[...,6])+1)*.5*(hi[6]-lo[6]+1e-6)+lo[6]
    width = .002+.040*jax.nn.sigmoid(latent)
    speed = jnp.abs(jnp.diff(jnp.concatenate([jnp.asarray(previous,dtype=jnp.float32)[:,None],width],axis=-1),axis=-1))/.05
    values, grads, speeds = map(np.asarray,(value,gradient,speed))
    require(np.isfinite(values).all() and np.isfinite(grads).all() and np.isfinite(speeds).all()
            and not grads[...,:6].any() and not grads[...,7:].any(),
            "Unexpected derivative outside of the original physical gripper output")
    max_speed = speeds.max(axis=1)
    positive = values>0
    worst = speeds.argmax(axis=1)
    ties = (speeds == max_speed[:,None]).sum(axis=1)>1
    bins = {n:int(v.sum()) for n,v in (
        ("seam",positive & ~ties & (worst==0)),("within_p10",positive & ~ties & (worst>0) & (worst<10)),
        ("after_p10",positive & ~ties & (worst>=10)),("ties",positive & ties))}
    require(sum(bins.values()) == int(positive.sum()), "Incomplete argmax accounting")
    return dict(grip_training_margin_m_s=GRIP_MARGIN,original_limit_m_s=GRIP_LIMIT,
                positive_penalty=int(positive.sum()),inactive=int((~positive).sum()),active_argmax=bins,
                raw_seam_original_limit_exceeded=int((speeds[:,0]>GRIP_LIMIT).sum()),
                raw_full_h40_original_limit_exceeded=int((max_speed>GRIP_LIMIT).sum()),
                jaw_first_output_nonzero_gradient=int(np.count_nonzero(grads[:,0,6])),
                seam_p95_mm_s=float(np.percentile(speeds[:,0]*1000,95)))


def main(site, statistics_path, output):
    rows = list(archived(site))
    require(len(rows) == 907,"The fixed archive has 907 R4 TRAIN outputs")
    stats = read(pin(statistics_path,STATISTICS))
    checks = diagnostics(np.stack([r[3] for r in rows]),np.asarray([r[4] for r in rows]),stats)
    digest = hashlib.sha256()
    rejected = Counter()
    for case, query, action, _, _, reason in rows:
        digest.update(f"{case}/{query}/{action}\n".encode())
        if reason is not None:
            rejected[reason["kind"]] += 1
    write(output,dict(scope="ORIGINAL_R4_TRAIN_OUTPUT_GRIP_GRADIENT",checkpoint=R4,windows=len(rows),
          source_population_sha256=digest.hexdigest(),archived_named_rejections=dict(rejected),
          parameter_gradients_measured=False,model_forwards=0,training_updates=0,native_integrations=0,**checks))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("site","statistics_path","output"):
        p.add_argument(name,type=Path)
    main(**vars(p.parse_args()))
