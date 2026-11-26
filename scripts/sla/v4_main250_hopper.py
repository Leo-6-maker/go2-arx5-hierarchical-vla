"""Run the inherited V3 MAIN experiment under one real Hopper PBS allocation."""

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import threading
import time

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from v3_main_place_learn import BATCH, MAIN, SEED, UPDATES, data_scope, endpoint_files, gradient, sample_bank


def foundation_paths(custody):
    """Bind the original foundation bytes through the production Hopper layout."""
    from go2arx5_vla.sla_adapter.scale20k_model import parent_asset
    from scripts.sla.hopper_inputs import inputs

    base = custody / "assets.verified"
    recovered = custody / "foundation-layout-v2"
    original = inputs(base)
    migration = read(recovered / "TECHNICAL_RECOVERY.json")
    manifest = original["parent_manifest"]
    require(migration["manifest"] == manifest["sha256"],
            "The existing Hopper foundation recovery used a different manifest")

    def recover(path):
        return recovered / "assets.verified" / Path(path).relative_to(base)

    pinned = dict(manifest, path=str(recover(manifest["path"])))
    params = recover(original["parent_params"])
    original_asset = parent_asset(pinned, params)
    require(original_asset["revision"] == migration["revision"],
            "The recovered foundation revision differs")
    return pinned, params


def parameters(custody, reference):
    import jax
    from flax import nnx
    from go2arx5_vla.sla_adapter import scale20k_model, scale20k_package
    from go2arx5_vla.sla_adapter.model_v1 import Tokenizer
    from go2arx5_vla.sla_adapter.scale20k_contract import canonical_hash
    manifest = read(pin(reference/"CHECKPOINT.json", "d987be63fdf39ded1825fa37703511fd9449589c9b20d84814ac85d5b2854584"))
    archive = custody/"HOPPER-ADAPT20K-V1/MAIN/checkpoints/phase03800"
    sealed = read(pin(archive/"COMPLETE.json", MAIN))
    require(sealed["progress"]["phase_step"] == 3800 and len(sealed["arrays"]) == 140,
            "Recover exactly the full original MAIN parent")
    by_key = {row["key"]:row for row in sealed["arrays"]}
    require(len(by_key) == 140, "Duplicate array keys in the parent")
    for row in sealed["arrays"]:
        a = np.load(pin(archive/row["file"],row["sha256"]),mmap_mode="r",allow_pickle=False)
        require(list(a.shape)==row["shape"] and str(a.dtype)==row["dtype"],"Parent tensor metadata differs")
    package = reference.parent/"HOPPER_PARENT_VIEW"
    package.mkdir(mode=0o700)
    (package/"parameters").mkdir()
    for row in manifest["arrays"]:
        original = by_key[row["key"]]
        require(all(row[k]==original[k] for k in ("sha256","dtype","shape")) and
                Path(row["file"]).parts==("parameters",Path(row["file"]).name),"Core parent differs from Hopper")
        (package/row["file"]).hardlink_to(archive/original["file"])
    parent_manifest, parent_params = foundation_paths(custody)
    token = pin(reference/"TOKENIZER.model",manifest["tokenizer_sha256"])
    state = read(pin(reference/"STATE.json",manifest["state_sha256"]))["state"]
    actions = read(pin(reference/"ACTION.json",manifest["action_sha256"]))["actions"]
    graphs, weights, foundation, identity = scale20k_model.load(parent_manifest, parent_params)
    require(identity["frozen_fingerprint"]==manifest["frozen_model"]
            and canonical_hash(identity["inventory"])==manifest["partition_sha256"],"Frozen model differs")
    restored = scale20k_package.read_weights(package,manifest["arrays"],{k:weights[k] for k in ("shared","target")})
    scope={tuple(row["path"].split("/")) for row in identity["inventory"]
           if row["group"] in ("action_expert","domain_template")}
    model=nnx.merge(graphs["target"],restored["shared"],restored["target"],foundation)
    graph,active,frozen=nnx.split(model,nnx.All(nnx.Param,lambda path,_:tuple(path) in scope),...)
    previous=read(reference/"V3_RUN_CONTRACT.json")
    require(sorted("/".join(p) for p in scope)==previous["scope"] and len(scope)==19
            and sum(a.size for a in jax.tree.leaves(active))==previous["trainable_parameters"],"V3 scope differs")
    return graph,active,frozen,dict(state=state,actions=actions),lambda:Tokenizer(str(token)),previous


def packets(pool, prepare, steps):
    pending=deque()
    steps=iter(steps)
    def queue():
        step=next(steps,None)
        if step is not None:
            pending.append((step,[pool.submit(prepare,step,i) for i in range(4)]))
    queue()
    queue()
    while pending:
        step,futures=pending.popleft()
        rows=[f.result() for f in futures]
        queue()
        yield step,rows


def host_arrays(rows):
    import jax

    batch = jax.tree.map(lambda *arrays: np.concatenate(arrays), *(r[0] for r in rows))
    support = np.concatenate([r[1] for r in rows])
    return batch, support


def reader_profiles(prepare, first, destination):
    import jax
    from scripts.pi0292.durable import write

    baseline = None
    profiles = []
    for workers in (1, 4, 8):
        digests, timings = [], []
        before = time.monotonic()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for step, rows in packets(pool, prepare, range(16)):
                batch = host_arrays(rows)
                if step == 0:
                    require(all(np.array_equal(a, b) for a, b in
                        zip(jax.tree.leaves(first), jax.tree.leaves(batch), strict=True)),
                        "Ordered CPU workers changed the original first logical batch")
                combined = hashlib.sha256()
                for leaf in jax.tree.leaves(batch):
                    combined.update(memoryview(np.ascontiguousarray(leaf)).cast("B"))
                digests.append(combined.hexdigest())
                current = time.monotonic()
                timings.append(current - before)
                before = current
        if baseline is None:
            baseline = digests
        require(baseline == digests, "Logical batches differ across CPU worker counts")
        profiles.append(dict(workers=workers, batches=len(timings), cold_first_s=timings[0],
                             total_s=sum(timings), p50_s=float(np.median(timings[1:])),
                             p95_s=float(np.percentile(timings[1:], 95)),
                             cumulative_process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        write(destination / "HOST_BATCH_PROOF.json", dict(
            profiles=profiles, first_batch_matches_serial_reader=True,
            ordered_batch_digests=baseline, optimizer_updates=0))


def allocation():
    from scripts.sla.hopper_site import site_epoch, wall_seconds, DEADLINE
    job=os.environ["PBS_JOBID"]
    pbs="/cm/shared/apps/pbspro/current/bin/qstat"
    row=json.loads(subprocess.check_output([pbs,"-f","-F","json",job],text=True,timeout=25))["Jobs"][job]
    require(row["job_state"]=="R" and row["project"]=="CFP04-CF-007"
            and row["Job_Owner"].split("@")[0]==os.environ["USER"]
            and int(row["Resource_List"]["ngpus"])==1,"Expected the project-owned one-GPU PBS resource")
    host=os.uname().nodename.split(".")[0]
    require(row["exec_host"].split("/")[0].split(".")[0]==host,"Wrong compute host")
    visible=os.environ["CUDA_VISIBLE_DEVICES"].strip()
    require(visible and "," not in visible,"PBS must assign exactly one visible accelerator")
    model=subprocess.check_output(["nvidia-smi","-i",visible,"--query-gpu=name,memory.total",
            "--format=csv,noheader"],text=True,timeout=10).strip()
    require("H200" in model,"Wrong actual accelerator")
    deadline=min(DEADLINE,site_epoch(row["stime"])+wall_seconds(row["Resource_List"]["walltime"]))
    return dict(job=job,gpu=model,allocated_cpus=len(os.sched_getaffinity(0)),deadline=deadline)


def train(capsule,custody):
    from scripts.pi0292.durable import OwnerLock, write

    with closing(OwnerLock(capsule/"OWNER.lock")):
        result=capsule/"run"
        result.mkdir(mode=0o700)
        site=allocation()
        write(result/"START.json",site)
        learn(capsule,custody,result,site)


def learn(capsule,custody,result,site):
    import jax
    from go2arx5_vla.sla_adapter import batch_v2, scale20k_training as original

    require(jax.default_backend()=="gpu" and len(jax.local_devices())==1
            and "H200" in jax.local_devices()[0].device_kind,"Wrong JAX execution device")
    jax.config.update("jax_compilation_cache_dir",str(capsule/"jit-cache"))
    reference=capsule/"incoming/reference"
    graph,active,frozen,stats,tokenizer_factory,previous=parameters(custody,reference)
    archive=custody/"foundation-layout-v2/train-ready/TRAIN-069"
    commands=capsule/"incoming/V3-MAIN-CONSUMED-DATA-02"
    original_reader=data_scope(archive,commands,stats)
    bank,counts=sample_bank(original_reader)
    require(hashlib.sha256(bank.tobytes()).hexdigest()==previous["exact_sampler_bank_sha256"]
            and sha(commands/"MANIFEST.json")==previous["data_manifest_sha256"],"Inherited sample bank differs")
    numerical=read(reference/"V3_NUMERIC_ACTUAL.json")
    require(numerical["scope_sha256"]==previous["scope_sha256"] and numerical["parent"]==MAIN
            and numerical["data"]==previous["data_manifest_sha256"] and numerical["status"]=="MASK_BACKWARD_PASS",
            "Missing inherited real-model gradient qualification")
    local=threading.local()

    def prepare(step, chunk):
        if not hasattr(local,"reader"):
            local.reader=data_scope(archive,commands,stats)
            local.tokenizer=tokenizer_factory()
        start=chunk*8
        rows=bank[step,start:start+8]
        block=original.prepare_host_batch(local.reader,local.tokenizer,"target",range(step*BATCH+start,step*BATCH+start+8),rows)
        return block,np.stack([local.reader.support(int(j)) for j in rows])

    direct = original.prepare_host_batch(original_reader, tokenizer_factory(), "target", range(BATCH), bank[0])
    support = np.stack([original_reader.support(int(j)) for j in bank[0]])
    del original_reader
    reader_profiles(prepare, (direct, support), result)
    del direct, support
    with ThreadPoolExecutor(max_workers=4) as pool:
        blocks = [f.result() for f in [pool.submit(prepare, 0, c) for c in range(4)]]
        run_updates(graph,active,frozen,bank,previous,counts,pool,prepare,blocks,result,site,batch_v2,original)


def run_updates(graph,active,frozen,bank,contract,counts,pool,prepare,blocks,dest,site,batch_v2,original):
    import jax
    import jax.numpy as jnp
    import optax
    import orbax.checkpoint as ocp
    from scripts.pi0292.durable import write

    draws=original.frozen_draws("target",BATCH)
    key=jax.random.PRNGKey(SEED)
    backward=gradient(graph)
    lr=contract["opt"]["initial_lr"]
    schedule=optax.cosine_decay_schedule(lr,UPDATES-1,alpha=1e-6/lr)
    opt=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(schedule,b1=.9,b2=.95,eps=1e-8,weight_decay=1e-5))
    update=jax.jit(lambda p,g,s:opt.update(g,s,p))
    backward_and_add = original.compile_fm_accumulation(backward)

    def measure(step, source, params, micro):
        assembled, support = host_arrays(source)
        obs,actions=batch_v2.device_batch(assembled)
        support=jax.device_put(support)
        noise,tau=draws(key,step*BATCH)
        total=jax.tree.map(jnp.zeros_like,params)
        losses=[]
        for start in range(0,BATCH,micro):
            selection=slice(start,start+micro)
            ob=jax.tree.map(lambda a:a[selection],obs)
            loss,total=backward_and_add(total,micro/BATCH,params,frozen,ob,actions[selection],
                                        support[selection],noise[selection],tau[selection])
            losses.append(loss)
        return jnp.mean(jnp.stack(losses)),total

    before=time.monotonic()
    ref_loss,ref_grad=measure(0,blocks,active,1)
    ref_loss=float(ref_loss)
    norm=float(optax.global_norm(ref_grad))
    require(np.isfinite(ref_loss) and np.isfinite(norm) and norm>0,"Invalid H200 reference gradient")
    probes=[dict(microbatch=1,loss=ref_loss,gradient_norm=norm,cold_seconds=time.monotonic()-before,qualified=True)]
    for micro in (4,8,16):
        begin=time.monotonic()
        try:
            loss,g=measure(0,blocks,active,micro)
            loss=float(loss)
            relative=float(optax.global_norm(jax.tree.map(lambda a,b:a-b,g,ref_grad)))/norm
        except jax.errors.JaxRuntimeError as error:
            if "RESOURCE_EXHAUSTED" not in str(error) and "out of memory" not in str(error).lower():
                raise
            probes.append(dict(microbatch=micro, qualified=False, status="GPU_MEMORY_EXHAUSTED"))
            write(dest/"NUMERICAL_PROFILES.json",dict(profiles=probes,optimizer_updates=0))
            continue
        loss_relative=abs(loss-ref_loss)/max(abs(ref_loss),1e-12)
        probes.append(dict(microbatch=micro,cold_seconds=time.monotonic()-begin,loss=loss,
            relative_loss=loss_relative,relative_gradient_l2=relative,
            component="target_flow_matching",qualified=loss_relative<=1e-4 and relative<=1e-3))
        write(dest/"NUMERICAL_PROFILES.json",dict(profiles=probes,optimizer_updates=0))
        del g
    del ref_grad
    # Select a different microbatch only after its repeated timing and recovery
    # have passed. The exact inherited single-window reference remains available.
    write(dest/"RUN_CONTRACT.json",dict(contract,actual_gpu=site["gpu"],cpu_workers=4,prefetch_batches=2,
          training_microbatch=1,profile="REF_WITH_ORDERED_PREFETCH",stage_visits=counts))
    state=opt.init(active)
    completed=0
    stopped=False
    status="INTERRUPTED"
    last_save=time.monotonic()
    def stop(*_):
        nonlocal stopped
        stopped=True
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,stop)

    def save():
        path=dest/f"STEP{completed:04d}"
        if not path.exists():
            with ocp.PyTreeCheckpointer() as ckpt:
                ckpt.save(path.resolve(),dict(active=active.to_pure_dict(),optimizer=state,rng=key,step=completed))
        return path,endpoint_files(path)

    try:
        for step,source in packets(pool,prepare,range(UPDATES)):
            if stopped or (dest/"STOP.json").exists() or time.time()>=site["deadline"]-900:
                break
            begin=time.monotonic()
            loss,g=measure(step,source,active,1)
            magnitude=float(optax.global_norm(g))
            loss=float(loss)
            require(np.isfinite(loss) and np.isfinite(magnitude) and magnitude>0,"Invalid actual MAIN update")
            delta,state=update(active,g,state)
            active=optax.apply_updates(active,delta)
            jax.block_until_ready((active,state))
            completed=step+1
            with (dest/"UPDATES.jsonl").open("a",buffering=1) as ledger:
                ledger.write(json.dumps(dict(update=completed,loss=loss,lr=float(schedule(step)),gradient_norm=magnitude,
                    visit_sha256=hashlib.sha256(bank[step].tobytes()).hexdigest(),wall_s=time.monotonic()-begin))+"\n")
            if completed%100==0 or time.monotonic()-last_save>=1800:
                path,files=save()
                write(dest/"RECOVERABLE.json",dict(endpoint=str(path),completed_updates=completed,files=files))
                last_save=time.monotonic()
        status="FORMAL_250" if completed==UPDATES else "FINITE_WALL_OR_STOP"
    finally:
        path,files=save()
        write(dest/"TRAIN_TERMINAL.json",dict(status=status,completed_updates=completed,endpoint=str(path),
            formal_endpoint=completed==UPDATES and status=="FORMAL_250",files=files))


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("capsule",type=Path)
    cli.add_argument("custody",type=Path)
    args=cli.parse_args()
    os.umask(0o077)
    train(args.capsule,args.custody)
