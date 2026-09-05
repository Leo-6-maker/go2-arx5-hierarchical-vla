# Stage36-RQ — delivered-WBC authority and long-horizon oracle qualification

The canonical gate is `PI_GATE_STATE_0021` on Issue #11, with PR #22 at the
accepted base `7c1e864ee975e4e94196792ed8408f326fb4dfb9`. The branch is a
mechanical, provenance-preserving implementation of the frozen Stage36
ladder; it does not alter the scientific contract.

## A–D CPU admission

The CPU harness first binds the redelivered archive
`WBCTrackingShare_20260831.zip` by its exact size (`963451251` bytes), archive
SHA256 (`b52a53be...6a018167`), 887 members, 742 files, 145 directories, and
the existing extracted tree SHA256
`63311df752c9aaa5289287849635309e558f79284c91b80ddf07ec5638992567`.
Every archive member is compared byte-for-byte with the existing extraction;
the extraction and all historical Stage22–Stage35 trees remain in place.

The selected delivered run and its arm/dog checkpoints, native config, URDF,
reach table, trajectory module, WBC environment, legged-robot module, wrapper,
and requirements file are all selected by explicit path and expected SHA256.
Duplicate basenames are recorded as evidence but never used to choose a
runtime input. The sealed Stage35 corpus is read in place, with no generation,
editing, or promotion of its analytic support witness. Its exact corpus tree,
manifest, protocol, and seal are bound to:

```text
tree      b98960f9dc4bfd04b5a0de0f693ddea4f50c8849ef80da83428280c1e2e42414
manifest c1e4021a77817af851ec8c1ba5d0c4b5f116558c16f03ff0e24457e1fd6e0642
protocol 81585c50b29ba67d65e346d3b6fb10420a52ca283d78e54d7476073ecacca22d
seal     5ec9dc293546ea33c39c225d184921e77f1ae6f86756ec1bd0304cb19aa63e89
```

Runtime admission imports IsaacGym before Torch and checks the reconstructed
Python 3.8.20 / PyTorch 2.3.1+cu121 / CUDA 12.1 / torchvision 0.18.1+cu121 /
torchaudio 2.3.1+cu121 / Gym 0.23.1 surface. It imports the delivered
trajectory, reachability, config, WBC, and wrapper modules without creating an
actor or simulator. CORE24 calls the delivered `to_canonical` path on CPU.
CONTINUOUS8 queries the exact reach table and checks finite moving-base
witnesses, reset identity, and fixed-base impossibility. No WBC rollout is
claimed by either admission.

Geometry qualification evaluates every frozen `K={4,6,9,12}` candidate over
the CORE24 set using the committed 101-anchor preview comparison and requires
both every case and the aggregate to pass all preregistered limits. A current
strict run yields `K_STAR=null`; therefore `T_STAR=null` with
`NOT_EVALUATED_K_STAR_NULL`. No threshold is loosened and no candidate is
promoted from an aggregate-only result.

## E campaign boundary

If A–D and the queue/runtime preflight pass, the generated plan admits one
`auto` PBS allocation with at most 96 rows. With `K_STAR=null`, it contains
only the dense seed-13 `BENCHMARK_CONTINUOUS` cases: CORE24 plus CONTINUOUS8
when C8 admission passes. The existing native launcher is invoked with the
exact arm/dog checkpoints and exact trajectory NPZ inputs. There is no qsub or
retry path in the campaign runner.

The allocation uses one H200 under `CFP03-CF-004`, scratch-backed HOME/TMP/cache
paths, and a new output root `stage36_eval_${PBS_JOBID}`. It never writes into
historical job 600695. It does not run GR00T, training, backward, optimizer or
scheduler steps, resume, checkpoint creation, sampler/get_action, C2,
validation-E, holdout-F, WBC hardware, or a second allocation. Failure
preserves partial logs/evidence and stops for GPT/PI.

The final return seals result JSON, raw row hashes, result manifest, runner,
wrapper, and logs, and stops at
`GPT_PI_REVIEW_AFTER_STAGE36_LONG_HORIZON_LADDER`.
