# PI0267-R1 actual data and recovery-canary return

Authority: `a0f10306419d77242ba31c84d34ce7ce3c59f419`, Issue11 `5944285937`.
This is an intermediate execution return, not endpoint quality or study completion.

- Original PI0267 PRIMARY_PARTIAL/V2/64 arrays remain unchanged. Exact V2 reproduction precedes the separate V3 rescore; all NEW E0/E1 groups are16/16. This difference is scorer-contract readmission, not learned improvement.
- All24 fixed TRAIN families admitted:10752 proposals,10248 distinct rendered RGB pairs. Real codec round-trip retains the exact OLD statistics `4439e24348bc43f5c93d380c9ae3c25b4057dbd8111e1e9bf9563b868c53bc73`; no numerical adaptation.
- Independent CPU label replay covers10752 actual proposals. Actual input boundary checks cover2112 distinct inputs. A subsequent CPU-only replay covers the complete16384 N/R training microbatch plan; no EVAL payload or model query is consumed by these checks.
- PUBLIC dataset fixed revision `bbb704fe47ebf92d42d435f2402502665767f8a3`, prefix `evidence/PI0267/R1/virtual-train-corpus-20261002`, tree `6f8cbf6260c8e77152e00cdc43168fd7de6171ab66e0dcdc2087c23fe0152f62`. Full anonymous file readback and all214 internal members pass. See DATA_PUBLIC_RECEIPT.json.

## Real recovery and the engineering deviation

Original canary job642625 retained its source/fileset/logs/checkpoints. N's original continuous8 versus4+restore4 had exactly equal model/optimizer/LR/actual random packets, but its final global CPU RNG differed. It was not declared a strict canary PASS.

The CPU-only causal replay establishes: continuous8 CPU RNG equals split4's saved CPU RNG; exactly one actual torch int64 base-seed draw maps it to the failed split8 CPU RNG. HuggingFace restores RNG before creating the replacement DataLoader iterator. With this fixed zero-worker, deterministic-input corpus, that replacement iterator consumes an extra CPU base-seed draw.

The narrow lifecycle repair invokes the existing complete `_load_rng_state` after iterator creation and before the first resumed training microbatch. It does not change condition, labels, random packets, optimizer, scheduler, loss or gradients. The original failed split remains intact. The original complete N4 checkpoint is reused in a new `split_recovery1` execution path.

Recovery job642632 closes N and R strict real canaries: model and optimizer maximum absolute differences0, LR difference0, native/S4 random packets equal, mode restored, and Python/NumPy/CPU/CUDA saved RNG states all equal. No canary weights become official initializers.

Actual discarded canary updates36 = planned32 + technical recovery4. The extra4 are charged to the original256 technical allowance, leaving252. Scientific training remains N4096/R4096 from the same fixed L2048 donor with fresh optimizers.

## Official continuation

Unique official job642638.hopper-m-02 was submitted to the lawful auto/H200 route and subsequently entered RUNNING on hopper-25. N then R execute from the same sealed input/fileset and paired visitation/noise plan. No healthy training code is edited or duplicate job submitted.

Complete2048/4096 Trainer states for both arms, fixed post-freeze evaluation, independent CPU replay and bounded secondary WBC diagnostics remain required. No quality result has yet been claimed. EVAL input/label/score payloads remain closed until both endpoints and four complete Trainer inventories freeze. Formal selected/deployment remain NONE; mail/monitor/merge remain disabled.
