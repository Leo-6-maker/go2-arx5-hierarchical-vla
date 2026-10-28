# C1 V2/V3 data transfer and CPU admission

**Actual preparation PASS.** Twenty-seven native-contract-admitted correction suffixes and the fixed V3 plan are on Hopper. This handoff launches no optimizer or model. The newly authorized 14:00 extension is controlled by the owner; old time-stop and failed preparation receipts remain preserved.

- Bank SHA256: `f2c7b43af6edc42795ee0595ed0a9c9f05f440369945d5b43bfc38419e21cb8d`.
- Plan SHA256: `0194583c35e93000771aed598817091a6d670b9f1720d31da6a06d6b37934f0e`; 8192 rows, 4096 expert +4096 correction.
- Correction bucket draws: [1024,1024,1024,819,205]; exact V3 equal weighting among eligible branches within each bucket. All rows match their sealed original UID/data identity.
- New data transferred: 162 NPY files across 27 suffixes, 1,310,708,736 bytes. Tree SHA256 `254aeb875b6961dcbc8c43c3acaa9885c03b15b2111b3179a44185a901353ffa`.
- Copy scope: only those new correction arrays plus V2 bank, V3 plan/audit and versioned C1 data-admission/manifest metadata. No failed prefix, expert array, normalizer or model file copied.

Core performed full member SHA/shape/dtype/time/finite validation and plan identity checks, preserving old receipts/statistics. Hopper verified streamed bytes, ran the existing `authority.stage_card` checks for parent14/Adam6144/paired RNG/original expert inputs, and then independently reread every transferred file from disk with SHA256 and NPY header validation. All passed.

Prepared active card: Hopper `design/C1_CORRECT1024_CARD.json`, SHA256 `d44c337962a31abd99e03a4049541ee5de8999f0e9f800d2a66dbafa61076e65`. It explicitly requires **new C1 continuous8 vs fresh4+restore4 canary**; no C0 canary result is adopted. Owner capacity/time-reserve fields inherited from the C0 base must be refreshed by the owner's final C1 routing, with any revised card recorded under its new hash. This prepared card and exact original remain available as evidence.

`hop_C1_TARGET_DISK_READBACK_NATIVE_V2_BUCKET_V3.json` is the final independent disk readback. The adjacent inventories contain original receipt byte counts and hashes. Hopper healthy code and code seal were not edited. Model/physics/codec/optimizer/PBS calls by this preparation: zero.

An initial relay attempt hit Windows command-length limits before creating either stream process. The relay now stages fixed manifests under each host's C1-specific `correction/transfer_tools/` and invokes short script paths; the completed file scope and hashes are unchanged.
