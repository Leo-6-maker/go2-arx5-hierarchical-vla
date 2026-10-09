# Erratum: correction codec diagnostic was added as an admission gate

**Correction bank under the released native contract: 27/32 admitted. C1 optimizer was not started; owner stop is the 11:30 time gate.** The original 15/32 output is preserved as the result of an implementation-added condition. It is not a protocol-valid scientific failure of the correction bank.

## Error and source authority

Release `eae6dbc1f9281b204a4e28e5f7c100d705837318`, canonical 6072813602, protocol §2 and machine limits require native XYZ <= 1e-5 m, SO3 <= 1e-4 rad and gripper <= 1e-5, finite valid labels, and no forbidden clipping/label manipulation. There is no zero diagnostic preclip-count clause. The original PI0285 `codec_check.py` accumulates preclip counts as diagnostics and admits by physical roundtrip thresholds.

I added `valid = valid and err['native_preclip_out_of_range_count'] == 0` to PI0286 recovery. This used a float64 intermediate diagnostic as an extra admission gate. Prior reports that called it an unchanged frozen gate were incorrect. The historical `README.md` claim “no preclip range violation” is superseded by this erratum, as are the old reports' interpretation of 15/32 as a protocol scientific NOT_ADMITTED verdict. Original receipts, CODEC records, raw labels and reports remain intact.

## Saved numerical evidence

Twelve late suffixes had only small float64 identity-rotation upper-bound excesses: normalized 8.499867476530198e-13 / 9.605649609056854e-11 in channels 3/7. Actual native forward processing clips those intermediate values under the original unchanged `clip_outliers` setting. The real encoded FP32 labels have zero out-of-range values and are elementwise identical to directly casting the preclip intermediate to FP32. Native inverse clipping changes zero encoded values. All 12 actual saved anchor-0 roundtrips satisfy the released physical thresholds; their previously stored full-H40 maxima also satisfy them.

The real 285118@3s SO3 failure remains excluded: three saved windows exceed 1e-4 rad, with maximum 0.010654909013566369 rad. Four physical failures remain excluded: 285113@3s, 285113@12s, 285114@12s and 285118@12s. No original score, clipping operation, label or normalization statistic changed.

| Takeover | Requested | Faithful restore | Physical success | Native-contract codec-valid branch |
|---|---:|---:|---:|---:|
| 3 s | 16 | 16 | 15 | 14 |
| 12 s | 16 | 16 | 13 | 13 |
| Total | 32 | 32 | 28 | 27 |

## Minimal implementation repair and V2 derivation

The owner authorized removal of the added count==0 condition under the original release, retaining the diagnostic and all original physical/nonfinite/degenerate gates. Core was stopped, both source/owner locks were acquired, and no active source process was found. Exactly one line was removed; no other healthy seal member changed.

| Artifact | Before SHA256 | After SHA256 |
|---|---|---|
| Core recovery.py | 62c6cd64e9a0ce6980243881f4948122d8ba3170fba52e535dbc0e89a546128d | 8b97c64e00cfcb403c75681c12cdb57faa45569cb9495ccc0ea893bd2157fbca |
| Core CODE_SEAL.json | cf82000848cf8a4d315cce33aedf21d9df5f4e27acae7d7c31b865b5f39440f7 | 05487f9e035d56ed0224cd8858cd3e5972e6a6138e52bbbb95cf6b8abca79e10 |

Old code and seal are independently retained in `correction/implementation_erratum/`. Local and Core text differ only in line endings; their repaired normalized-text SHA is `c33a01bc0a2dd04c1057696d7b192200560939f0b31d0003ccfaf7358e23b110`.

The actual source validity statements pass exact physical boundaries and reject above-boundary, nonfinite and degenerate cases. All three real 285118 SO3 errors still fail; the 12 preclip-only errors satisfy the source predicates. V2 is derived solely from all sealed H40 maxima and saved failure errors, with zero new full encode, model, teacher or physics calls. Old bank/plan/manifest, statistics and window-index before/after hashes match.

`BANK_NATIVE_CONTRACT_V2.json` is a separate 27/32 bank; SHA256 `f2c7b43af6edc42795ee0595ed0a9c9f05f440369945d5b43bfc38419e21cb8d`. It preserves all original per-slot V1 statuses/failures alongside V2 status. It does not grant optimizer launch.

## One actual fixed-sampler attempt

The original implemented sampler was attempted exactly once with 27 candidates. It failed its exact-global-equal-branch flow constraint while retaining fixed correction bucket draws [1024,1024,1024,819,205] and 4096 correction/4096 expert draws. Hold-only branches are forced to at least 1517 draws by its global branch quotas, exceeding the hold bucket's 205 draws. Available branch counts by bucket are [15,15,15,17,27]. No fallback sampler, changed weight or changed bucket quota was used.

The release says equal episode/branch weighting within source and existing behavior buckets. The implementation applies equal **global branch totals**, then simultaneous global bucket totals. Equal weighting **within each behavior bucket** is a different conditional weighting definition and need not imply equal global branch totals when suffix coverage differs. The original full expert episodes have broad phase coverage; correction suffixes have incomplete coverage. This observed flow failure describes the implemented global constraint. It does not prove infeasibility of every sampler meeting the release wording. No second sampler was designed or tried, and no scientific quota was adjusted to obtain PASS.

Sampler receipt: `C1_PLAN_NATIVE_CONTRACT_V2.json`, SHA256 `787c263115aad5c2e241f82aa57c09aa6410ac184e4fac0c7d2a2eccb044db4e`. Its status is `IMPLEMENTED_GLOBAL_EQUAL_BRANCH_FIXED_BUCKET_FLOW_INFEASIBLE_NATIVE_CONTRACT_V2`. No training plan or transfer is admitted. Owner's time stop is independently the 11:30 gate; this sampler diagnostic is not substituted as the scientific stop reason.

## Local evidence handoff

- `ENCODED_LATE12_NATIVE_CLIP_EVIDENCE.json/.md`: actual dtype/clip/roundtrip evidence and exact native function sources.
- `BANK_NATIVE_CONTRACT_V2_COMPACT.json`: 32 slot statuses and hashes, 27 admitted IDs, preserved V1 link.
- `NATIVE_CONTRACT_V2_LOCAL_BOUNDARY_CHECK.json`: actual-source predicate boundaries and real failures.
- `native_contract_v2_receipts/`: byte-exact derivation, sampler, boundary and old/new seal receipts.
- `NATIVE_CONTRACT_V2_LOCAL_RECEIPTS.json`: local receipt SHA256/byte inventory.

No optimizer, model or physics was launched for the repair/derivation; raw slot arrays were not reopened. This is an implementation correction and evidence erratum. Learning causality remains UNPROVEN and no model is newly qualified.
