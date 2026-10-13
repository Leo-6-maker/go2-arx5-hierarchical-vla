# PI0294 M512 Policy Card

Status: LOCAL_FORMAL_CHECKPOINT_UNQUALIFIED. This is not a working-policy claim.

One shared GR00T-N1.7-3B was trained for 512 actual optimizer updates on
frozen v3.0. Both tasks use dual RGB, language and measured EEF/gripper state;
task ID is audit/sampling metadata, not an inference policy selector.
The visual/language backbone remained frozen; 537 trainable tensors changed.

- Base revision: `2fc962b973bccdd5d8ce4f67cc63b264d6886495`.
- Official source: `51d4c89f72fda44cbf77285c6a8114b52676b8a1`.
- Runtime source: `a171997a704072e127d16410ba584c8aa0ddd573`.
- Dataset manifest: `b8ddcb9808e6dfac4f3ac5fa5a4828128fa9644a330c17d1c57bf833ce5889df`.
- Statistics: `fb5f5bc6969fae47f51b8a6acefdd7312ed948a0fad4b9f9a79e1fbb9aa325c7`.
- Local full-state inventory: `6fdc10c2cfb13841bbabd444aa0ab8a4a6aef09adfbffae7d5fb4dee24747a8a`.

Actual policy TRAIN development, DEV comparison, candidate selection,
confirmation and old-task regression are all NOT_RUN. Task success rates and
language-conditioned behavior are unknown. Loss 0.315059 does not establish
either skill. Teacher route success is not model route success.

The 25,541,023,437-byte local full state is preserved. HF LFS preupload failed
with private-account storage HTTP 403; no verified model revision or READY
exists. M512 independent resume has not run. Exact full-state engineering
resume passed separately at the original numerical tolerances.

Selected: NONE. Deployment: NONE. WBC, mobile Go2 and hardware: NOT_RUN.
The intended domain is small-scale fixed-base ARX5 MuJoCo manipulation only.
