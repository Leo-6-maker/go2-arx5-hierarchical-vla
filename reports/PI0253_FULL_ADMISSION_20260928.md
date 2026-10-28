# PI0253 FULL admission and actual submission

`PI0253_FULL_ADMISSION_V1: PASS_FULL_ADMISSION` under release `82554098fb55b39ceca4b955252f8952c91c0dc0`, canonical authorization `5870278574`. D0/D1 execution canonical: [5871378817](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-5871378817).

## Current scientific state

- Original G2000 M64 all64/old16 × B0/B1 four-group PASS retained; TRAIN-A/C0 first-valid physical FAIL retained. No old physical episode retry.
- Fixed D1 has 8 logical cells: 2 exact historical outputs reused, 6 new queries, 24 new native Euler evaluations. Q/S raw and decoded XYZ agree exactly, SO3 max 2.98e-8 rad, gate/issues/metrics identical. No worker calculation change was needed.
- Legal A_attempt2 expert reconstructed through original native trajectory sampling; desired positions match every stored expert tick exactly. Correct H40 reference reaches within 8.56 micrometres of the target. Genuine G same-tick mean tracking error remains 137.424 mm; it is not a difference of two minima.
- Original TRAIN seal verified over 29,389 files / 297,448,331 bytes, tree `66559a759c187fdf4d55c175a0808417369c033323e35710b6ca8aeda3786e81`. All 9792 desired-label windows and timestamps verified; A has 153 independent rollouts, 153 reset windows and 612 early windows. B–D also have valid reset/early support. No replacement data or CALIB gradient/statistics use.
- D0 boundaries remain explicit: no archived zero-tick expert root/dof/history bytes and no per-rollout world anchor in materialized TRAIN rows. No byte-identical initial-state or sufficient-coverage causal claim. Historical PI0237 CALIB-report peek is disclosed; no current CALIB dataset/selector/candidate read or query.

## FULL entry and persistence

Actual CPU checks passed for the new execution entry: G fixed-revision initialization, real resolved 10000-step config, original complete G loss and a poisoned terminal loss rejection, PCG64 9792+208 schedule, step-addressed FULL RNG, 2500/5000/7500 suffix cursors and synthetic optimizer/scheduler/RNG restore, cosine horizon and real archive callback routing. These are CPU fixtures, not completed project training or checkpoint restores from future outputs.

The first CPU preparation found a 2000-step default in a configuration caller; the shared FULL configuration adapter now binds 10000 in every caller. The unexecuted preparation card and failure log are preserved under `PI0253/engineering_prepare/attempt-01`. Project optimizer/query accounting for that repair is zero.

G initializer: revision `efecea485ae77000599d19171b02491f8d514f42`, full Trainer tree `cfbfeb54bea2b5ac266047dfd764cfa6b041a8a77b9688737ea70b56935d5226`; actual model files rehashed and hardlinked on Hopper, with original frozen C_T processor files. Fresh optimizer/scheduler for this new stage.

Final new card SHA256: `3bece4c02d8158ec09a35fc597d6ffd34290b8d1559ea55ed877da75c6714277`.
Admission SHA256: `96228f628846c92a90104d7659bd1dfdd342200a8b6bdf9508321fd79e283bb7`.

Actual fileset minimum available was 80,606,969,856 bytes (75.07 GiB); next full save plus >=10 GiB reserve needs 31,620,064,634 bytes. `quota_headroom=UNKNOWN`: quota returned no numeric value; global df is not used as a quota proof. Persistent scratch remains the source on upload failure. HF PUBLIC authenticated empty-prefix write/readback preflight passed.

Formal complete Trainer saves: 2500/5000/7500/10000; 1000 model-only observation is not a candidate or resume source. Fixed-revision full-file readback precedes local retirement. Inherited unchanged G sampler parity is used, with zero new nonselection parity rollouts and no fabricated paired arm.

## Actual PBS state

Single accepted job: `636930.hopper-m-02`, name `pi0253_full_g`, submitted 2026-09-28 22:15:53 Asia/Singapore. At the submission readback: **Q**, routed from `auto` to `small`, no hold, project `CFP03-CF-004`, one GPU / 12 CPU / 225 GB RAM / 16 hours. The worker enforces one actual H200 before training. Mail points `n`; no monitor created.

The stage has one 10000-step scheduler and budget. At durable step2500 it pauses to evaluate the original fixed early barrier, then continues by exact full-state resume if that barrier permits. Only **both** step1000 and step2500 B0 having 64/64 `LINEAR_SPEED_REJECT` is the prescribed quality early stop. Ordinary intermediate joint FAIL continues. No new optimizer or stage-budget reset at that pause.

Next: fixed retention (640 logical / at most576 new), independent CPU recomputation, four final-family live side reports, then original CALIB unique selector and winner retention admission, followed by original TRAIN A–D one-chunk and conditional receding. Live side reports do not tune training or selector.

`selected_checkpoint=NONE`. FULL completion, formal selection and physical success remain separate pending outcomes. No Corelab training, new M64 branch, loss/seed/gate/WBC/ABI/dt change, DEV/F/DAgger/real robot/merge/monitor.
