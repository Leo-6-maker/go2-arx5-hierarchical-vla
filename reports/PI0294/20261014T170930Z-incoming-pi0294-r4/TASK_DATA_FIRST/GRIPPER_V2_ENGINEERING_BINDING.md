# Bounded task-gripper reset/working-target correction

This uses the adopted task-integration engineering scope, remaining original
resources and the same 14:10 window. No new scientific recipe, seed, scene,
WBC gain, mass, force, damping or safety threshold is introduced.

V1's 33-step **valid safety FAIL remains permanently sealed**. Do not rerun the
failed W0/T1 pilot. This version is distinct: mechanical travel remains
[0,0.044]m, while commanded working targets/reset reserve one 20Hz command
interval at the declared 0.04m/s: exactly 0.002m at either endpoint. These
bounds are derived before any new physical trial, not fitted to task scores.

The nominal open reset becomes 0.042m. Teacher logical open/closed are converted
once into actual mechanical fractions q/0.044 for targets 0.042/0.002m. The
committed EEF13 labels record those fractions. The actuator still applies
fraction × 0.044 exactly once; no actual q/qvel or model output is clipped.
Input measurements remain actual mechanical fractions. Values outside the
working target envelope are rejected, not silently corrected.

CPU compilation must again prove all original robot properties unchanged,
legal reset/contact and actual 2mm finger margin, and verify the compiled task
library/source. V2 needs an immutable CPU acceptance receipt. That receipt
does not claim physics or task PASS. V1 sentinel bytes stay fixed and default
V1 entry remains blocked; a V2 safety failure separately stops V2.

Only the remaining predeclared W0/T2, W1/T1 and W1/T2 pilot cards may run. V1's
physical failure remains in the original four-pilot denominator; there is no
replacement W0/T1, bonus pilot or new allocation. If V2 reaches a valid native
safety failure, stop it rather than tuning/searching another margin or damping.
Report original task success, data admissibility and missing skills separately.
