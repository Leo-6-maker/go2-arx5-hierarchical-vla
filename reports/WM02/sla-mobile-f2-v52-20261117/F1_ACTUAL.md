# Complete F1 and first matched F2 layout

All12 F1 requests are terminal on source `2797d0c2c458667e241aa0d24d6d0b005f3caf1a`. This is the four-layout F1 subset of the fixed24-condition experiment;the F2 subset and complete model decision remain unfinished.

| Layout | B0 | B1 | B2 | B2−B1 | B1−B0 |
|---|---|---|---|---:|---:|
| T0 | ILLEGAL_CONTACT | SUCCESS | SUCCESS | +0 | +1 |
| T1 | ORIGINAL_FULL_H40_POLICY_REJECTED | SUCCESS | SUCCESS | +0 | +1 |
| T2 | SUCCESS | SUCCESS | ORIGINAL_FULL_H40_POLICY_REJECTED | -1 | +0 |
| T3 | SUCCESS | ORIGINAL_FULL_H40_POLICY_REJECTED | ILLEGAL_CONTACT | +0 | -1 |

| Arm | Composite successes | Contact STOP | H40 rejection | Technical failures | Native ticks | Verified generations |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 2/4 | 1 | 1 | 0 | 72650 | 201 |
| B1 | 3/4 | 0 | 1 | 0 | 75312 | 211 |
| B2 | 2/4 | 1 | 1 | 0 | 86085 | 255 |

B1−B0 is+1 and B2−B1 is−1 on these four paired development layouts. This is a small descriptive subset,not a significance claim or a complete eight-condition model comparison. Same scene/reset/noise does not establish identical complete WBC histories;the actual post-init packet differences remain in the native prefix report.

There are already fixed promotion blockers: B0 and B2 each have only2/4 F1 successes and a hardSTOP;B1 still fails two of the original three saved rejection inputs in the fixed offline qualification. None of these failures is replaced or retested. Complete F2 anyway to obtain the full paired comparison and decide the conditional E3 evidence branch;no confirmation,DEV or TEST is opened by this subset.

B1/F2-T0 has now genuinely grasped,lifted25.39mm,held1.25s and moved90.02mm in the required direction,withoutSTOP or technical failure. It used66 verified generations/66 fullP10s and22,203 native ticks(5,703 initialization+16,500 task). This is a new F2 result;the original PR403F2 0/4 remains unchanged. A new success does not waive B1’s failed offline gate.

| Measured F1 cost | B0 | B1 | B2 |
|---|---:|---:|---:|
| Sum of episode wall seconds | 1178.15 | 1206.29 | 1411.61 |
| Sum of reported case-process CPU seconds | 588.14 | 608.98 | 728.11 |

Case-process CPU does not include all child/worker CPU. GPU busy-seconds were not separately measured. AllF1 ticks reconcile with confirmed atoms,partial ticks and12×5703 initialization ticks. Both contactSTOP traces record0 following integrations. This report generated no new model output or dynamics.

The first matched F2/T0 triplet is now complete:

| Arm | Composite result | Lift(mm) | Hold(s) | Required motion(mm) | Native ticks |
|---|---|---:|---:|---:|---:|
| B0 | SUCCESS | 25.12 | 1.20 | 90.14 | 22703 |
| B1 | SUCCESS | 25.39 | 1.25 | 90.02 | 22203 |
| B2 | SUCCESS | 24.53 | 1.05 | 95.85 | 23703 |

Frozen B0 also succeeds. Both paired success differences on this layout are0;the success cannot be attributed to additional500-step training or endpoint loss. The original PR403F2 0/4 is preserved as a different historical condition set. Complete the other three F2 layouts before the full comparison.
