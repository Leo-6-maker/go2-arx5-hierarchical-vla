# New live-screen H40 rejections — read-only attribution

Actual saved inputs and outputs as of `2026-11-17T16:36:06.968896+00:00`;no new model generation,native request or dynamics integration. Both rejected proposals were checked by the unchanged complete-H40 compiler and were not installed. The native execution source remains `2797d0c2c458667e241aa0d24d6d0b005f3caf1a`.

| Condition | Query / task time | Head / tail / interior acceleration(m/s²) | Sufficient observed violation |
|---|---:|---|---|
| E2-F1-T1-B0 | 1 / 0.50s | 0.277729 / 0.108762 / 0.122375 | head 0.277729 > original0.25,index0 |
| E2-F1-T2-B2 | 54 / 27.00s | 0.117765 / 0.019200 / 0.261575 | interior 0.261575 > original0.25,index34 |

B0/F1-T1 fails on the head after0.5s. B2/F1-T2 fails at query54 after27s on an **interior** second difference(index34,zero-based,using H40 XYZ points33–35),while its head and tail remain below0.25. The interior0.261575m/s² violation is independently sufficient to refuse that proposal. These figures do not rule out simultaneous constraints;the complete original compiler remains the authority.

The B2 training intervention only penalizes the declared first/last pairs. Passing all96 saved rolling states does not imply full legality on fresh closed-loop states. The new interior failure does not justify removing the H40 suffix or broadening the loss during the fixed comparison. No joint-space/WBC diagnosis is inferred from a proposal rejected before installation.

This is a partial comparison. Complete all24 preregistered episodes before selecting or rejecting an arm. Do not automatically open the correction-data branch while boundary failures remain the main unexplained mechanism;use the full panel and concrete states to decide. Action/input SHA256,previous-command50ms clock and exact rejection strings are preserved in [the numeric evidence](E2_REJECTION_PREFIX_ACTUAL.json).
