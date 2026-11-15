# V6 fixed TRAIN8: first actual student condition

Snapshot **2026-11-14T17:54:51.847243+08:00**. This supersedes only the evaluation-progress snapshot in the [first4 and complete3K report](EXECUTION_20261114.md). The complete3K model, training recipe, 96000 target visits, fixed conditions and historical failures are unchanged.

TRAIN01/restart3 completed as a **valid policy failure:0/1 full-task successes**,0 technical failures and0 hard safety stops. There were **10 real SLA queries**,0 evaluation optimizer updates,7953 native integrations. The first nine raw H40 proposals passed the original full checks and each consumed its original P10. The tenth query at tick7953 was rejected by the original full-preview velocity/acceleration gate before installation;its consumed count is0. No Teacher fallback, expert regeneration, endpoint basis, clipping or safety relaxation.

Independent readback verifies **90/90 actual student commands**, nine distinct original decision hashes with10 commands each, all original solver/gripper-confirmed. Exact contiguous25-tick intervals run5703→7953:2250 student-execution integrations,plus5703 preparation. All nine queries remain PREGRASP;this is neither a completed grasp nor a task success. The failed command is not relabeled as executed supervision.

Native worker wall443.9214s;complete native/neural pair CPU514.8391s;GPU worker wall469.3364s and childCPU81.1850s. The read-only consumption audit adds0 models/updates/physical requests/integrations. Complete-state SHA `b01d15817391f7e74971678ad3304060e8651dd7582761b1a0ba7611f115800d`;confirmed-command SHA `1b244490cd20958ce9814ea8c554e5970d1edc9d019eab59be5c866b5b5ed232`.

The next original TRAIN01/restart4 condition is progressing in the same finite background group. All seven remaining original conditions remain in the denominator;none is replaced or retried. IndependentDEV48 has not started. Old RP8 stays failed and S16 unstarted. Continue the original panel and preserve every outcome;do not tune on this first result or claim the whole endpoint passed.

[Machine-readable first outcome and all nine decision identities](FIRST_STUDENT_ACTUAL_20261114.json).
