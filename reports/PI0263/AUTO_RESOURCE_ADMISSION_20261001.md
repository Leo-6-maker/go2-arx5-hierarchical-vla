# PI0263 auto routing and project admission check — 2026-10-01

Authority and scientific plan remain PI0263 at 1c574b6e6e9d6a7851e64ca57e89c468a4c9eb28. This is a resource admission clarification, not a scientific retry or budget reset.

## Sources inspected

- Complete local NUS IT Hopper onboarding guide (16 January 2025), 47 pages; budget pages 18–20, automatic routing page 25, batch example page 31 visually checked. Both project allocation and auto routing appear together in that example. The restricted source PDF and screenshots remain local and are not published.
- Current Hopper native `hpc ai`, `hpc pbs help`, `hpc how many GPUs can I use`, `hpc project` guidance.
- Live PBS auto queue configuration: Route; destinations small, medium, large, special; default queue charge rate 1. Both existing jobs carry PBS_O_QUEUE=auto.
- Public NUS IT instructions: https://nusit.nus.edu.sg/services/getting-started/how-to-run-batch-job/ . Cloud routing explanation is supporting context, not a substitute for Hopper live policy: https://nusit.nus.edu.sg/technus/understand-pbs-job-submission-in-hpc-cloud/ .

## Result

`auto` chooses an execution queue from requested resources; it does not remove project accounting or expiry checks. Native Hopper guidance reserves autox express routing for self-funded project users, so autox is not a demonstrated fallback for this allocation.

F main 640396 remains Q in small after submission through auto, project CFP03-CF-004, one H200, 12 CPUs, 225 GB, six-hour request. The current scheduler comment reports `Budgets-am_hook: Project has already ended`. This is an observed hook result; the exact project balance/end date remains unverified because `hpc project` requires a renewed `amgr login` with the account owner's password.

X canary 640412 remains system-held on afterok:640396, also submitted through auto. Neither main arm has started. No job deleted, resubmitted, manually held, rerouted, or changed; staged model code and identity seals remain intact. No credentials recorded.

Owner must authenticate privately in their Hopper terminal (`amgr login`, then `hpc project`) to expose active authorized project(s) or confirm renewal. Once eligibility is verified, continue the existing jobs using lawful project assignment; no routine scientific GO is needed.

## Scientific accounting

F recovery canary remains PASS: 16 discarded completed updates and 304 raw training head calls. Main updates 0/4000; X canary 0. Donor P2000 1152 rows reused and independently replayed on CPU; no new endpoint inference here. All evaluation/retirement/deployment boundaries stay as originally frozen. selected_checkpoint=NONE; deployment_checkpoint=NONE. Goal incomplete at resource admission, not a scientific FAIL.
