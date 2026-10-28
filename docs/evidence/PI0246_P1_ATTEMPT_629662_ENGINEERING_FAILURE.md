# PI0246 P1 attempt 629662 — engineering failure

- PBS job: `629662.hopper-m-02`, terminal `F`, `Exit_status=1`, assigned `hopper-25` with one GPU allocation.
- It exited at wrapper line 26 before the Python runner, GPU query, model load, or stock inference. `queries=0/32`; training, backward, optimizer updates, WBC/sim, and DEV/F reads were all zero.
- `.ER` reports `mkdir: cannot create directory '/Project_Storage': Permission denied`; `.OU` records `PI0245_P1_FAILURE line=26 status=1`.
- qstat history has no `Resource_List.group_list`. On the login node, `/Project_Storage/CFP-03/CFP03-CF-004` is mode `2770`, group `HPC_Users_3104`; the PBS wrapper had not requested that supplementary execution group. The assigned compute node's mount/group state was not logged, so this identifies the missing group binding as the leading mechanical cause while leaving mount availability to the new fail-closed preflight.
- Original failure artifacts and submitted wrapper are preserved under `/Project_Storage/CFP-03/CFP03-CF-004/pi0245_physical_temporal_pilot_20260924/engineering_attempts/629662/` and hash-match their scratch sources:
  - `.ER`: `056610de8885bdd933a4172fced426bb7451934566f6e780022dccaadfddf6ae`
  - `.OU`: `7e84d79581725ad99373303a313ceaa29068316331c0c6e04c69ba398879b429`
  - submitted wrapper: `be1f0bae8252673b1d668f8e6774db585b24efd722e0d5606308aae1ad0e0f7c`
- Mechanical repair: request `HPC_Users_3104`; log effective groups, mount, and path traversal; test the project root exists and is writable before creating the archive or querying the GPU.
- Validation: exact Stage10 Python runtime wrapper/binding tests `6/6 PASS`; `bash -n` PASS; P2 schedule/P1-admission CPU tests `3/3 PASS`; Python compile and `git diff --check` PASS.
- Repair commit: `c1056fab38bfda56d4d9371b9006b89a401fd092`; no scientific configuration or result was changed.
