# PI0288 isolated integration

Run from this repository's worktree. The established SSH aliases route through Corelab login to `corelab-4080s`. Public integration files stage under `/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0288/project-src`; machine-local config is outside Git. Reuse the existing native SSD candidate and dependency payload, without rebuilding the twenty ROS packages.

```powershell
py -3.12 scripts/pi0288_mujoco/owner.py stage
py -3.12 scripts/pi0288_mujoco/closeout.py
py -3.12 -m pytest tests/target_mujoco -q --basetemp .\Projects\go2-arx5\PI0288\cpu-tests
```

`stage` compiles only the original observer and zero-step model audit against installed MuJoCo headers/libraries. It does not start physics or copy private source/models. Local evidence closeout recomputes saved originals; it does not perform new experiments.

Actual native entry invocation is `owner.py interface-tests`, then `owner.py native-reference --condition <registered name>` only after an actual G0 receipt. **Current default-model preflight rejects startup before physics.** Do not use invocation as a quality retry or change initial pose to bypass this rejection.

Owner takes one lease, checks domain228 is empty, verifies core pin, reserves budget before stack start, uses localhost-only ROS and fresh whole-stack reset per episode. Stop targets only process groups it launched. Raw records are under `PI0288/evidence/<uuid>`; original failure records remain. Runtime temporary/ROS logs are confined to the project runtime directory.

Direct private runtime entry, after sourcing the existing candidate environment and the dependency paths in `owner.py`:

```sh
python3 project-src/scripts/pi0288_mujoco/run.py \
  --root /var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0288 \
  --config /var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0288/private/config.json \
  --mode interface-tests
```

`task-reference`, `teacher-dev`, `teacher-pilot`, `collect` are declared but deliberately not connected to native task execution in this bounded closeout. Their CPU scene/teacher/data primitives are implemented and tested. They are not usable data collection paths and no completed task runtime is claimed. Native startup/G0/G1 and verified gripper specification must be resolved before connecting and executing them.

`validate_dataset.py` and `prepare_finetune.py` require genuine admitted data and receipts; absence is a failure, not readiness. Official processor batches have not run. No GR00T neural forward/training is included.
