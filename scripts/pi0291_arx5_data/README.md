# PI0291 CPU Native Data Entrance

This package is separate from PI0290 physical acquisition and the frozen WBC.
`native_dataset.DatasetLoader` reads admitted real ARX5 RGB and absolute committed
EEF10 commands. It calls the official processor, not a hand-written tensor packer.
The official collator is invoked by the CPU qualification command and must also
be used by a separately authorized future training client.

## Actual Inputs

`export_dataset.py` accepts the run root and the independently verified
`DATASET_ADMISSION.json`, whose `episodes` list includes `id`, `split`, `group`,
`seed`, `admitted`, and optionally `samples_sha256`. Only admitted TRAIN/DEV rows
from domain `FIXED_BASE_ARX5_TASK_SIM` enter the export. Commissioning, pilot,
failed attempts, and PI0290 data do not enter its statistics population.

Raw files retain the PI0290 list-based recording structure: `samples.json` has
641 actual observations and 640 committed commands; `references.json` binds the
committed prefix8 to its actual planned reference. Raw RGB files remain in the
bound episode root. `DATASET_MANIFEST.json` records that exact private path;
moving a dataset requires a new host-aware path binding and byte verification.

The compact NPZ contains absolute committed640x10, actual anchors601x10,
original RGB paths/hashes, language, time grids, and valid-window flags. It does
not contain pre-relative actions, oracle objects, joints, teacher stages, or
success as policy inputs. There are601 genuine H40 windows per complete episode.
Actual gripper feedback is never clipped to its command: measured elastic
overshoot within the already frozen1mm joint-validity tolerance is retained.
Commands still have closed0/open1 bounds. The raw auditor independently checks
the underlying finger positions and reports the actual fraction range.

Statistics use all admitted TRAIN full H40 windows with equal window/timestep
weight; overlapping interior commands intentionally repeat. State uses each
legal anchor once. Native min/max is used, not q01/q99, and DEV never refits it.
The fixed stock inverse clips, so the qualification separately reports preclip
and inverseclip counts and actual normalized decode errors. No old monkeypatch
is installed to suppress an inconvenient result.

## CPU Commands

Use the existing processor-only CPU runtime with `CUDA_VISIBLE_DEVICES=''`,
`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`.

```sh
python export_dataset.py --root RUN_ROOT --admission DATASET_ADMISSION.json --output NEW_NATIVE_DIRECTORY
python qualify_loader.py --dataset-root NEW_NATIVE_DIRECTORY --official-source PINNED_SOURCE --cosmos-processor-cache FIXED_PROCESSOR_CACHE --output DATALOADER_SMOKE.json
python -m unittest test_native_dataset -v
```

Official source must be `51d4c89f72fda44cbf77285c6a8114b52676b8a1`, with unchanged
processor/pose/state-action/utils bytes. Existing tokenizer/image processor
files are loaded locally only. No neural model is constructed, loaded, or run.
The processor gives132-dimensional padding with10 valid action dimensions; the
fixed version does not emit a `state_mask`, so raw validity is enforced before
processing and its122 padded state entries are checked for zero.

`probe_contract.py` uses old PI0290 data only as a explicitly labelled plumbing
diagnostic. Its real-pixel, mixed-episode, variable-text, and two-worker passes
are not ARX5 data admission or ARX5 statistics. Actual formal ARX5 loader status
remains NOT_RUN until the independently admitted new episodes are supplied.

Scene/contact safety, independent T1 scoring, camera calibration/projection,
and actual video frame-count gates are separate evidence requirements, not
claims made by this loader alone.

`audit_data.py` implements these independent raw checks using saved high-rate
states and `mj_forward` only. It verifies all listed original file hashes,
641 actual FK/TCP/gripper/contact/camera snapshots, every high-rate time/limit/
velocity/control/safety row, and repeats the T1 score without teacher stages or
commands. It decodes available original videos and tests actual-data oracle/
clock/history mutations in memory. The script forbids all `mj_step` variants,
does not render, never modifies raw records, and writes measured CPU usage for
the sole owner to charge serially. Use the physical source matching the saved
episode rather than a later changed commissioning configuration.
