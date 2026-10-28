# PI0290 Actual Command Ledger

Historical executed commands only; **not authorization to rerun the twelve IDs**. Existing episode directories and start ledger reject overwrite/retry. Retrieve immutable frozen code from the private raw HF prefix or git sourcec4853f3 for CPU inspection. Credentials and host-specific runmap stay private.

All Corelab Python calls used the isolated `PI0290/runtime/venv/bin/python -B`, `PYTHONPATH=PI0290/code/src`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` and at most four affinity CPUs. `ROOT` below denotes the verified private Corelab SSD PI0290 root.

```bash
# D0, source9809117609b77cae4583b787906c037f7f1ce0ab; zero integration
python scripts/pi0290_task_data/runtime_probe.py --root "$ROOT" --source-sha 9809117609b77cae4583b787906c037f7f1ce0ab
# After occupied GPU preflight, actual CPU software render
MUJOCO_GL=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json LIBGL_ALWAYS_SOFTWARE=1 CUDA_VISIBLE_DEVICES='' LP_NUM_THREADS=4 python scripts/pi0290_task_data/software_render_probe.py --root "$ROOT"
```

The same software-render environment applied to all actual physical runs and optics/reset diagnostic renders. No forced NVIDIA device or new GPU allocation was used for episodes.

```bash
# C2900 V1, actual safety stop, no repeat
python scripts/pi0290_task_data/run_episode.py --root "$ROOT" --id C2900 --source-sha 126abb6347641653e8c02813d32eab237cdbb912
# C2901 V2 actual empty-grasp negative; original postprocessor recovered later
python scripts/pi0290_task_data/run_episode.py --root "$ROOT" --id C2901 --mode empty_grasp --source-sha 66c2f5bc58571c046c9db5ff55594a475a19b3fb
python scripts/pi0290_task_data/recover_postprocess.py --root "$ROOT"
python scripts/pi0290_task_data/run_episode.py --root "$ROOT" --id C2902 --mode single_side --source-sha 50bacdf2de2cf505b5cebfd6d9dc9a6b45fa5e52
python scripts/pi0290_task_data/run_episode.py --root "$ROOT" --id C2903 --source-sha 50bacdf2de2cf505b5cebfd6d9dc9a6b45fa5e52
# CPU-only actual saved RGB video; done once for C2903 and each pilot
python scripts/pi0290_task_data/video.py --root "$ROOT" --id C2903
```

CPU contract tests, original-data audit and camera/reset preflight passed before the pilot seal. Optics diagnostics used saved C2903 qpos/qvel and mj_forward, not integration. The existing fixed-revision tokenizer path is in the private runmap; only Tokenizer.from_file was used, not a network.

```bash
python scripts/pi0290_task_data/prepare.py --root "$ROOT" --source-sha c4853f3b7bd9747dff485d58780a0c0abb1e3402
python scripts/pi0290_task_data/prepare.py --root "$ROOT" --source-sha c4853f3b7bd9747dff485d58780a0c0abb1e3402 --pilot-seal
# Actual serial driver invoked this once for each P2900..P2907, then video.py
python scripts/pi0290_task_data/run_episode.py --root "$ROOT" --id P2900 --source-sha c4853f3b7bd9747dff485d58780a0c0abb1e3402
# D3 post-pilot CPU only
python scripts/pi0290_task_data/audit_data.py --root "$ROOT" --tokenizer "$EXISTING_FIXED_TOKENIZER"
python scripts/pi0290_task_data/closeout_data.py --root "$ROOT" --audit-source-sha 15562ea29eb9448ef2c6eb0b48a741d85d438d86
# D4 archival/derived native windows only, existing HF auth reused
python scripts/pi0290_task_data/archive_private_hf.py --root "$ROOT"
python scripts/pi0290_task_data/export_native_windows.py --root "$ROOT" --upload
python scripts/pi0290_task_data/make_evidence_figure.py --root "$ROOT"
python scripts/pi0290_task_data/final_verify.py --root "$ROOT"
```

HF raw READY7511-byte SHA was additionally read on Hopper through stdlib HTTPS, without raw download or model/PBS calls. Git pushes were normal fast-forward; a concurrent maintenance update caused one rejection and was read/merged without force-push. User-designated local WBC was checked only with hashes/source reads, never build/run/smoke scripts.
