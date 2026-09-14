{
  "artifacts": [
    "pi0182_result.json",
    "origin_classified_rows.jsonl",
    "s20_exposure_counts.json",
    "stats_processor_binding.json",
    "target_leakage_negative_test.json",
    "training_package/MULTI_VISITED20_V1_ADMISSION.json"
  ],
  "hard_zeroes": {
    "backward": 0,
    "gpu_execution": 0,
    "h200_submissions": 0,
    "hardware_steps": 0,
    "holdout_f_read": 0,
    "model_constructed": 0,
    "new_model_calls": 0,
    "optimizer_steps": 0,
    "simulator_steps": 0,
    "training": 0,
    "validation_e_read": 0,
    "wbc_steps": 0
  },
  "model_inputs": [
    "body_rgb",
    "wrist_rgb",
    "state_v1",
    "language"
  ],
  "origin_rule": "q0=INITIAL_SUPPORT; q1-q3=AUTONOMOUS_VISITED",
  "remaining": [
    "PI-selected Legacy/L1 base checkpoint",
    "H200 allocation and launch"
  ],
  "s20_stats_recomputed": false,
  "summary": "CPU/static PI0182 admission for MULTI_VISITED20_V1; no training or runtime execution occurred."
}
