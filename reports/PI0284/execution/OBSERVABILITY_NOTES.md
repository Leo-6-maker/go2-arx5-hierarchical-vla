# Actual audit boundaries

- Native loss, both optimizer-group learning rates, observed module modes, consumed encoded-input identity and native noise/time identity are recorded for every formal microbatch.
- Detailed gradient inspection executes for updates 1–8 and multiples of 128 (including the explicitly listed checkpoint boundaries). Thus a full 2048-update arm has 48 detailed microbatch records out of 4096. Other detailed gradient values are UNOBSERVED. The inherited compact status string contains `INTERVAL16`; that string does not describe the executed interval. Training code and healthy runtime are retained unchanged.
- The preparatory `opening_first8_draws` computes differences internal to positions 0–7. The final exposure audit uses the original complete-episode difference, including the transition into anchor position 0. These two definitions are retained and must not be treated as interchangeable counts.
- Passive CUDA allocated/reserved bytes are memory telemetry, not GPU utilization. Loader and forward/backward durations are measured intervals, not a proof of achieved real-time control.
- DROID numeric drift against saved pristine predictions is descriptive. It does not provide task-success ground truth or establish complete pretrained behavior retention.
