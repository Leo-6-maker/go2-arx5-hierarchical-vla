"""PI0252 A/G step-addressed RNG profile over the shared Trainer wrapper."""
from __future__ import annotations

import pi0251_step_rng as _rng


_rng.configure_namespace("PI0252", "M64_AG")
SEED = _rng.SEED
MAX_STEPS = _rng.MAX_STEPS
SCHEMA = _rng.SCHEMA
step_key_seed = _rng.step_key_seed
build_key_table = _rng.build_key_table
active_step_context = _rng.active_step_context
rng_context = _rng.rng_context
verify_identity = _rng.verify_identity
install_training_step_rng = _rng.install_training_step_rng
