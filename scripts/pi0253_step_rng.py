"""PI0253 FULL namespace; use the existing forward/backward RNG isolation."""
import pi0251_step_rng as _rng

_rng.MAX_STEPS = 10000
_rng.configure_namespace("PI0253", "FULL")
SEED, MAX_STEPS, SCHEMA = _rng.SEED, _rng.MAX_STEPS, _rng.SCHEMA
step_key_seed = _rng.step_key_seed
build_key_table = _rng.build_key_table
active_step_context = _rng.active_step_context
rng_context = _rng.rng_context
verify_identity = _rng.verify_identity
install_training_step_rng = _rng.install_training_step_rng
