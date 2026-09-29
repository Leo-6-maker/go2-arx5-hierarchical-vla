# CPU recovery fixture follow-up

This additive check follows the immutable original raw pilot archive. It does not change any pilot/scorer/collector, admitted model, dataset, normalization, candidate or training budget.

The existing qualified _euler_rollout and _module_eval implementation is reused on a scalar CPU Linear fixture. Four Euler states receive nonzero gradients under checkpoint recomputation and all module modes/phase restore. The6000 cosine/warmup10 scheduler is checked at0/10/6000; a synthetic cursor3000 saves/restores model, AdamW, scheduler, Python/NumPy/Torch RNG and cursor. Continued fixture parameter/optimizer/scheduler states exactly match an uninterrupted fixture after deliberately consuming11 unrelated restore RNG draws. No GR00T model is loaded or trained.

Receipt: resume_s4_cpu_fixture_receipt.json. This proves the CPU primitive fixture, not a real6000 GR00T Trainer recovery, real dataset cursor or processor augmentation identity. Those remain explicit requirements once admitted data and a subsequent execution contract exist. The draft sampler/source-visit noise table has6000 synthetic rows and0 actual training rows; no synthetic rows are presented as data.

The contemporaneous PI0254 CPU source check demonstrates that original stochastic crop/color jitter can escape Torch-only RNG pairing. A future real paired processor must bind actual transformed image/input identities as well as native tau/epsilon/noise. This is a recorded missing boundary, not permission to change PI0254 U/R or execute a new training arm here.
