# Core–Vanda shared M32 implementation

The scientific contract and task list are in [the November 3 campaign](CORE_VANDA_POLICY_DELIVERY_CAMPAIGN_20261103.md). This file describes the changed application seams; it is not a deployment receipt.

- `shared_site.py` binds Vanda to its real PBS allocation, project entitlement and enforced project scratch quota. Core retains its original local-path admission.
- `activation_offload.py` declares a separate CUDA-resident activation policy. Unused frozen category tables retain their existing CPU placement; parameter values, precision, losses, M32 scope and sampling are unchanged. Zero activation-copy counts are required for this profile, rather than fabricated offload counts.
- `shared_vanda.py` uses the existing native relay and canonical onceIO records. One finite allocation executes native10, full 42-state restore and raw TRAIN16 inference, then reloads raw for fresh8192. Native10 measurements must fit all8192, four saves/restores, export and close into remaining PBS time. Each worker exits before the next model is loaded.
- `shared_resume.py` admits an optional complete, closed checkpoint from the same scientific run. Optimizer, scheduler, RNG, sampler cursor and prior committed trace remain required. A partial failure never silently starts a fresh baseline.
- `shared_delivery.py` exports only the complete CLOSED8192 inference subset. The original full checkpoint header and restore witness remain attached; optimizer/scheduler/RNG training archives remain at the producer. Inference subset admission does not weaken full training-resume validation.
- `native_relay shared-dev` requires the delivered checkpoint's own Core TRAIN16 inference qualification before its one original DEV68. Intermediate endpoints do not automatically receive behavioral panels. Fixed 68-row results, coverage and policy metadata are saved, including unrun and engineering failures.
- `supervision_audit.py` analyzes saved actual joint/finger/contact records and submitted commands without simulation or training. Its sidecar is neither an expert label nor student input.

Actual ownership transfer, PBS submission, native results, output transport and resident acceptance require the existing operator's real receipts. A source merge or CPU test does not establish them. Existing Core capsules are immutable, BOOK scheduling and the original November 6 dispatch cutoff remain in force, and no infrastructure or target safety boundary changes.
