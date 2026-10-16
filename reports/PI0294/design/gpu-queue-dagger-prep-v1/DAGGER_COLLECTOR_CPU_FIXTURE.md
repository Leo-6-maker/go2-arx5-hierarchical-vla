# Q3 Chunk Transaction Fixture

Actual results: COLLECTOR_FIXTURE_RECEIPT.json. No live collector was deployed.
Implementation is an in-memory transaction state machine plus independent
correction-label and aggregation contracts. It is not the durable remote relay.

State order:
WAIT_OBSERVATION -> OBSERVED -> SELECTED -> WAIT_ACK -> WAIT_OBSERVATION.
After the last actual prefix the final fresh observation leads to COMPLETE.
STOPPED and UNKNOWN_STOPPED_NO_RETRY cannot dispatch again.

Fake environment behavior is an injected ACK containing an exact original
prefix and actual integer tick progress. No callback invokes mj_step/render or
a neural model. Zero objectcontact still advances source time but never turns
into success. Partial advance is explicit early-end or safety failure, not a
complete8-step chunk. No-motion, earlyend, both-rejected, missing correction,
stale/repeated observation/clock, duplicate dispatch/ACK and altered prefix have
negative tests. Reference rejected before S1 can use NOT_RUN, not fakeS1PASS.

Proposal, dispatch intent, installed prefix and observed result are distinct.
Hash/noise identity, source scene/split/model/stats, actualEEF/gripper and RGB
hashes persist in records; BASE observation is absent for source. Privileged
truth is not a transaction observation field. Expert invalid proposals remain
unavailable; counterfactual complete expert labels can be retained even when
student executes. Failure/diagnostic and intervention banks are distinct;
successful_BC_demo_admitted is alwaysfalse pending independent episode audit.

Student visitation requires a previous actual student prefixACK on the current
observation/tick. SYNTHETIC_CPU_FIXTURE is explicitly ineligible for scientific
student-visited counts and optimization. Fixed scene-group split registries
override re-labelled record claims; DEV never enters TRAIN. Pool capacity is
finite and duplicate/corrupted records fail rather than silently evict/repeat.

The JSON schema describes transaction records; signed/verified private proof
loading, full feature/data export, persistent crash recovery and actual
collector runtime remain future work. Existing E0 relay is not modified.
