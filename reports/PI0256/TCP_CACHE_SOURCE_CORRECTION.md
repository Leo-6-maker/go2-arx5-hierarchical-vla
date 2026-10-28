# Source-proven Preparation-v2 cache correction

This is the reserved source correction allowed by PI0256 section3, preserving
the sealed measured-TCP/fresh-observation contract. It is not a new curve, pose,
threshold, controller or task. Original P0 and all executed v1 outcomes remain
immutable. No PASS is inherited across versions.

Actual A00 new `post_reset.json` and first `raw_ticks.jsonl` pre-step record
share simulation timestamp 0.019999999552965164. Across camera fetch, cached
position changes from [0.1818458587,-0.0437117219,0.5640423298] to
[0.0503753275,-0.0122959707,0.6088343263]. The difference equals the rotated
native TCP offset [0.1424,0,0.0001057] within1e-7 m. There is no physics step.

Actual native source binds `end_effector_state` as a VIEW of `rigid_body_state`
at initialization. The unchanged native post-physics hook copies link6 and
adds the TCP offset in-place to that view. Actual camera helper calls
`fetch_results`, which overwrites raw native state and thus overwrites that
derived TCP cache. This also makes a post-camera one-time goal-error refresh
use link6 instead of the sealed measured TCP. A valid physical FAIL of v1
remains a FAIL of v1, but it does not demonstrate correct v2 cache behavior.

The read-only source proof compiles only the actual hook's four cache-copy/TCP
shift statements on CPU. Original view loses TCP on simulated raw-state fetch;
a separate derived buffer retains it, leaves raw state unchanged and is still
updated by those identical native statements. This is a cache mechanism test,
not an actor rollout or evidence that preparation will qualify.

Correction: clone this derived EEF buffer ONCE in the project-owned B/C
constructor, before the implicit reset step. Existing native WBC methods,
weights, gains, dynamics, ordering, histories and reset distribution stay
unchanged. Assert unchanged physical/root/DOF/clock/reset/episode and raw-state
hashes, unchanged cache values, and distinct storage. Do not reanchor, clear
history or repeatedly reset caches. Keep identical C0 curve/FK binding,
thresholds, reference/camera cadence and original soft joint predicate.

`PREPARATION_V2_TCP_SYNC_CONTRACT.json` registers six proof checks Bfix00..05
under the same old conditions. They use the six reserved preparation starts.
Only Bfix6/6 can release original fresh C00..05, all on this single effective
version. If all C run, total A6+B-v1-6+Bfix6+C6=24, the hard preparation cap.
No source correction permits ignoring a valid joint/safety/quality failure.

Stage the correction separately while the original finite batch is healthy.
Freeze source, CPU proof and actual numeric reference before the first proof
check. No active worker is edited or restarted. Zero GR00T queries/updates and
zero task-phase ticks; selected/deployment NONE.
