# ADR-003: Gripper remains a parallel channel

**Status:** Accepted for the initial baseline

**Decision:** WBC consumes the 9D EEF trajectory. The GR00T gripper scalar is
kept as a separate timestamped channel.

**Rationale:** Stage13B bound the delivered physical WBC action as Go2 12D +
ARX5 arm 6D = 18D and found no gripper actuator ownership.

**Consequence:** No gripper value is silently appended to WBC preview or joint
action tensors. Hardware/sim gripper execution is out of Stage14 scope.
