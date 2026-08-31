# Stage14 generator-support connection

Status: `DEFERRED UNTIL REFERENCE INJECTION IS QUALIFIED`

Stage13B reconstructed the delivered historical generator distribution
node-locally from audited source formulas and retained a 10,000-trajectory
descriptor sample. Stage14 does not claim that this reconstructed sample is a
realized training visitation set. The four levels remain separate:

1. generator expressivity;
2. configured parameter support;
3. empirical reconstructed sample;
4. realized training occupancy and policy generalization.

The native external-reference gate must be resolved before attaching E1/E2
tracking error to a generator-support distance. Otherwise a frame/anchor error
would be mistaken for distribution distance or policy OOD. After the explicit
benchmark-to-native transform is bound, the planned analysis is to report
descriptor distance versus position/rotation error and termination by class,
with the label `GENERATOR_SUPPORT_DISTANCE`, never `TRUE_TRAINING_OOD`.

The generator's ability to represent a trajectory family remains an
expressivity claim. It does not imply that the trained policy tracks every
trajectory in that family.
