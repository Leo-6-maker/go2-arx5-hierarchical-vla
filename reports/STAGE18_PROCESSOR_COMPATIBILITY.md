# Stage18 LeRobot and Official GR00T Processor Compatibility

## Logical record

The dependency-light bridge emitted six logical LeRobot-compatible records
with:

- `observation.state`: 10D `StateV1`;
- `action`: first desired physical 10D command;
- `future_action`: `[40,10]` desired EEF+gripper trajectory;
- `action_timestamps`: 40 native source timestamps;
- body/wrist image paths and language;
- metadata identifying desired labels and retained achieved audit state.

No LeRobot package was installed in the Stage10 environment.

## Official processor evidence

The existing D3 official processor log for the target-action dataset records
finite 10D state/action groups, padded universal shape `[40,132]`, 400 active
values, and EEF round-trip max absolute error `2.9395358e-08` with gripper
error `0`. This is prior D3 processor evidence, not a task-success result.

The new Stage18 processor-only smoke, launched with the complete Stage10
`env.sh` and venv, passed `Gr00tN1d7Processor` construction and processing.
It produced processed state `[1,132]`, universal action `[1,40,132]`, and
decoded physical action `[1,40,10]`; all values were finite. Using D3
target-action statistics, the physical round-trip maximum absolute error was
`3.9073288e-06` and gripper error was `0`. The processor was not run as a
model forward and no output was sent to WBC.

An earlier invocation omitted Stage10 `env.sh` and returned 401 because it did
not use the established HF cache context. That transient invocation is not
the final result. The remaining Stage18 deviation is the independent D3
camera timestamp/metadata mismatch.
