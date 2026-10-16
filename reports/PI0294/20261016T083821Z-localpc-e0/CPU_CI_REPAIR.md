# CPU CI Fixture Repair

The GitHub check on `95a33e04d93a122eaff85bbfef742cfc29654972` passed the
default suite and isolated PI0292 contracts, then failed the isolated PI0294
suite. The same nine failures had already been reproduced on the untouched
E0 protocol release `22e5f0baeed0af662a7cf1c97edae1f4060947fe`.

Root causes:

- Continuation fixtures still described the superseded R1 approval and CPU
  caps, while production admission correctly requires R3 mode separation.
- The resident-stage positive fixture lacked R3's real S1 qualification input;
  its invalid-main negative fixture lacked the parent binding needed to reach
  the intended resume-path rejection.
- One R4 test patched an absolute module name different from the module that
  pytest had imported, leaving the real file reader unmocked.

The repair changes test fixtures and patches the actually imported module.
It adds explicit rejection coverage for the old approval, disabled mode
authority, insufficient teacher successes/clone contexts, teacher rejection
and real native safety failure. No production admission, scientific limit,
physics, S1, WBC or model behavior changes. The pending E0 capsule is unchanged.

Actual local post-repair verification:

- Default suite: 762 passed, 14 explicit skips, 58 subtests passed; 17.60s.
- Isolated PI0292: 108 passed, 150 subtests passed; 1.81s.
- Isolated PI0294: 186 passed, 136 subtests passed; 4.46s.
- One existing default-suite tensor conversion warning remains.

The isolated suites do not run physical episodes, load neural models or submit
PBS. GitHub post-repair status requires a separate native check readback; local
success alone is not a GitHub green-check claim.
