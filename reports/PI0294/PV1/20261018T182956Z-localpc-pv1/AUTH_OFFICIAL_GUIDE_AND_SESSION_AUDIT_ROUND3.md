# Official Guide And Client Session Audit - Round 3

Status: `CLIENT_METHOD_VERIFIED_NATIVE_AUTH_NOT_RESTORED`.
Date: 2026-10-20, Asia/Singapore. Authority remains the original PV1 window
and POST-CR1 engineering goal `5d1393154054b977134b7d8c456f66ca478d0bf9`.
This is a diagnostic receipt, not a new scientific protocol or successful login.

## Official Sources Actually Read

- [NUS HPC home](https://nusit.nus.edu.sg/hpc/) and
  [Hopper onboarding](https://nusit.nus.edu.sg/hpc/hopperonboarding/): read
  through the local browser. The official Hopper presentation link redirects
  to NUS Microsoft/SharePoint sign-in. Its full contents were NOT read; no
  third-party repost is substituted for the official presentation.
- [NUS introductory guide](https://nusit.nus.edu.sg/hpc/introductory-guide-for-new-hpc-users/):
  expanded account/access sections read. They identify NUS-ID authentication
  across HPC hosts. They do not document a token TTL or renewal on every SSH
  connection. The site-installed `hpc project` wrapper separately directs
  users with failed budget authentication to `amgr login`.
- [Altair PBS Professional 2024.1 Budgets Guide](https://help.altair.com/2024.1.0/PBS%20Professional/PBSBudgetGuide2024.1.pdf):
  sections 1.2.1, 2.8.5, 2.8.7 and 2.8.9.3 read; printed BG-48 configuration
  table visually checked. Download SHA256:
  `b405d0c4bec66c6996ffbcb47763178f7040fd2dde79d2a7b5b23a1825d423fb`.
  This is vendor documentation, not proof of NUS's private server settings.

## Native Read-Only Session Comparison

Compared the existing multiplexed SSH connection and a fresh non-multiplexed
SSH connection, using the configured alias and strict host-key verification.
Each ran an inherited-environment and a clean-Python-environment project query.

| Check | Actual result |
|---|---|
| Principal, UID, login host, resolved Home | Equal across both SSH contexts |
| Administrator `/etc/am.conf` | Same hash; owned by root |
| User `AM_*` environment overrides | None |
| PYTHONPATH/PYTHONHOME/CONDA_PREFIX/VIRTUAL_ENV | Unset; cleaning them did not change the result |
| Observed gateway connection vs administrator configuration | Match in all four cases |
| Four project queries | All CLI exit 1, HTTP 401, code 15006 |
| Query response intervals | Approximately 0.0013-0.0306 seconds |

Configuration hash:
`dc72cc3dee4e365f48081eb0b4e4360d957e5f98372a3a9a47113bb059c58cfd`.
Private endpoints, host-bound paths and full native receipts remain local.
Prior round2 independently checked own-cache permissions as 0600/readable/
writable with a writable parent. That does not establish token validity.

## Input And Official Client Method

Two additional bounded checks used NO real credential:

1. A dummy string with punctuation, case and digits traversed the real nested
   SSH/PTTY input path and matched the expected value remotely. No `amgr`
   command or budget authentication request was made by this input fixture.
2. The official installed CLI was run with dummy input, all socket connections
   and child processes denied, and Home writes/mutations denied. It built a
   POST to a login/auth route, without a query string. Its Basic authorization
   exactly matched the current OS username plus the dummy input. Only scheme,
   boolean equality and request-type metadata were retained, not header values
   or bodies. Both request-shape fixtures stopped at the denied socket call;
   no Home mutation was attempted.

Thus the tested client login path uses the supplied username/password, not
the expired cached token as a Bearer login credential. This is NOT evidence
that the real password is accepted or that a live login succeeded.

## Interpretation And Limits

No client-side login-method, wrong-Home, Python-environment override, stale
SSH-control-session or gateway-selection mismatch was found by these checks.
The previous real normal-login trace still supplies the distinct evidence:
HTTP 401 after about 10.043 seconds, followed by project code 15006.

The vendor table defines `AM_AUTH_TIMEOUT` as a server-side option with default
10 seconds and marks it unused by clients. Its absence from this client
configuration is therefore not itself a defect. The observed login interval
is consistent with that timeout, but it does NOT prove the server's actual
setting or distinguish backend authentication failure from credential rejection.
Server/AMS/directory-authentication logs are needed to settle that distinction.
Do not change global configuration, extend the 175-second gate, reset a
password, delete a token file, use another account, or try teller-service
`sshlogin` as an assumed ordinary-user refresh mechanism.

The next useful external evidence is the site's actual login/AMS failure
reason for the approximately 22:01 SGT normal-login trace, including whether
the configured authentication timeout fired. No nTouch ticket was submitted.
Repeated identical password attempts, more PBS or a different memory request
would not test this remaining hypothesis. Native authentication is still not
restored; project membership/balance are not freshly qualified.

## Effects And Original-Budget Accounting

- New real password logins in this round: 0. Read-only project queries: 4.
- Two network-denied official-client request fixtures and one SSH dummy-input
  fixture: CPU only, not login success or policy evidence.
- Idempotent original-PV1 settlement key:
  `POST_CR1_AUTH_DOCUMENTED_SESSION_DIAG_ROUND3`; Hopper CPU upper 180 seconds,
  local CPU upper 300 seconds, Core CPU 0. No budget or clock reset.
- New PBS/model/neural-query/physics/optimizer effects: all 0; unknown effects [].
- Campaign H200 used 20473 seconds, reserved 0, PBS 10/24. B0 remains 14/48
  (9 successes, 5 valid task failures), optimizer updates 0.
- Original numerical runtime candidate remains `30e2e1526394787d892c153dd68a49afe45c7b46`,
  exact Actions 39436335491/job 118674212836 succeeded. This round changes
  documentation only; private diagnostic helpers are not a deployed owner loop.
- WBC/plant/gripper-PD/S1/scoring/R0/139-bank, B0-before-training, one H200,
  final 20-hour reserve and original deadline unchanged. Child11 stays closed;
  selected/deployment NONE; CR1/COV1 closed; CR2 awaits its own scientific GO.

**No PBS may be submitted before real same-context project authentication and
all original native admission gates pass.**
