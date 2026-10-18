# Recorded Transport And Progress-Journal Repair

[Actual bounded CPU observations](N1_ACTUAL_RECORDED_TRANSPORT_AND_PROGRESS_JOURNAL_REPAIR.json)
were collected by the existing owner after native673388 terminated and all930
scientific draws were classified. This is new mechanism evidence and a narrow
repair, not a retrospective unique explanation of that allocation's ACK timeout.

## Actual Route Test

The original local-PC -> AMD ->4080S and local-PC ->Hopper SSH paths transported
one original saved double-camera observation packet80 times. Every Core reply
and Hopper checksum matched;160RPCs,15,252,620 logical request/response bytes,
95,081-byte recorded JSON, median two-site cycle0.03879s, maximum1.24919s.
These are stored pixels/checksums, NOT fresh physics observations, neural
queries or task successes. No new PBS, GPU, model, physics or training ran.
All four remote checksum processes and all local SSH handles are terminal;
original manifests and engineering STOP bytes are unchanged. Scientific
counts remain20attempts/930draws/zero new updates.

## Controlled Before/After Mechanism

The client formerly synchronously fsynced a local wire-progress receipt after
writing its one request but before reading the reply. A21s delay injected only
at that progress-fsync call on the real CPU-only SSH path measured21.022s.
The old published8d34b22e journal exhausted the unchanged20s deadline and poisoned
the session despite the remote checksum responder returning its reply.

The repair keeps the durable intent BEFORE any wire bytes. It records written
progress in memory, then persists that exact timestamped progress together
with response or failure. A crash before that batch retains the durable unknown
intent and must not retry. The successful response batch is still durable
before return; pre-wire and response persistence still consume the original
whole-call20s watchdog. Remote mutation journals/counts remain untouched.

With the same written-fsync fault hook armed, the repaired isolated request
returned in0.576s, parsed its reply and exited SSH0. The injected sleep is not
triggered because no progress-only fsync occurs in flight. This removes a
demonstrated failure mechanism; it does not prove the actual historical ACK
failure was caused by slow storage. Nominal measured fsync maxima were only
2.06ms(Core-channel journal) and1.29ms(Hopper-channel journal).

## Preserved Diagnostic Failure And Finite Close

The initial probe controller assumed its old close method would observe SSH
exit0 after the injected timeout. That assertion actually failed: a large
unread reply can fill local stdout while close waits without draining it.
Its raw close receipt remains returncodeUNKNOWN. After the controller exited,
the exact local PID/start handle was verified missing and the remote checksum
worker independently verified terminal; no exit code is invented, no process
was signalled, and the BEFORE request was never resent. Only the previously
unattempted AFTER fixture was continued, within the same two-request cap.

A shared bounded close helper now closes stdin and drains abandoned stdout
for at most the original30s and one2MiB packet bound. It discards and hashes
bytes, NEVER parses them as an ACK, clears no poisoned state, retries nothing
and signals no process. Unknown liveness stays unknown. CPU subprocess tests
reproduce an unread200,000-byte pipe response and verify bounded draining,
idle timeout and oversized-output refusal. This close adapter was prepared
after the SSH probe; new real-SSH/native acceptance is still pending, with no
extra controlled remote fault request added.

## Required Next Boundary

No scientific capsule was changed or new allocation submitted. Retain both
original engineering STOPs, all valid negatives and both partial attempts.
After exact publication/CI, a confirmation-specific cold boundary must bind
these changes, all prior terminal effects/processes and the original plan/
candidate/clock. Only the remaining original second confirmation allocation
may proceed, with its OWN600s/80cycle/fault qualification before model load.
No third confirmation or13th source-family PBS is authorized.

The original fixed40 conditions, zero new training, original20/60/900s and
2700s closeout, frozen WBC/S1/score, TF-R3 STOP and selected/deployment NONE
remain. Full40 results, final video publication and N1 final closeout are
still incomplete; code/probe PASS is not the scientific end state.
