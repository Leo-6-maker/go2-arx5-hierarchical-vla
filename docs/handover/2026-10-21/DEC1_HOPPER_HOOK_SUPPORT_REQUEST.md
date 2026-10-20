# Hopper Execution-Hook Investigation Request

Status: **DRAFT_NOT_SENT_NO_TICKET_RECEIPT**. Use the existing authorized HPC
support channel. Do not include passwords, tokens, raw environment dumps,
private host paths or model/data payloads. This is not a request to bypass a
hook, release a system hold or increase resources.

## Request

Please inspect the server/MOM hook logs for owned job
`679029.hopper-m-02`, project `CFP04-CF-007`, at the following Singapore times:

- Allocation trace begins2026-10-21 00:13:49 SGT on hopper-27.
- Native stime2026-10-21 00:13:50 SGT; obittime00:14:10 SGT.
- User-visible trace says: `hook rejected request, requiring job to be rerun`.
- Job entered H with Hold_Types=s, Exit_status=-16, run_count=1.
- The rejecting hook's identity and detailed reason were not exposed in the
  user-visible trace. No PBS_READY, launcher, START_NEURAL or model receipt
  was produced. The scientific application did not begin.
- Request: one H200,12 CPU,225GB host memory,8h. No system-hold release or
  node/hook bypass was attempted.

After checking owner privileges against the installed PBS documentation, the
exact failed prelaunch job was first user-held to exclude recovery and then
deleted once. It is now native F, both leases are CLOSED, and the old GPU
reservation is cleared. Original20s accounting plus1s fractional uncertainty
was retained as21 charged seconds, not recorded as a zero-cost failure. The
total campaign remains20494 charged GPU seconds and11 submissions. This is
an investigation of the retained failed attempt, not a request to recover or
release a live job.

Could you confirm:

1. Which hook rejected this request, and its actual logged reason?
2. Whether the cause was node/system state, account/project admission or a
   user-correctable submission issue, with the exact required correction?
3. Whether an ordinary new cold submission under the same active project is
   now appropriate, or whether a site condition remains unresolved?

The local owner separately experienced a read-only SSH hostname-resolution
failure. We have not proved this caused the site hook rejection. Likewise,
intermittent AMS token expiry has been observed independently, but it is not
being asserted as the hook's cause. We will not repeatedly submit or alter
authentication in place of the requested site diagnosis.

## Evidence Index

- Historical first hook observation:
  `reports/PI0294/PV1/20261018T182956Z-localpc-pv1/PV1_CHILD12_SITE_HOLD_AND_CR2_STATIC_ACTUAL.json`.
- Actual terminal disposition and settlement:
  `reports/PI0294/DEC1/20261018T182956Z-localpc-pv1/DEC1_HELD_JOB_DISPOSITION_AND_SETTLEMENT.json`.
- The existing local owner retains exact native job/trace/accounting receipts
  privately and can provide a minimally scoped copy through the approved
  support channel. No support response or ticket has yet been verified.
