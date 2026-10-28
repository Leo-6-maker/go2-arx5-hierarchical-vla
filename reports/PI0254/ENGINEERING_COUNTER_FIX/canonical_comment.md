## PI0254 canonical continuation: engineering counter repair sealed, before U resubmission

Authority remains commit `7e5a4aae69912ba8e593d50d313f288abc16a914` / canonical 5882151489. Scientific design and deployment selected=NONE are unchanged.

Initial U job **637485.hopper-m-02** is terminal F / Exit_status1. It failed in the newly added counting prehook during checkpoint recomputation, before optimizer update1. Raw optimizer intents=0; completed updates=0; update-audit events and checkpoint receipts are empty. Retained accounting: one native + four S4 network forwards; one recomputation prehook aborted before the model ran. No successful backward or training-quality claim is made. All original cards, logs, identities and failed training files remain archived with exact hashes.

The repair captures the step at outer head entry for audit logging and permits absent ContextVar only under the existing backward_recompute phase. It changes no RNG, loss, precision, model, gradient or checkpoint algorithm. Both actual Hopper offline CPU preflights passed, including the actual counter plus qualified four-step checkpointed Euler/mode contexts on synthetic tensors: 1 native +4 forward +4 recompute, four nonzero state gradients, restored modes. The Corelab four-point receipt-binding fixture also passed. Both independent online HF write/readback preflights passed; prefixes are empty and public.

Published execution/evidence freeze: **c30328f408f95c8aeb0e44cec65440266029ad93**.
- Training code commit: b747c4e587aa6a60d0bcc604fda89d3dd11f6d86.
- New U card SHA256: 0a5bd8865093dcf19365c06651f73c973e6ab0a72099a048855ea06800f00232.
- New R card SHA256: 6940aa4096a4c127a08260287ab226666a3462fae9f8e0d6b998853385539b87.
- Full card diff proves only source.execution_code_commit and the custom pi0254_train.py hash changed; all scientific fields are identical.
- [Deviation and preserved attempt](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/tree/c30328f408f95c8aeb0e44cec65440266029ad93/reports/PI0254/ENGINEERING_COUNTER_FIX).

P1 remains complete:320 valid inference queries/1280 inference head forwards. No new model/optimizer queries were used for this repair. U has consumed its initial submission; the next is its sole permitted engineering resubmission, same frozen G at initial cursor0/fresh AdamW because no optimizer state was consumed. R remains unsubmitted. Raw optimizer cap9216 and all endpoint/evaluation rules remain unchanged. Publication is not evidence of a restarted job; the actual submission receipt will follow. No PR pointer until the single final closeout pointer, no merge, sim, monitor or mail.
