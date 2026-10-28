# Historical Owner Source Reference

This snapshot preserves the local Python orchestration used during adoption, storage maintenance, data delivery, PBS admission, training, two-SSH relay and closeout. The 166 sources were exported from the private owner directory, not scraped from credentials or chat.

**Do not execute or deploy these files.** Each `.py.txt` includes a top-level stop guard before imports (after any required future import). Some contain destructive historical maintenance logic or old clocks. Their presence is neither current authorization nor an instruction to repeat them. Removing the guard is not an approved recovery path.

Paths, private accounts and private network addresses are role placeholders such as `@HOPPER_SCRATCH@`. This is a source-reference archive, not a tested portable launcher. The exact runnable originals remain private with their original hashes in MANIFEST.json. No JSON runmaps, logs, keys, passwords, tokens, datasets, videos, model tensors or environments were included. The private shell entry is already represented by the tracked CoreLab `ENTRY_IMPLEMENTATION/enter.sh` snapshot.

Review or extract reusable logic into the normal tracked modules only under a new, scoped engineering task. Obtain exact machine bindings locally; never recover them from public text or ask for tokens. Existing production runtime capsules must remain immutable.

The exporter lives at `scripts/maintenance/export_owner_reference.py`; its tests cover path/secret refusal, preflight-before-write, JSON exclusion, overwrite refusal and the early execution guard. All 166 exported sources parsed and compiled; that does not mean their historical workflows were rerun or newly qualified.
