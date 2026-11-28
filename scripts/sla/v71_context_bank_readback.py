"""Bind existing and early causal residuals to the exact prepared C1 reader order."""

from collections import Counter
import json
from pathlib import Path
import sys
from tempfile import mkdtemp

import numpy as np

from m38_gpu_preflight_v1 import pin, read, sha


def arrays(ref):
    with np.load(pin(ref["path"], ref["sha256"]), allow_pickle=False) as f:
        return {key: f[key].copy() for key in f.files}


def run(work):
    old = read(pin(work/"v7-full-tracking-context.a9de00e5/ACTUAL.json",
                   "9e15c1ec1b7cd69cbe86fbc8a24b3f5be0e52f1d693d8ac26ba0dbca4d07b70c"))
    new = read(pin(work/"v71-early-tracking-context.7fn4vw5o/ACTUAL.json",
                   "02f8365b20fe96cd2535741bed58f6cb2894d0c39692f105b694b921ee90c780"))
    root = work/"V7-C1-PAIRED-INPUTS-01"
    c1 = read(pin(root/"ACTUAL.json", "b0e286e212c2e5cfc2b9b65d9b4bea7fad210510c4999d502fbe89f07c43e67d"))
    original, early = arrays(old["context"]), arrays(new["context"])
    context = np.concatenate([original["accepted_minus_actual_se3_fixed_mount"], early["accepted_minus_actual_se3_fixed_mount"]])
    ticks = np.r_[original["native_step"], early["native_step"]]
    widths = np.r_[original["previous_actual_width_m"], early["previous_actual_width_m"]]
    with np.load(pin(root/"PREVIOUS_COMMAND.npz", c1["context_sha256"]), allow_pickle=False) as f:
        assert np.array_equal(ticks, f["query_tick"]) and np.array_equal(widths, f["width_m"])
        assert np.array_equal(f["consumed_start_tick"], ticks-25) and f["command_valid"].all()
    assert context.shape == (8245, 6) and np.isfinite(context).all()
    assert [(r["episode"], r["count"]) for r in old["episodes"]] == [(r["id"], r["windows"]) for r in c1["episodes"][:96]]
    assert [(r["episode"], r["count"], r["reset"]) for r in new["episodes"]] == [(r["id"], r["windows"], r["reset"]) for r in c1["episodes"][96:]]
    with np.load(pin(root/"BANK.npz", c1["bank_sha256"]), allow_pickle=False) as f:
        visits = f["treatment_target"][:, -2:].ravel()-8031
        assert (visits >= 0).all() and (visits < 214).all() and len(visits) == 500
    times, phases = early["seconds_since_handoff"], early["phase_audit_only"]
    # Audit strata only: no change to the sealed C1 bank or any model input.
    tiers = np.where(times == 0, "HANDOFF", np.where(times <= 2, "EARLY_0_2S",
        np.where(np.isin(phases, ["LOWER", "OPEN", "WITHDRAW"]), "LOWER_OPEN_WITHDRAW", "LATER_PATH")))
    output = Path(mkdtemp(prefix="v71-c1-context-readback.", dir=work))
    path = output/"CONTEXT8245.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, accepted_minus_actual_se3_fixed_mount=context,
            native_step=ticks, previous_actual_width_m=widths, valid=np.ones(8245, dtype=bool))
    value = dict(status="C1_8245_CAUSAL_CONTEXT_ORDER_AND_COMMAND_JOIN_PASS", code_sha256=sha(Path(__file__)),
        original_context_sha256=old["context"]["sha256"], early_context_sha256=new["context"]["sha256"],
        c1_receipt_sha256=sha(root/"ACTUAL.json"), c1_reader_order_sha256=c1["reader_order_sha256"],
        context=dict(path=str(path), sha256=sha(path)), windows=8245, episodes=98,
        exact_native_tick_and_previous_command_match=True, original_windows=8031, early_windows=214,
        audit_strata_population=dict(Counter(tiers.tolist())), original_C1_planned_500_visits_by_stratum=dict(Counter(tiers[visits].tolist())),
        actual_C1_training_visits=0, bank_changed=False, combined_RGB_training_reader_qualified=False,
        A2_model_input_or_projection_qualified=False, new_model_forwards=0, optimizer_updates=0, physics_integrations=0)
    receipt = output/"ACTUAL.json"
    receipt.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    print(receipt, sha(receipt), flush=True)


if __name__ == "__main__":
    run(Path(sys.argv[1]))
