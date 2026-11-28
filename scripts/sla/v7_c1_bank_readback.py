"""Independent CPU checks of the C1 paired bank against immutable original labels."""

from collections import Counter
import io
import json
from pathlib import Path
import sys
import tarfile
from tempfile import mkdtemp

import numpy as np

from m38_gpu_preflight_v1 import pin, read, sha


def run(work):
    prepared = work/"V7-C1-PAIRED-INPUTS-01"
    receipt = read(pin(prepared/"ACTUAL.json","b0e286e212c2e5cfc2b9b65d9b4bea7fad210510c4999d502fbe89f07c43e67d"))
    original = work/"v7-c1-input-preparation.IDcxBJ"
    package = Path("/mnt/local/hdd1/UserData/liuyu-nfs/Projects/GR00T-WBC/WM02/slago2-embodiment-a93-r3-hub-20261127-01")
    index = read(pin(package/"index.json",receipt["fixed_original_index_sha256"]))
    row_meta,previous,ticks,masks = [],[],[],[]
    for ep in index["episodes"]:
        with tarfile.open(package/ep["path"],"r") as tar:
            with tar.extractfile("labels.npz") as stream,np.load(io.BytesIO(stream.read()),allow_pickle=False) as f:
                for i in range(ep["windows"]):
                    row_meta.append((ep["episode"],str(f["phase"][i]),ep["task"],ep["scene_group_sha256"]))
                previous.extend(f["prior_actual_width_m"]);ticks.extend(f["native_step"]);masks.extend(f["valid"])
    assert len(row_meta) == 8031
    for ep in receipt["episodes"][-2:]:
        root = Path(ep["source"])
        source = read(pin(root/"MANIFEST.json",ep["manifest_sha256"]))
        data, = source["episodes"]
        with np.load(pin(root/data["labels"]["file"],data["labels"]["sha256"]),allow_pickle=False) as f:
            assert len(row_meta) == ep["offset"]
            row_meta.extend((data["id"],str(phase),data["task"],data["scene_sha256"]) for phase in f["phase"])
            ticks.extend(f["native_step"]);masks.extend(f["valid"])
        with np.load(pin(root/data["gripper"]["file"],data["gripper"]["sha256"]),allow_pickle=False) as f:
            previous.extend(f["prior"])
    assert len(row_meta) == len(previous) == len(ticks) == 8245
    with np.load(pin(prepared/"BANK.npz",receipt["bank_sha256"]),allow_pickle=False) as f:
        bank = dict(f)
    with np.load(pin(original/"BANK.npz",receipt["original_bank_sha256"]),allow_pickle=False) as f:
        prior = {key:f[key][:250] for key in ("target","robodojo","libero","auxiliary_slots","rates")}
    with np.load(pin(prepared/"PREVIOUS_COMMAND.npz",receipt["context_sha256"]),allow_pickle=False) as f:
        assert np.array_equal(f["width_m"],previous) and np.array_equal(f["query_tick"],ticks)
        assert np.array_equal(f["consumed_start_tick"],np.asarray(ticks)-25) and f["command_valid"].all()
        assert np.array_equal(f["full_h40"],np.asarray(masks).all(axis=1))
    for key in ("robodojo","libero","auxiliary_slots","rates"):
        assert np.array_equal(bank[key],prior[key])
    a,b = (bank[key] for key in ("control_target","treatment_target"))
    assert a.shape == b.shape == (250,96) and a.dtype == b.dtype == np.int64
    assert np.array_equal(a[:,:94],b[:,:94])
    assert all(len(np.unique(row)) == 96 for row in np.concatenate((a,b)))
    assert np.array_equal(a[:,bank["auxiliary_slots"][0]],prior["target"][:,bank["auxiliary_slots"][0]])
    assert np.array_equal(a[:,:92],prior["target"][:,:92]) and np.array_equal(a[:,92:94],prior["target"][:,94:96])
    phases = Counter();control_episodes=set();early_counts=Counter()
    for i in range(250):
        shared=a[i,:94]
        assert (shared[:48]<4450).all() and ((shared[48:92]>=4450)&(shared[48:92]<7916)).all()
        assert ((shared[92:94]>=7916)&(shared[92:94]<8031)).all()
        for k in (94,95):
            control,treatment = int(a[i,k]),int(b[i,k])
            assert control<7916 and treatment>=8031
            ordinary,correction=row_meta[control],row_meta[treatment]
            assert ordinary[1:] == correction[1:] and ordinary[1] in ("PREGRASP","DESCEND","CLOSE","LIFT")
            assert np.asarray(masks[control]).all() and np.asarray(masks[treatment]).all()
            assert correction[0] == receipt["actual_early_starts"][k-94]
            phases[ordinary[1]]+=1;early_counts[correction[0]]+=1;control_episodes.add(ordinary[0])
    output=Path(mkdtemp(prefix="v7-c1-bank-readback.",dir=work))/"ACTUAL.json"
    value=dict(status="ALL_250_C1_PAIRED_ROWS_AND_8245_CONTEXT_WINDOWS_INDEPENDENTLY_READ_BACK",
        prepared_receipt_sha256=sha(prepared/"ACTUAL.json"),bank_sha256=sha(prepared/"BANK.npz"),
        context_sha256=sha(prepared/"PREVIOUS_COMMAND.npz"),common_slots=94,contrast_slots=2,steps=250,
        common_rows_original_exact=True,source_rows_and_learning_rates_original_exact=True,
        auxiliary_original_rows_exact=True,all_batches_unique=True,actual_context_windows=8245,
        actual_ordinary_control_episodes=sorted(control_episodes),actual_early_slot_visits=dict(early_counts),
        contrast_phase_visits=dict(phases),phase_scene_task_effective_H40_matched=True,
        two_resets_same_scene_family=True,new_independent_layouts=0,
        full_combined_training_reader_qualified=False,training_accesses=0,model_forwards=0,optimizer_updates=0,native_integrations=0)
    output.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    print(output,sha(output),flush=True)


if __name__ == "__main__":
    run(Path(sys.argv[1]))
