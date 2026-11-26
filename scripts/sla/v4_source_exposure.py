"""Bind public source skill coverage to the actual sealed MAIN3.8K visits."""

import argparse
import ast
from bisect import bisect_right
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import struct


def source_reports(custody):
    from scripts.sla.hopper_inputs import inputs

    settings=inputs(custody/"assets.verified")
    main=custody/"HOPPER-ADAPT20K-V1/MAIN"
    endpoint=main/"checkpoints/phase03800/COMPLETE.json"
    marker=json.loads(endpoint.read_bytes())
    steps=marker["progress"]["phase_step"]
    assert steps==3800
    updates=[json.loads((main/"updates"/f"phase{i:05d}.json").read_bytes()) for i in range(1,steps+1)]
    result={}
    for domain in ("robodojo","libero"):
        pin=settings["data"][domain]["manifest"]
        raw=Path(pin["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==pin["sha256"]
        catalogue=json.loads(raw)
        source,ready,width=("episodes","valid","windows") if domain=="robodojo" else ("rows","admitted","window_count")
        rows=[r for r in catalogue[source] if r["split"]=="TRAIN" and r[ready]]
        ends=list(itertools.accumulate(r[width] for r in rows))
        bank=(custody/"banks.verified"/(domain+".npy")).read_bytes()
        assert bank[:8]==b"\x93NUMPY\x01\x00"
        offset=10+int.from_bytes(bank[8:10],"little")
        header=ast.literal_eval(bank[10:offset].decode("ascii"))
        assert header["descr"]=="<i8" and len(header["shape"])==1
        visits=Counter()
        for i,row in enumerate(updates):
            record=row["result"]
            batch=bank[offset+i*32*8:offset+(i+1)*32*8]
            assert len(batch)==32*8 and row["status"]=="COMPLETE" and record["phase_step"]==i+1
            assert hashlib.sha256(batch).hexdigest()==record["domains"][domain]["visit_slice_sha256"]
            for (index,) in struct.iter_unpack("<q",batch):
                assert 0<=index<ends[-1]
                visits[rows[bisect_right(ends,index)]["task"]]+=1
        result[domain]=dict(train_episodes=len(rows),train_windows=ends[-1],
            verified_visits=sum(visits.values()),visits_by_task=visits,manifest_sha256=pin["sha256"])
    target=custody/"foundation-layout-v2/train-ready/TRAIN-069/MANIFEST.private.json"
    target_manifest=json.loads(target.read_bytes())
    return dict(verified_original_updates=steps,main_checkpoint_sha256=hashlib.sha256(endpoint.read_bytes()).hexdigest(),
        sources=result,target_train069_counts=Counter(r["task"] for r in target_manifest["episodes"]),
        new_updates=0,new_integrations=0,new_model_forwards=0)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("custody",type=Path)
    print(json.dumps(source_reports(parser.parse_args().custody),indent=2,allow_nan=False))
