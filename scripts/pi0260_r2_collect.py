"""One finite R2 collection owner. Reuse completed results; never schedule retries."""
import fcntl
import os
from pathlib import Path
from pi0260_r2 import ROOT,admission,sha
from pi0255_motion3 import write
import pi0260_data as data

assert data.ROOT==ROOT and data.R2
lock=(ROOT/'state/collection_owner.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
admission()
write(ROOT/'state/collection_owner.json',dict(pid=os.getpid(),status='RUNNING',
    overlay_sha256=sha(ROOT/'design/overlay.json'),source_sha256=sha(__file__)))
for split in ('TRAIN','KIN-VAL'):data.run(split)
write(ROOT/'state/collection_owner.json',dict(pid=os.getpid(),status='COLLECTION_COMPLETE_REPLAY_CODEC_PENDING',
    overlay_sha256=sha(ROOT/'design/overlay.json'),source_sha256=sha(__file__),counters=data.counters()))
