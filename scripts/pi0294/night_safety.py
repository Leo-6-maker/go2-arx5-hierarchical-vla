"""Read every saved native state; never clip, alter or forgive joint limits."""
import json
import struct
from pathlib import Path
import numpy as np
from .task_state import parse_state
from scripts.pi0292.durable import write


class JournalRangeGuard:
    def __init__(self, model, journal):
        self.model=model;self.journal=Path(journal);self.offset=0;self.checked=0
        self.joints=np.flatnonzero(model.jnt_limited)

    def check(self):
        import mujoco
        with self.journal.open('rb') as stream:
            stream.seek(self.offset)
            while True:
                start=stream.tell();header=stream.read(4)
                if len(header)!=4:break
                size,=struct.unpack('<I',header);raw=stream.read(size)
                if len(raw)!=size:break
                packet=parse_state(raw);self.offset=stream.tell();self.checked+=1
                for j in self.joints:
                    q=float(packet['qpos'][self.model.jnt_qposadr[j]])
                    lo,hi=self.model.jnt_range[j]
                    if not lo<=q<=hi:
                        proof=dict(status='RAW_NATIVE_MECHANICAL_RANGE_EXCESS',joint=mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_JOINT,int(j)),
                            q=q,range=[float(lo),float(hi)],step=packet['step'],time_s=packet['time_s'],
                            checked_packets=self.checked,original_native_observer_unchanged=True,limit_relaxation=0)
                        dest=self.journal.parent/'HIGH_RATE_RANGE_FAILURE.json'
                        if not dest.exists():write(dest,proof)
                        raise ValueError('NIGHT_RAW_NATIVE_RANGE_STOP:'+proof['joint'])
        return self.checked
