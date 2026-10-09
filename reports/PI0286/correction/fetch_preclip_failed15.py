import base64
import hashlib
import json
import sys
from pathlib import Path
here=Path(__file__).resolve().parent
sys.path.insert(0,str(here.parents[2]/'scripts/pi0276_execution'))
from coordinator import remote
s="import base64\nfrom pathlib import Path\nprint(base64.b64encode(Path('/home/liuyu-nfs/Projects/go2-arx5/PI0286/correction/PRECLIP_FAILED15_DETAILS.json').read_bytes()).decode())\n"
b=base64.b64decode(remote('core',s).strip())
assert hashlib.sha256(b).hexdigest()=='525c5fbd2babe9312e837e4a0ef0d9ef8c63e265219595640094e792b0258330'
path=here/'PRECLIP_FAILED15_DETAILS.json'
if path.exists():assert path.read_bytes()==b
else:path.write_bytes(b)
d=json.loads(b)
rows=[]
for w in d['window_details']:
    for c in w['channels']:
        if c['out_of_range_count']:
            rows.append(dict(slot=w['slot'],anchor=w['anchor'],count=w['sealed_count'],
                             target_dtype=w['target_dtype'],preclip_dtype=w['preclip_dtype'],**c))
print(json.dumps(dict(status=d['status'],rows=rows,normalization_source=d['normalization_source']),indent=2))
