import base64,hashlib,json,sys
from pathlib import Path
here=Path(__file__).resolve().parent
sys.path.insert(0,str(here.parents[2]/'scripts/pi0276_execution'))
from coordinator import remote
s="import base64\nfrom pathlib import Path\nprint(base64.b64encode(Path('/home/liuyu-nfs/Projects/go2-arx5/PI0286/correction/ENCODED_LATE12_NATIVE_CLIP_EVIDENCE.json').read_bytes()).decode())\n"
b=base64.b64decode(remote('core',s).strip());assert hashlib.sha256(b).hexdigest()=='d19f3c2647af2e46705baf64a9b262e1f02cce445e4ac9df41f029a254adbf51'
p=here/'ENCODED_LATE12_NATIVE_CLIP_EVIDENCE.json'
if p.exists():assert p.read_bytes()==b
else:p.write_bytes(b)
d=json.loads(b);w=d['window_details'][0]
print(json.dumps(dict(status=d['status'],parameter_dtypes=d['statistics_parameter_dtypes'],
  first_window={k:w[k] for k in ('slot','target','state','preclip','encoded','encoded_active10','decoded','actual_clip_calls','actual_processor_outputs','roundtrip_errors')},
  all_roundtrip_errors=[dict(slot=x['slot'],errors=x['roundtrip_errors']) for x in d['window_details']]),indent=2))
