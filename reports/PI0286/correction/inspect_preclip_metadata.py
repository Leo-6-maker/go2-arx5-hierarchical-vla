import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts/pi0276_execution'))
from coordinator import remote
print(remote('core', '''import json
from pathlib import Path
r=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0286')
d=json.loads((r/'correction/CORRECTION_BANK.json').read_text())
print(list(d))
print((r/'code/environment.sh').read_text())
print(json.dumps([{k:v for k,v in e.items() if k not in ('rows','restore')} for e in d['all_requested_slot_reports'] if e['status'].startswith('FAIL_SUPERVISION')],default=str)[:1200])
'''))
