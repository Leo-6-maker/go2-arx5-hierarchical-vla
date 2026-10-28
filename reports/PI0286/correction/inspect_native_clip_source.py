import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts/pi0276_execution'))
from coordinator import remote
print(remote('core', '''from pathlib import Path
for name in ('gr00t/data/utils.py','gr00t/data/state_action/state_action_processor.py'):
 p=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0280/official')/name
 lines=p.read_text().splitlines();print(str(p))
 selected=set()
 for i,l in enumerate(lines):
  if 'clip(' in l or 'def apply_action' in l or 'def unapply_action' in l or 'astype' in l or 'def normalize' in l or 'def unnormalize' in l:
   selected.update(range(max(0,i-5),min(len(lines),i+15)))
 for i in sorted(selected):print(f'{i+1}: {lines[i]}')
'''))
