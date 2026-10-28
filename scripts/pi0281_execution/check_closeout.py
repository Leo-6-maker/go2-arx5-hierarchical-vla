import json
from pathlib import Path
r=Path(__file__).resolve().parents[2]/'reports/PI0281'
v=json.loads((r/'final/FINAL_RESULT.json').read_text())
assert len(v['source'])==12 and len(v['wbc'])==6
assert sum(x['quality'] for x in v['source'] if x['lane']=='EXPERT_DROID_BRIDGE_ROUNDTRIP')==4
assert not any(v['wbc_eligible'].values())
assert all(x['quality'] is None for x in v['target_expert']+v['target_recorded_model'])
assert v['accounting']['effective_totals']['technical_replacements']==3
assert v['accounting']['effective_totals']['physical_starts']==23
assert v['training']==v['target_policy_calls']==v['backward']==0
assert v['selected']==v['deployment']=='NONE'
print('PASS_FIXED_PANEL_AND_CONDITIONAL_STOP_CONSISTENCY')
