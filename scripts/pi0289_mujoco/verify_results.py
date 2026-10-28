"""Independent local CPU arithmetic on read-back arrays; no neural/physics query."""
import json
import math
from pathlib import Path
import runpy
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from go2arx5_vla.target_mujoco.reference import score_reference

def main():
    out=ROOT/'reports/PI0289/execution'
    p=out/'evidence/a7ff3a9a26c34aab9843fa5c4abecef3/cpu'
    source=json.loads((p/'SCORE_INPUT.json').read_text())
    rows=[json.loads(x) for x in (p/'NATIVE_FEEDBACK_LAYERS.jsonl').read_text().splitlines()]
    result=score_reference(source['condition'],source['initial_pose'],rows,issued=source['issued'],accepted=source['accepted'],common_future=source['common_future'],runtime_evidence=source['runtime_evidence'])
    expected=json.loads((p/'NATIVE_FEEDBACK_SCORE.json').read_text())
    assert result==expected and result['complete_10s'] and not result['pass']
    independent={}
    for name in ['requested_pose','installed_pose']:
        errors=[]
        for r in rows:
            if r['t']<8:continue
            a=np.array(r[name]);b=np.array(r['achieved_pose'])
            xyz=float(np.linalg.norm(a[:3]-b[:3]))
            cosine=abs(float(np.dot(a[3:]/np.linalg.norm(a[3:]),b[3:]/np.linalg.norm(b[3:]))))
            errors.append((xyz,2*math.acos(min(1.,cosine))))
        rms=np.sqrt(np.mean(np.square(errors),axis=0))
        assert abs(rms[0]-result['errors'][name]['last2_position_m']['RMS'])<1e-10
        assert abs(rms[1]-result['errors'][name]['last2_SO3_rad']['RMS'])<1e-10
        assert rms[0]<.015 and rms[1]>.05
        independent[name]=dict(XYZ_RMS_m=float(rms[0]),SO3_RMS_rad=float(rms[1]))
    clock_dir=out/'evidence/7bdc332eae2843c0aa3edeffe4a52fde/cpu'
    clock=json.loads((clock_dir/'CLOCK_PAIRS.json').read_text())
    lifecycle=json.loads((clock_dir/'LIFECYCLE.json').read_text())
    clock['reset_isolation_checked']=any(x.get('detail')=='WRONG_EPISODE' for x in lifecycle['events'])
    clock['expired_reference_hold_checked']=any(x['reason']=='REFERENCE_EXPIRED' for x in lifecycle['events'])
    clock_check=runpy.run_path(str(ROOT/'scripts/pi0288_mujoco/doctor.py'))['check_clock'](clock)
    assert clock_check['status']=='BLOCKED_CLOCK_OR_RESET'
    (out/'CLOCK_CONTRACT.json').write_text(json.dumps(dict(status='BLOCKED_CLOCK_OR_RESET',actual_validator=clock_check,negative_episode='7bdc332eae2843c0aa3edeffe4a52fde',negative_tests='PASS',scope='No old epoch offset reused; receiver clock pairs exceed original25ms ceiling'),indent=2)+'\n')
    receipt=dict(status='PASS_INDEPENDENT_CPU_REPRODUCES_VALID_HOLD_FAIL',sample_count=len(rows),complete_10s=True,independent_quaternion_dot_metrics=independent,clock_gate=clock_check['status'],physics_integrations=0,neural_forwards=0,optimizer_updates=0)
    (out/'FINAL_CPU_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))

if __name__=='__main__':main()
