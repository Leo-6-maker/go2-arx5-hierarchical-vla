"""Finite accounting/import/card check without any neural or optimizer call."""
import json,sys
from pathlib import Path

def check(root):
    root=Path(root);sys.path.append(str(root.parent/'PI0283/code'))
    from authority import load
    from phase import validate,old_helpers
    from train_arm import completed_kind
    limits=load(root)
    cards=[validate(root,root/'design'/(arm+'_CARD.json')) for arm in ('CTRL2048','LR5X2048')]
    difference={k for k in cards[0] if cards[0][k]!=cards[1][k]}
    assert difference=={'stage','peak_LRs'} and cards[0]['peak_LRs'][0]==cards[1]['peak_LRs'][0]
    assert old_helpers().restore_optimizer.__module__=='pi0283_phase_observers'
    for kind,phase,cap in [('formal','main',4096),('canary','continuous',32),('memory','memory',16),('technical','main',64)]:
        rows=[dict(arm='CTRL2048',phase=phase,step=i+1,work_kind=kind) for i in range(cap)]
        step=1 if kind=='technical' else cap+1
        assert completed_kind(rows[:-1],'CTRL2048',phase,step,limits)==kind
        try:completed_kind(rows,'CTRL2048',phase,step,limits)
        except AssertionError:pass
        else:raise AssertionError('Completed-work cap not enforced: '+kind)
    plan=[json.loads(x) for x in Path(cards[0]['plan_path']).read_text().splitlines()]
    assert len(plan)==16384 and [v['index'] for v in plan]==list(range(16384)) and all(v['split']=='TRAIN' for v in plan)
    result=dict(status='PASS_CPU_RUNTIME_CARDS_SAME_PARENT_SINGLE_LR_DIFFERENCE_PLAN_AND_FINITE_CAPS',formal=4096,canary=32,memory=16,technical=64,all_updates=4208,neural_calls=0,optimizer_updates=0,physics=0)
    (root/'runtime/CPU_TRAINING_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':check(sys.argv[1])
