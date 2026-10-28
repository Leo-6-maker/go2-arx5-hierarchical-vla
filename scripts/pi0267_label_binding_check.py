"""Zero-model regression: exact runtime binding must never accept altered inputs."""
import hashlib
from pathlib import Path
from pi0267_training_audit import read,rows,runtime_labels,check_input

root=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0267')
labels,_=runtime_labels(root,lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest())
actual=rows(root/'R/main/training/sample_audit.jsonl')[4]
expected=next(r for r in read(root/'FULL_TRAIN_INPUT_PLAN.json')['rows'] if (r['arm'],r['unit'],r['micro'])==('R',3,0))
check_input(actual,expected,labels)
for changed,binding in ((dict(actual,labels_masks_sha256='corrupt'),labels),
                        (dict(actual,condition_sha256='corrupt'),labels),(actual,{})):
    try:check_input(changed,expected,binding)
    except AssertionError:pass
    else:raise AssertionError('ALTERED_INPUT_OR_ORIGINAL_STRICT_FAILURE_NOT_REJECTED')
print('PASS_EXACT_RUNTIME_LABEL_BINDING_REJECTS_CORRUPTION_PRESERVES_STRICT_FAILURE')
