import pytest
from scripts.sla.diagnose_s16_sampling_v71 import admission,OPERATION,CONFIGURATIONS


def test_diagnostic_cannot_admit_training_native_or_unregistered_sampling():
    spec=dict(operation=OPERATION,scope='SLA_FIXED_CHECKPOINT_DIAGNOSTICS_ONLY',physics=False,
        native_execution_admitted=False,planned_generations=160,optimizer_updates=0,
        configurations=[list(v) for v in CONFIGURATIONS])
    admission(spec)
    for change in (dict(optimizer_updates=1),dict(physics=True),dict(native_execution_admitted=True),
                   dict(configurations=[[500,80]]),dict(planned_generations=161)):
        with pytest.raises(PermissionError):admission(dict(spec,**change))
