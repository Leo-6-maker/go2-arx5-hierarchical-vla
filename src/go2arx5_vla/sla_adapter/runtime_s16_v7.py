"""S16 uses the unchanged V6 reader and FM32 learner with a fixed visit bank."""
import hashlib
import numpy as np

from .runtime_reference_v6 import ReferenceTargetEngine
from .runtime_v1 import read_pin
from .config_v1 import learning_rates
from .s16_offline_v7 import OPERATION, fixed_indices, visit_bank


class S16OfflineEngine(ReferenceTargetEngine):
    def visits_for_dataset(self, dataset):
        self.selected_indices = fixed_indices(dataset.manifest, read_pin(self.spec['S16_selection']))
        bank = np.asarray(visit_bank(self.selected_indices), dtype='<i8')
        self.visit_bank_binding = dict(sampling='EACH_FIXED_S16_WINDOW_TWICE_PER_EFFECTIVE_BATCH',
            selected_indices=list(self.selected_indices), planned_target_visits=16000,
            planned_source_visits=0, noise='EXISTING_ABSOLUTE_CURSOR_FLOW_NOISE',
            visit_map_sha256=hashlib.sha256(bank.tobytes()).hexdigest(), student_phase_input=False)
        return bank

    def learning_rate_values_for_spec(self, spec):
        return learning_rates()

    def __init__(self, spec, output):
        super().__init__(spec, output)
        self.bindings.update(recipe=OPERATION, S16_selection=spec['S16_selection']['sha256'],
            offline_fitting_only=True, native_execution_admitted=False,
            limited_fresh_named_N20_fitting=False, full_N20_TRAIN_statistics_retained=True)
