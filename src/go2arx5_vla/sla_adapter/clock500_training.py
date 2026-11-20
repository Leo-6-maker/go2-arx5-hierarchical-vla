"""Keep the CLOCK training runtime separate from cold native layout tooling."""
from .adapt5k_training import AdaptEngine
from .clock500 import ARMS, ENDPOINT, PARENT, width_weights
from .scale20k_training import target_auxiliary


class ClockEngine(AdaptEngine):
    """Reuse the original ADAPT update with an opt-in width term for arm B."""

    def __init__(self, *args, clock_arm, phase_step, stage_start, stage_stop, **kwargs):
        if (clock_arm not in ARMS or stage_start!=PARENT or stage_stop!=5000
                or not PARENT<=phase_step<ENDPOINT):
            raise ValueError('CLOCK500_REQUIRES_EXACT_D2_CURSOR_AND_REGISTERED_ARM')
        self.clock_arm = clock_arm
        super().__init__(*args,phase_step=phase_step,stage_start=stage_start,stage_stop=stage_stop,**kwargs)

    def _target_auxiliary(self, graph, quantiles):
        return target_auxiliary(graph,quantiles,event_weighted=self.clock_arm==ARMS[1])

    def _training_only_auxiliary_inputs(self, dataset, index, previous):
        if self.clock_arm==ARMS[0]:
            return ()
        episode,anchor = dataset.rows[index]
        rows = dataset.labels[episode['id']]
        if not rows['supervision_mask'][anchor].all():
            raise ValueError('CLOCK500_REQUIRES_ORIGINAL_FULL_H40_SUPERVISION')
        weights = width_weights(rows['nominal_policy_H40'][anchor],rows['consumed_mask'][anchor],
                                previous,str(rows['phase'][anchor]))
        return (weights[None,:],)

    def update(self, guard):
        if self.step>=ENDPOINT:
            raise ValueError('CLOCK500_EXACT_500_UPDATE_MAXIMUM_REACHED')
        return super().update(guard)
