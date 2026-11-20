"""Pre-registered timing supervision for a separate ADAPT2000 experiment."""
import numpy as np

from .adapt5k_contract import phase_bucket
from .adapt5k_training import AdaptEngine
from .scale20k_training import target_auxiliary


PLAN = '1b2a25b7c1720051a2627d30831fb18130462374'
PARENT = 2000
ENDPOINT = 2500
ARMS = ('CLOCK-A-original-width','CLOCK-B-event-weighted-width')


def width_weights(label, consumed, previous, phase):
    """Classify TRAIN labels, then weight only their pre-registered H40 points."""
    label = np.asarray(label)
    consumed = np.asarray(consumed)
    if (label.shape!=(40,10) or consumed.shape!=(40,) or consumed.dtype!=np.bool_
            or not np.isfinite(label).all() or np.any((label[:,9]<0)|(label[:,9]>1))
            or not np.isfinite(previous) or not .002-1e-10<=previous<=.042+1e-10):
        raise ValueError('CLOCK500_REQUIRES_ORIGINAL_FINITE_FULL_H40_AND_PREVIOUS_COMMAND')
    weight = np.ones(40,dtype=np.float32)
    bucket = phase_bucket(label,consumed,previous,phase)
    if bucket=='P10_DUE_CLOSE':
        timeline = np.r_[previous,.002+.040*label[:,9]]
        crossing = np.flatnonzero((timeline[:-1]>=.022)&(timeline[1:]<.022))
        event = int(crossing[0])
        weight[max(0,event-2):min(40,event+3)] = 4.
    elif bucket=='NOT_DUE_CLOSE':
        weight[:10] = 4.
    return weight


def event_weighted_width_loss(predicted, expected, weight, *, xp):
    if predicted.shape!=expected.shape or predicted.shape!=weight.shape or predicted.shape[-1]!=40:
        raise ValueError('CLOCK500_WIDTH_TERM_REQUIRES_ALL_40_POINTS_PER_SAMPLE')
    error = ((predicted-expected)/.040)**2
    return xp.mean(xp.sum(weight*error,axis=-1)/xp.sum(weight,axis=-1))


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
