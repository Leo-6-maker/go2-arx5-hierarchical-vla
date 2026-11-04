"""Read-only S-controller reference observer; never a plant or controller replacement.

Mirrors only the existing deterministic finger-target slew from accepted commands.
Measured finger position is not the internal target. Unknown reset state remains
an interval; no actual/future controller state is used to collapse that interval.
"""
import numpy as np

class SourceFingerTarget:
    def __init__(self,*,known_reset=False):
        self.travel=.044;self.dt=.002;self.speed=.04
        self.lower=np.full(2,self.travel if known_reset else 0.)
        self.upper=np.full(2,self.travel)
    def consume(self,opening):
        if not np.isfinite(opening)or not 0<=opening<=1:raise ValueError('Physical source command fraction0..1 required')
        desired=np.full(2,float(opening)*self.travel);slew=self.speed*self.dt
        self.lower+=np.clip(desired-self.lower,-slew,slew)
        self.upper+=np.clip(desired-self.upper,-slew,slew)
        return self.lower.copy(),self.upper.copy()
    def interval(self):return self.lower.copy(),self.upper.copy()
