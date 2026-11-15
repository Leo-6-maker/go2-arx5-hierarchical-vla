"""Observed grasp qualification and a separately named, event-triggered BASE route."""
import numpy as np

from .mobile_wbc_reference_v1 import m0_reference

VERSION='MOBILE_OBSERVED_AFTER_GRASP_V2'
MODE='M_ASSIST_AFTER_GRASP_V2'


class ObservedGrasp:
    def __init__(self,rest_z):
        self.rest_z=float(rest_z);self.ever_grasp=False;self.since=None
        self.previous_step=None;self.qualified=False

    def observe(self,sample):
        step=sample['native_step'];self.ever_grasp|=bool(sample['native_grasp'])
        contiguous=self.previous_step is None or step-self.previous_step==25
        holding=bool(sample['native_grasp'])and sample['object_pose'][2,3]-self.rest_z>=.020
        if not holding:self.since=None
        elif self.since is None or not contiguous:self.since=step
        self.qualified=bool(holding and self.since is not None and (step-self.since)*.002>=1.)
        self.previous_step=step
        return self.qualified


class AfterGraspRoute:
    def __init__(self,axis):
        self.axis=axis;self.start=None;self.mount=None;self.tcp=None

    def trigger(self,sample,mount,qualified):
        if self.start is None and qualified:
            self.start=sample['native_step'];self.mount=np.asarray(mount).copy();self.tcp=sample['tcp'].copy()
            return True
        return False

    def preview(self,native_step):
        if self.start is None:return 'M_FREE',None
        if native_step<self.start or(native_step-self.start)%25:
            raise ValueError('ACTUAL_AFTER_GRASP_ROUTE_CLOCK')
        _,base=m0_reference(self.tcp,self.mount,self.axis,'WORLD_HOLD',(native_step-self.start)//25)
        return 'M_ASSIST',base

    def evidence(self):
        return None if self.start is None else dict(native_step=self.start,
            basis='ACTUAL_NATIVE_GRASP_LIFT20MM_ONE_SECOND',external_BASE_only=True,
            origin='MEASURED_MPC_MOUNT_AT_TRIGGER',initial_mount=self.mount.tolist(),
            actual_TCP_at_trigger=self.tcp.tolist(),base_preview_before_trigger=False)
