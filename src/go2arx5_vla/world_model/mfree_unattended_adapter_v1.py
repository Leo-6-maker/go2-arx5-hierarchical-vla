"""Shared expert-family cancellation above the unchanged native controller."""
import json
import time
from pathlib import Path
from .target_sequential import TargetSequentialAdapter
from .mfree_unattended_ledger_v1 import Ledger


class IsolatedAdapter(TargetSequentialAdapter):
    def _mark_own_stop(self):
        if self.spec['mobile_condition']['method']!='EXPERT':return
        if getattr(self,'_shared_stop_marked',False):return
        stop=self.output/'MOBILE_SAFETY_STOP.private.json';physics=self.output/'physics.jsonl';reason=None
        if stop.exists():reason='ORIGINAL_MOBILE_SAFETY_STOP'
        if physics.exists():
            with physics.open('rb')as handle:
                handle.seek(max(0,physics.stat().st_size-32768));lines=handle.read().splitlines()
            for line in reversed(lines):
                try:value=json.loads(line)
                except json.JSONDecodeError:continue
                if value.get('kind')=='stop':reason=value.get('reason','ORIGINAL_NATIVE_STOP');break
        if reason:
            Ledger(self.spec['unattended_control']).isolate(self.spec['mobile_condition']['family'],
                dict(reason=reason,case=str(self.spec['registered_root']),source='EXPERT_NATIVE_SAFETY'))
            self._shared_stop_marked=True

    def guard(self):
        try:super().guard()
        except BaseException:self._mark_own_stop();raise
        if getattr(self,'isolation_teardown',False):return
        now=time.monotonic()
        if now>=getattr(self,'_shared_next_read',0):
            self._shared_families=json.loads((Path(self.spec['unattended_control'])/'ISOLATION.private.json').read_bytes())
            self._shared_next_read=now+.1
        if self.spec['mobile_condition']['family']in self._shared_families:
            raise RuntimeError('SHARED_EXPERT_FAMILY_SAFETY_ISOLATION_ABORT')

    def advance_to(self,*args,**kwargs):
        try:return super().advance_to(*args,**kwargs)
        except BaseException:self._mark_own_stop();raise
