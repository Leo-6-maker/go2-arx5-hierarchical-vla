"""WM02 S-domain thin adapter reuses the pinned WM01 complete native seam.

No target backend is constructed here. Missing MPC/controller exporter and
current target admission remain explicit P1 gaps, never replaced by qpos/qvel.
"""
from .native import NativeAdapter
from .transitions import TransitionRecorder
from .storage import digest,dump_private


class SourceAdapter(NativeAdapter):
    domain='S_FIXED_BASE_DIAGNOSTIC'

    def advance_recorded(self,plan,index,*,run_id,actor_version,restore_ref):
        row=self.advance_atom(plan,index,raw=plan['action_raw'][index])
        row.update(run_id=run_id,actor_version=actor_version,reference_compiled=plan['rows10'],
            control_restore_ref=restore_ref,valid_state_target=True,
            reward_train=None,native_success_certificate=False,domain=self.domain,
            installed_H40=plan['rows10'],native_progress=dict(point=index,forecast_index=self.forecast_index),
            preview_condition_only=True)
        from .transitions import validate_record
        validate_record(row,native_dt=.002)
        return row

    def save_complete(self,path):
        dump_private(path,self.capture())
        return dict(path=str(path),sha256=digest(path),schema='WM01_NATIVE_RESTORE_V1',
                    domain=self.domain,complete_native_and_controller=True)
