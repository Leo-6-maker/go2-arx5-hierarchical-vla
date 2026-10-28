"""Adopted V2 progress core; native admission is enforced by its worker.

The unadopted draft remains immutable. The adopted algorithm is exactly the
draft's component-wise _advance; this wrapper versions plans and snapshots.
"""
import numpy as np
from .component_progress_draft import ComponentProgressDraft
from .persistent_approach_draft import DraftConfig,pose

VERSION='FULLBODY_COMPONENT_PROGRESS_QUALIFICATION_V2'


class ComponentProgress(ComponentProgressDraft):
    def plan(self,**kw):
        result=super().plan(**kw)
        result.update(version=VERSION,launch_allowed=False,
            labels='EXACT_SUBMITTED_PREFIX_AND_H40_ONLY_AFTER_AUTHORIZED_NATIVE_INSTALL_ACK',
            admission='NATIVE_WORKER_REQUIRED_CORE_CANNOT_LAUNCH')
        return result

    def snapshot(self):
        result=super().snapshot();result['version']=VERSION
        return result

    @classmethod
    def restore(cls,snapshot):
        if snapshot['version']!=VERSION or snapshot['launch_allowed'] is not False:
            raise ValueError('EXACT_ADOPTED_CORE_VERSION_REQUIRED')
        p=cls(task=snapshot['task'],actual_tcp=snapshot['initial'],stage_target=snapshot['target'],
            actual_BASE=snapshot['initial_base'],station_world=snapshot['station'],pad_radius_m=snapshot['pad_radius_m'],
            table_top_m=snapshot['table_top_m'])
        cfg=DraftConfig(**snapshot['config'])
        from dataclasses import replace
        if cfg not in (DraftConfig(),replace(DraftConfig(),linear_speed_m_s=.04)):
            raise ValueError('EXACT_ADOPTED_CORE_CONFIGURATION_REQUIRED')
        p.config=cfg;p.cursor=pose(snapshot['cursor']);p.route=[pose(v) for v in snapshot['route']]
        if len(p.route) not in (1,4) or not 0<=snapshot['segment']<=len(p.route):raise ValueError('ADOPTED_ROUTE_CURSOR')
        p.segment=snapshot['segment'];p.last_time=snapshot['last_time'];p.last_observation=snapshot['last_observation']
        p.history=[dict(r) for r in snapshot['history']];p.stop_reason=snapshot['stop_reason']
        p.previous_BASE_velocity=np.asarray(snapshot['previous_BASE_velocity'],dtype=float)
        if p.previous_BASE_velocity.shape!=(3,) or not np.isfinite(p.previous_BASE_velocity).all():raise ValueError('FINITE_RESTORED_BASE')
        return p
