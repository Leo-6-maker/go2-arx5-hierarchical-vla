"""Declared post-query HOLD or small XY reference interventions, never WBC edits.

The raw generated H40 remains intent. HOLD deliberately does not execute it;
XY interventions preserve every desired knot and modify only the reference.
Both still require the original per-install native safety/consumption checks.
"""
from dataclasses import dataclass,asdict
import json
from pathlib import Path
import numpy as np

SCHEMA='WM02_POST_QUERY_REFERENCE_INTERVENTION_V1'
OFFSETS=dict(NOMINAL=(0.,0.),X_PLUS_050=(.0005,0.),X_MINUS_050=(-.0005,0.),
             Y_PLUS_050=(0.,.0005),Y_MINUS_050=(0.,-.0005),
             X_PLUS_025=(.00025,0.),X_MINUS_025=(-.00025,0.),
             Y_PLUS_025=(0.,.00025),Y_MINUS_025=(0.,-.00025),HOLD=(0.,0.))


@dataclass(frozen=True)
class Intervention:
    schema:str
    candidate:str
    offset_target_xy_m:tuple
    ramp_atoms:int

    def document(self):
        value=asdict(self);value['offset_target_xy_m']=list(self.offset_target_xy_m)
        return value


def parse(value):
    if isinstance(value,Intervention):value=value.document()
    if (not isinstance(value,dict)or set(value)!={'schema','candidate','offset_target_xy_m','ramp_atoms'}
            or value.get('schema')!=SCHEMA or value.get('candidate')not in OFFSETS
            or type(value.get('ramp_atoms'))is not int or value['ramp_atoms']!=16
            or not np.array_equal(value.get('offset_target_xy_m'),OFFSETS[value['candidate']])):
        raise ValueError('EXPLICIT_HOLD_OR_REGISTERED_050MM_XY_REFERENCE_NO_CLIP_REQUIRED')
    return Intervention(SCHEMA,value['candidate'],OFFSETS[value['candidate']],16)


def apply(reference_path,intervention):
    """Return a new whole reference path; neither desired poses nor dt change."""
    intervention=parse(intervention);path=np.asarray(reference_path,float).copy()
    if intervention.candidate=='HOLD':return np.repeat(path[:1],len(path),axis=0)
    if any(intervention.offset_target_xy_m):
        u=np.minimum(np.arange(len(path),dtype=float)/intervention.ramp_atoms,1.)
        weight=u**3*(10+u*(-15+6*u))
        path[:,:2]+=weight[:,None]*np.asarray(intervention.offset_target_xy_m)
    return path


def validate_scope(spec):
    """Bind a new explicit panel row; this supplies no physical permission."""
    config=spec.get('program_reference_intervention')
    if config is None:return None
    from .fusion_live_program_replay_v1 import assignment,PHASE
    from .storage import digest
    intervention=parse(config);a=assignment(spec)
    if (spec.get('phase')!=PHASE or a['split']not in ('TRAIN','DEV','DIAGNOSTIC')
            or spec.get('task_condition')!='BOUNDED_FULL_H40_MECHANICAL_TRACKING_AT34303_V1'
            or spec.get('live_skill_geometry_scale')!=1.
            or spec.get('live_skill_time_stretch_factor')not in (1,2)):
        raise ValueError('FRESH_REGISTERED_POST_QUERY_REFERENCE_SCOPE_REQUIRED')
    path=Path(spec['response_learning_panel'])
    if (path.is_symlink()or digest(path)!=a['panel_sha256']
            or a['panel_sha256']!=spec['response_learning_panel_sha256']):
        raise ValueError('EXACT_PRE_EFFECT_INTERVENTION_PANEL_REQUIRED')
    panel=json.loads(path.read_text())
    if panel.get('domain',{}).get('time_stretch_factor')!=spec['live_skill_time_stretch_factor']:
        raise ValueError('EXACT_SHARED_REGISTERED_WHOLE_PROGRAM_TIMING_REQUIRED')
    if not any(row.get('native_run')==Path(spec['registered_root']).name
               and row.get('parent')==a['parent']and row.get('split')==a['split']
               and row.get('reference_intervention')==intervention.document()
               for row in panel['assignments']):
        raise ValueError('ORIGINAL_PANEL_ROW_MUST_DECLARE_ACTUAL_INTERVENTION')
    return intervention
