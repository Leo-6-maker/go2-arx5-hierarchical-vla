"""Declared finite precontact connector; immutable skill and original gates."""
import numpy as np


def package(selected, design):
    if (design['schema']!='TFNR1_PRECONTACT_ENTRY_V1'
            or design['source_id']!=selected['source_id']
            or design['source_program_sha256']!=selected['source_program_sha256']
            or design['distance_m'] not in (.01,.02,.04)):
        raise ValueError('FROZEN_PRECONTACT_DESIGN_BINDING_REQUIRED')
    direction=np.asarray(design['source_approach_unit_world'],dtype=float)
    if direction.shape!=(3,) or not np.isfinite(direction).all() or abs(np.linalg.norm(direction)-1)>1e-10:
        raise ValueError('SOURCE_APPROACH_DIRECTION_REQUIRED')
    pre=np.asarray(selected['contact']).copy()
    pre[:3,3]-=design['distance_m']*direction
    return dict(selected,approach=pre,direct_join=False,entry_contact_stage='contact',
        entry_design_sha256=design['design_sha256'])


def phases(entry, response, certificate):
    links=response['links']
    if len(links)!=2 or any(x['status']!='Success' for x in links):
        raise ValueError('PAIRED_APPROACH_CONTACT_ENTRY_REQUIRED')
    boundary=float(links[0]['time'][-1])*certificate['uniform_time_scale']
    if not 0<boundary<float(entry['time'][-1]):
        raise ValueError('ENTRY_PHASE_BOUNDARY_REQUIRED')
    return dict(entry,free_approach_end_s=boundary,contact_connector_end_s=float(entry['time'][-1]))


def stage(entry, at, task):
    if at<entry.get('free_approach_end_s',entry['time'][-1])-1e-12:
        return 'approach' if task=='T1' else 'contact'
    return 'contact'
