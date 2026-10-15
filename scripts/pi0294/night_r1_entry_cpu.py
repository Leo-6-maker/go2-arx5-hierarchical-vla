"""Finite CPU checks for the source-bound C2 entry, no simulator execution."""
import hashlib,json
import numpy as np
from .night_r1_entry import package,phases,stage
from .night_r1_teacher import Teacher
from .mp_client import contact_permitted


def check():
    pose=np.eye(4);pose[:3,3]=[.45,-.08,.494]
    original=dict(index=0,task='T1',source_id='FIXTURE',source_program_sha256='FIXTURE',contact=pose,approach=pose,direct_join=True)
    design=dict(schema='TFNR1_PRECONTACT_ENTRY_V1',source_id='FIXTURE',source_program_sha256='FIXTURE',distance_m=.04,
        source_approach_unit_world=[0,0,-1],design_sha256='FIXTURE')
    out=package(original,design)
    assert np.array_equal(out['contact'],pose) and np.array_equal(original['approach'],pose)
    assert abs(out['approach'][2,3]-.534)<1e-12 and out['entry_contact_stage']=='contact'
    entry=phases(dict(time=np.array([0.,2.,4.]),poses=np.array([pose]*3)),
        dict(links=[dict(status='Success',time=[0.,1.]),dict(status='Success',time=[0.,1.])]),dict(uniform_time_scale=2.))
    assert stage(entry,0.,'T1')=='approach' and stage(entry,1.999,'T1')=='approach' and stage(entry,2.,'T1')=='contact'
    t=Teacher.__new__(Teacher);t.entry=entry;t.card={'task':'T1'};t.record={'events':[{'stage':'close','program_time_s':0.},{'stage':'lift','program_time_s':1.}]}
    assert [t.stage_at(x) for x in (0,2,4,5)]==['approach','contact','close','lift']
    t.entry={'time':np.array([0.,4.])}
    assert [t.stage_at(x) for x in (0,2,4,5)]==['approach','approach','close','lift']
    for field,value in [('source_id','WRONG'),('source_program_sha256','WRONG'),('distance_m',.03),('source_approach_unit_world',[0,0,-2])]:
        try:package(original,dict(design,**{field:value}))
        except ValueError:pass
        else:raise AssertionError('ENTRY_BINDING_NEGATIVE_NOT_REJECTED:'+field)
    for phase in ('approach','align'):
        assert not contact_permitted(['task_right_pad','task_cube_geom'],-.0001,phase)
    assert contact_permitted(['task_right_pad','task_cube_geom'],-.0001,'contact')
    assert not contact_permitted(['task_right_pad','task_cube_geom'],-.0041,'contact')
    assert not contact_permitted(['base_link','task_table'],-.0001,'contact')
    return dict(status='PASS_SOURCE_BOUND_PAIRED_ENTRY_CPU',source_endpoint_unchanged=True,legacy_stage_regression=True,
        source_event_order_unchanged=True,original_contact_predicates_unchanged=True,negative_bindings_rejected=True,
        new_physics=0,new_IK=0,new_RRT=0,new_render=0,new_neural_queries=0)
