"""Finite R1 reference contracts; synthetic fixtures never become a task seed."""
import numpy as np
from .night_r1_reference import ExecutedReferenceHistory,local_quantized_candidates,stencils


def contracts():
    rows=[]
    def good(name,fn):fn();rows.append(dict(name=name,status='PASS_FIXTURE_ONLY'))
    def bad(name,fn):
        try:fn()
        except (ValueError,KeyError):rows.append(dict(name=name,status='PASS_EXPECTED_REJECTION'))
        else:raise AssertionError(name)
    t=np.arange(1001)*.002;q=np.zeros((1001,6));q[:,2]=.01*np.sin(t)
    def timing():
        value=local_quantized_candidates(q,t)
        assert len(value['attempts'])==4 and value['selected'] is not None
        assert not any(x.get('physics_qualification') for x in value['attempts'])
    good('FOUR_FIXED_TIMING_FACTORS',timing)
    bad('NAN_TIME',lambda:local_quantized_candidates(q,np.full(1001,np.nan)))
    bad('UNAUTHORIZED_GRID',lambda:local_quantized_candidates(q,t,grid=.025))
    bad('OUTSIDE_DURATION_SET',lambda:local_quantized_candidates(q,t,max_duration=256.))
    poses=np.repeat(np.eye(4)[None],40,axis=0);witness=np.zeros((40,6));src=np.arange(40)*.025
    h=ExecutedReferenceHistory()
    bad('NO_PREVIEW_AS_EXECUTED_HISTORY',h.last_two)
    h.propose('reject',poses,witness,src,native_start_time=10.);h.reject('reject')
    bad('REJECT_CANNOT_BE_EXECUTED',lambda:h.acknowledge_executed('reject',0,5000,10.05))
    h.propose('a',poses,witness,src,native_start_time=10.)
    bad('ACK_ID_MISMATCH',lambda:h.acknowledge_install('a','other'))
    h.acknowledge_install('a','a')
    bad('SOURCE_CLOCK_NOT_NATIVE_CLOCK',lambda:h.acknowledge_executed('a',0,5025,.05))
    h.acknowledge_executed('a',6,5175,10.35);h.acknowledge_executed('a',7,5200,10.4)
    def clocks():
        assert abs(h.last_two()[-1]['source_time']-.175)<1e-12
        assert abs(h.last_two()[-1]['actual_end_time']-10.4)<1e-12
    good('NATIVE_AND_SOURCE_CLOCKS_SEPARATE',clocks)
    bad('REPLAYED_PREFIX',lambda:h.acknowledge_executed('a',7,5225,10.4))
    bad('UNEXECUTED_PREVIEW_TAIL',lambda:h.acknowledge_executed('a',8,5225,10.45))
    bad('GAP_AT_CHUNK_JOIN',lambda:h.check_next(witness,native_start_time=10.45))
    def seam():
        assert h.check_next(witness,native_start_time=10.4)['max_acceleration']==0
        jumped=witness.copy();jumped[:,2]=.1
        proof=h.check_next(jumped,native_start_time=10.4)
        assert proof['max_acceleration']>1 and any('H40_SUBMISSION_BOUNDARY' in v['labels'] for v in proof['violations'])
        assert not proof['uses_actual_qpos_as_reference']
    good('ACTUAL_EXECUTED_COMMAND_SEAM_REJECTS_JUMP',seam)
    def classify():
        fixture=np.zeros((40,6));fixture[35:,2]=.01
        value=stencils(fixture,5.6+np.arange(40)*.05,connector_end=7.215003467542307)
        assert all(v['labels']==['WITHIN_SEGMENT'] for v in value['violations'])
    good('NEAR_CONNECTOR_IS_NOT_CROSSING_CONNECTOR',classify)
    return dict(status='PASS_REFERENCE_FIXTURE_CONTRACTS_ONLY',cases=rows,new_physics=0,new_IK=0,new_RRT=0,
                model_queries=0,task_qualification=False)
