"""Versioned time-domain wrapper; original spatial and physical predicates intact."""
import ast
import copy
import inspect
import numpy as np
from go2arx5_vla.target_sim import scorer as original
from . import score as push_original


def bind(function, duration, table):
    if duration not in (32,64,96,128) or table not in (.44,.47):raise ValueError('NIGHT_FIXED_TIME_OR_HEIGHT_DOMAIN')
    replacements={641:20*duration+1,560:20*(duration-4),601:20*(duration-2)+1,16000:500*duration,16001:500*duration+1}
    class Times(ast.NodeTransformer):
        def visit_Constant(self,node):
            if type(node.value) is int and node.value in replacements:return ast.copy_location(ast.Constant(replacements[node.value]),node)
            return node
    tree=ast.fix_missing_locations(Times().visit(ast.parse(inspect.getsource(function))))
    scope=dict(function.__globals__)
    if 'P' in scope:scope['P']=dict(scope['P'],table_top=table)
    exec(compile(tree,'<NIGHT_EXPLICIT_TIME_BINDING>','exec'),scope)
    return scope[function.__name__]


def score(data,card,*,complete,high_rate_ranges_PASS):
    duration=int(card['duration_s']);table=card['table_top_m']
    if card['task']=='T1':
        result=bind(original.score_episode,duration,table)(data,goal_center=[*card['goal_xy'],table+.02],initial_object_pose=data['object_pose'][0],table_height_m=table,complete=complete,evidence_source='REAL_MUJOCO_FULL_GO2_ARX5')
    else:result=bind(push_original.score_push,duration,table)(data,card,complete=complete)
    result['evidence_source']='REAL_MUJOCO_FULL_GO2_ARX5'
    result['complete_declared_duration']=result['complete_32s']
    result['complete_32s']=bool(duration==32 and result['complete_32s'])
    result['persistent_success']=bool(result['persistent_success'] and high_rate_ranges_PASS)
    result.update(duration_domain='SEED32' if duration==32 else 'SEED_SLOW'+str(duration),duration_s=duration,high_rate_raw_ranges_PASS=high_rate_ranges_PASS)
    return result
