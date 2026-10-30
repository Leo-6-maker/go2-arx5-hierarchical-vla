"""Vertical placement Teacher for the pinned caddy source port only.

Uses actual grasp geometry and persistent paths; never advances by time alone.
Native LIBERO success and additional demonstration checks remain separate.
"""
import numpy as np
from copy import deepcopy
import math
from .feedback_teacher import SourceSkillTeacher
from .libero_book_scene import VERSION
from .object_relations import inverse, pose


class LiberoBookTeacher(SourceSkillTeacher):
    def __init__(self, card):
        if card.get('source_port') != VERSION: raise ValueError('EXACT_LIBERO_SOURCE_PORT')
        super().__init__(card)

    def _book_supported_inside(self):
        s = self.state
        return (s.get('native_libero_goal') is True and s.get('object_supported_by_shelf') is True
                and np.linalg.norm(s['object_linear_velocity']) <= .02
                and np.linalg.norm(s['object_angular_velocity']) <= .1)

    def _book_targets(self):
        s = self.state
        insert = pose(self.spec['target_book_world']) @ inverse(s['object_pose']) @ pose(s['tcp'])
        above = insert.copy(); above[2, 3] += .12
        return above, insert

    def _withdraw_direction(self):
        return np.array([0., 0., 1.])


ROUTE_REVISION='LIBERO_ARX5_PORT03_TOP_EDGE_FRONT_CROSS_LOWER'
MARGIN_REVISION='LIBERO_ARX5_PORT04_FRONT_TRACKING_MARGIN'


def route_card(revision=ROUTE_REVISION):
    """The one left condition with a complete saved static/swept reference."""
    from .libero_book_cards import cards
    if revision not in(ROUTE_REVISION,MARGIN_REVISION):raise ValueError('KNOWN_STATIC_BOOK_ROUTE')
    card=deepcopy(cards()['left']);semantic=np.diag([-1.,1.,-1.])
    offset=np.array(card['book']['object_to_tcp_grasp']);angle=math.radians(20.)
    tilt=np.array([[1.,0.,0.],[0.,math.cos(angle),-math.sin(angle)],
                   [0.,math.sin(angle),math.cos(angle)]])
    offset[:3,:3]=semantic.T@tilt@semantic@offset[:3,:3]
    offset[:3,3]=[-.001786,.025,-.05]
    identifier=('NIGHT30_LIBERO_BOOK_LEFT_PORT03_TOP_EDGE_ROUTE'if revision==ROUTE_REVISION
                else'NIGHT30_LIBERO_BOOK_LEFT_PORT04_TRACKING_MARGIN')
    card.update(id=identifier,book_route_revision=revision)
    card['book']['object_to_tcp_grasp']=offset.tolist()
    card['book_route']=dict(approach_m=.04,lift_m=.075,front_cross_book_x_m=.454214,
        front_wall_max_x_m=.43716,front_wall_top_z_m=.51268,
        corridor_y_min_m=.07432,corridor_y_max_m=.20462,clearance_m=.001,
        waypoint_position_m=.001,waypoint_angle_rad=.01,
        withdraw_tool_m=.06,withdraw_exterior_x_m=.10)
    if revision==MARGIN_REVISION:card['book_route']['front_cross_book_x_m']=.464214
    return card


def checked_route_card(card,side='left'):
    revision=card.get('book_route_revision')
    if side!='left'or revision not in(ROUTE_REVISION,MARGIN_REVISION)or card!=route_card(revision):
        raise PermissionError('EXACT_STATIC_QUALIFIED_LEFT_BOOK_ROUTE_CARD')
    return True


def attempt_operation(card,side):
    if 'book_route_revision'in card:
        checked_route_card(card,side)
        return ('LIBERO_LEFT_PORT03_TOP_EDGE_ROUTE'if card['book_route_revision']==ROUTE_REVISION
                else'LIBERO_LEFT_PORT04_TRACKING_MARGIN')
    return 'LIBERO_'+side.upper()+'_FIRST01'


def teacher_for_card(card):
    return RoutedLiberoBookTeacher(card)if 'book_route_revision'in card else LiberoBookTeacher(card)


class RoutedLiberoBookTeacher(LiberoBookTeacher):
    """Existing feedback stages, with actual barriers inside the carry route.

    All carry waypoints stay in preinsert, so the original grasp-loss gate
    and CLOSED command remain active. A waypoint cannot advance by elapsed
    time or by an unexecuted H40 suffix. Native safety and scoring are inherited.
    """
    def __init__(self,card):
        checked_route_card(card)
        super().__init__(card)
        self.route=self.card['book_route'];self.waypoints={};self.waypoint_index={}

    def _book_targets(self):
        s=self.state;grasp=inverse(s['object_pose'])@s['tcp']
        target=pose(self.spec['target_book_world']);high=target.copy()
        high[0,3]=s['object_pose'][0,3]
        high[2,3]=self.lift_initial_object_z+self.route['lift_m']
        cross=high.copy();cross[0,3]=self.route['front_cross_book_x_m']
        lower=cross.copy();lower[2,3]=target[2,3]
        self.waypoints['preinsert']=[high@grasp,cross@grasp,lower@grasp]
        self.waypoint_index['preinsert']=0
        return lower@grasp,target@grasp

    def _route_clear(self):
        if self.stage=='withdraw':return not self.state['robot_object_contact']
        size=np.asarray(self.spec['book_size_m'])/2
        from itertools import product
        points=np.array(list(product((-1.,1.),repeat=3)))*size
        B=self.state['object_pose'];points=points@B[:3,:3].T+B[:3,3]
        r=self.route;gap=r['clearance_m'];index=self.waypoint_index['preinsert']
        lateral=(points[:,1].min()>=r['corridor_y_min_m']+gap
                 and points[:,1].max()<=r['corridor_y_max_m']-gap)
        front=index==0 or points[:,0].min()>=r['front_wall_max_x_m']+gap
        height=index==2 or points[:,2].min()>=r['front_wall_top_z_m']+gap
        return bool(lateral and front and height)

    def _reached(self):
        if self.stage not in self.waypoints:return super()._reached()
        if self.goal is None:return False
        from go2arx5_vla.trajectory.so3 import geodesic_angle
        reached=(np.linalg.norm(self.state['tcp'][:3,3]-self.goal[:3,3])<=self.route['waypoint_position_m']
                 and geodesic_angle(self.state['tcp'][:3,:3],self.goal[:3,:3])<=self.route['waypoint_angle_rad']
                 and self._route_clear())
        if not reached:return False
        stage=self.stage;index=self.waypoint_index[stage]
        if index+1==len(self.waypoints[stage]):return True
        self.events.append(dict(observation_index=self.index,time_s=self.state['time_s'],
            stage=stage,event='ACTUAL_ROUTE_WAYPOINT_REACHED',waypoint=index))
        self.waypoint_index[stage]+=1;self.path=None
        self.goal=self.waypoints[stage][index+1].copy()
        return False

    def _build_stage_path(self):
        s=self.state;stage=self.stage
        if stage=='approach':
            self.grasp_goal=s['object_pose']@self.grasp_offset
            self.retreat_direction=s['object_pose'][:3,:3]@self.direction
            goal=self.grasp_goal.copy();goal[:3,3]+=self.retreat_direction*self.route['approach_m']
        elif stage=='lift':
            goal=s['tcp'].copy();goal[2,3]+=self.route['lift_m']
        elif stage=='preinsert':goal=self.waypoints[stage][self.waypoint_index[stage]].copy()
        elif stage=='withdraw':
            if stage not in self.waypoints:
                retreat=s['tcp'].copy();retreat[:3,3]-=self.route['withdraw_tool_m']*s['tcp'][:3,0]
                outside=retreat.copy();outside[0,3]-=self.route['withdraw_exterior_x_m']
                self.waypoints[stage]=[retreat,outside];self.waypoint_index[stage]=0
                self.withdraw=outside.copy()
            goal=self.waypoints[stage][self.waypoint_index[stage]].copy()
        else:return super()._build_stage_path()
        from .feedback_teacher import straight_path
        self.path=straight_path(s['tcp'],goal);self.goal=goal;self.path_start=self.index
