import itertools
import json
import unittest
import numpy as np
from .persistent_approach_draft import PersistentApproachDraft,precontact_geometry


class ApproachDraftTests(unittest.TestCase):
    def planner(self,task='T1',far=True):
        actual=np.eye(4);actual[:3,3]=[.2,0,.7]
        target=actual.copy();target[:3,3]=[.6 if far else .2,-.08 if far else 0,.54]
        base=np.eye(4);station=base.copy();station[0,3]=.15
        return PersistentApproachDraft(task=task,actual_tcp=actual,stage_target=target,actual_BASE=base,station_world=station,pad_radius_m=.05),actual,base

    def test_both_tasks_start_from_actual_near_or_far_no_teleport(self):
        for task in ('T1','T2'):
            for far in (False,True):
                p,actual,base=self.planner(task,far);p.observe(time_s=0,actual_tcp=actual,actual_BASE=base)
                plan=p.plan(time_s=0,actual_tcp=actual,actual_BASE=base)
                self.assertFalse(plan['launch_allowed']);np.testing.assert_array_equal(plan['H40_world_poses'][0],actual)
                self.assertEqual(plan['execute_physics_steps'],200);self.assertEqual(plan['record_every_physics_steps'],25)

    def test_actual_perturbation_does_not_reanchor_persistent_cursor_or_station(self):
        p,a,b=self.planner();p.observe(time_s=0,actual_tcp=a,actual_BASE=b);p.plan(time_s=0,actual_tcp=a,actual_BASE=b)
        changed=a.copy();changed[0,3]-=.002
        p.observe(time_s=.4,actual_tcp=changed,actual_BASE=b)
        plan=p.plan(time_s=.4,actual_tcp=changed,actual_BASE=b)
        self.assertGreater(np.linalg.norm(plan['H40_world_poses'][0,:3,3]-changed[:3,3]),0.)
        self.assertEqual(plan['persistent_station_world'][0,3],.15)
        with self.assertRaises(ValueError):p.plan(time_s=.4,actual_tcp=changed,actual_BASE=b)

    def test_stall_stops_without_manipulation_or_success(self):
        p,a,b=self.planner()
        for i in range(81):diagnostic=p.observe(time_s=i*.05,actual_tcp=a,actual_BASE=b)
        self.assertEqual(diagnostic['stop_reason'],'NO_TASK_APPROACH_PROGRESS_NO_FORCED_STAGE_TRANSITION')
        self.assertFalse(diagnostic['task_success'])
        with self.assertRaises(PermissionError):p.plan(time_s=4.,actual_tcp=a,actual_BASE=b)

    def test_reference_lead_bounded_without_rewriting_actual_state(self):
        p,a,b=self.planner();copy=a.copy();p.observe(time_s=0,actual_tcp=a,actual_BASE=b)
        plan=p.plan(time_s=0,actual_tcp=a,actual_BASE=b)
        self.assertLessEqual(np.linalg.norm(plan['H40_world_poses'][:,:3,3]-a[:3,3],axis=1).max(),.024000001)
        np.testing.assert_array_equal(a,copy)

    def test_geometry_cannot_close_far_from_object_or_by_timer(self):
        corners=np.asarray(list(itertools.product((-.02,.02),repeat=3)))
        pads=np.stack([corners*np.array([1,.15,.85])+[0,y,.07] for y in (-.045,.045)])
        obj=np.eye(4)
        self.assertTrue(precontact_geometry(pad_corners_world=pads,object_pose=obj)['precontact_geometry_ready'])
        pads[:,:,0]+=.4
        self.assertFalse(precontact_geometry(pad_corners_world=pads,object_pose=obj)['precontact_geometry_ready'])

    def test_native_safety_remains_a_hard_stop(self):
        p,a,b=self.planner();p.observe(time_s=0,actual_tcp=a,actual_BASE=b,native_safe=False)
        with self.assertRaises(PermissionError):p.plan(time_s=0,actual_tcp=a,actual_BASE=b)

    def test_continuous8_equals4_save_restore4_cpu_plans(self):
        p,a,b=self.planner();q,_,_=self.planner()
        for i in range(8):
            if i==4:q=PersistentApproachDraft.restore(json.loads(json.dumps(q.snapshot())))
            # Fixture observations represent inputs only, not physical success.
            a=a.copy();a[0,3]+=.001
            for item in (p,q):item.observe(time_s=i*.4,actual_tcp=a,actual_BASE=b)
            left=p.plan(time_s=i*.4,actual_tcp=a,actual_BASE=b);right=q.plan(time_s=i*.4,actual_tcp=a,actual_BASE=b)
            np.testing.assert_array_equal(left['H40_world_poses'],right['H40_world_poses'])
            np.testing.assert_array_equal(left['BASE_soft_velocity'],right['BASE_soft_velocity'])
            self.assertEqual(p.snapshot(),q.snapshot())

    def test_real_orientation_progress_is_not_false_position_stall(self):
        p,a,b=self.planner();p.target[:3,:3]=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        for i in range(81):
            theta=.2*i*.05;actual=a.copy();c,s=np.cos(theta),np.sin(theta)
            actual[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]]
            diagnostic=p.observe(time_s=i*.05,actual_tcp=actual,actual_BASE=b)
        self.assertIsNone(diagnostic['stop_reason'])
