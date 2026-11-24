"""The original H0 teacher with a compact open-gripper approach waypoint."""

import numpy as np

from go2arx5_vla.world_model.multitask_teacher_v6 import MultitaskTeacher
from sea_compact_workspace import HOVER_ABOVE_GRASP_M


class CompactWorkspaceTeacher(MultitaskTeacher):
    def hover(self):
        return self.grasp_goal() + [0., 0., HOVER_ABOVE_GRASP_M]

    def goal(self, sample):
        if self.stage == "PREGRASP":
            self.cube = np.asarray(sample["object_pose"]).copy()
            hover = self.hover()
            if np.linalg.norm(np.asarray(sample["tcp"])[:3, 3] - hover) > .004:
                return hover
            self.stage = "DESCEND"
        return super().goal(sample)
