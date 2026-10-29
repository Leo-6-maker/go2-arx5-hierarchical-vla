"""Vertical placement Teacher for the pinned caddy source port only.

Uses actual grasp geometry and persistent paths; never advances by time alone.
Native LIBERO success and additional demonstration checks remain separate.
"""
import numpy as np
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
