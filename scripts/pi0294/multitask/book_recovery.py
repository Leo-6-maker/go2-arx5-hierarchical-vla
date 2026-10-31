"""One TRAIN student-prefix/expert-recovery episode, using existing capture.

The original DEV population is not a source of states or training labels.
The student prefix is never expert supervision. Successful recovery still
requires the original full task, native safety and independent raw audit.
"""
from .libero_book_teacher import route_card, RELEASE_REVISION, teacher_for_card

TAKEOVER = 32  # Four actual Prefix8 chunks /1.6s, fixed before execution.
CONDITION = dict(id='BOOK_RECOVERY_TRAIN000', xy_offset_m=[0., 0.])
CHECKPOINT = 'b8327ab9501eac641eae8947a04040fe84844a01e9e5f59cea1c2ffd17ad11f8'


def plan():
    return dict(schema='BOOK5_TRAIN_STUDENT32_EXPERT_RECOVERY_DIAGNOSTIC_V1',
        checkpoint_manifest_sha256=CHECKPOINT, condition=dict(id=CONDITION['id'],xy_offset_m=[0.,0.]),
        original_TRAIN_parent='BOOK_SOURCE_SEED000', takeover_observation=TAKEOVER,
        student_queries=4, student_prefix_intervals=32, task_intervals=640,
        warmup_integrations=2500, task_seconds=32, H40=40, prefix=8,
        actual_integrations_upper=18500, actual_RGB_pairs_upper=642,
        original_common_home_goal_and_safety_unchanged=True,
        DEV_or_TEST_used=False, optimizer_updates=0, formal_DEV_conditions=0,
        student_prefix_is_expert=False, training_admission=False,
        purpose='Can the unchanged expert recover from this frozen student on an existing TRAIN layout? Not a formal DEV retry or a learned full-task success.')


def validate_manifest(m):
    expected=dict(optimizer_updates=0, physics_integrations=18500,
        generation_queries=4, RGB_pairs=642, source_constructors=1, model_loads=1)
    if (m['kind']!='V7_BOOK_TRAIN_RECOVERY_DIAGNOSTIC'
            or m.get('recovery_plan')!=plan()
            or m.get('delta_checkpoint_manifest_sha256')!=CHECKPOINT
            or any(m['limits'].get(k)!=v for k,v in expected.items())):
        raise PermissionError('ONE_EXACT_TRAIN_RECOVERY_DIAGNOSTIC_NO_DEV_OR_EXPANSION')


def recovery_card(condition):
    if condition!=CONDITION:raise ValueError('ONE_EXISTING_TRAIN_LAYOUT_NOT_DEV')
    card=route_card(RELEASE_REVISION)
    card.update(id=CONDITION['id'],study_split='TRAIN_RECOVERY_DIAGNOSTIC',
        original_TRAIN_parent='BOOK_SOURCE_SEED000', initial_xy_offset_m=[0.,0.])
    return card


def make_student(base,instance,identifier,record,guard,*,teacher_type=teacher_for_card):
    if identifier!=CONDITION['id']:raise ValueError('EXACT_TRAIN_RECOVERY_IDENTITY')

    class Recovery(base):
        def __init__(self):
            super().__init__(instance,identifier,record,guard,
                seed_namespace='BOOK_RECOVERY_TRAIN32')
            self.expert=None

        def observe(self,state):
            super().observe(state)
            if self.index==TAKEOVER:
                if (self.queries!=4 or self.completed_generations!=4
                        or state.get('safety_valid')is not True
                        or state.get('native_grasp')):
                    raise PermissionError('SAFE_UNGRASPED_ACTUAL_STUDENT_PREFIX_BEFORE_HANDOVER')
                # The actual qualified routed Teacher keeps its exact card;
                # recovery provenance belongs to the surrounding recorder.
                self.expert=teacher_type(route_card(RELEASE_REVISION))
                record('expert_takeover',dict(sequence=self.index//8,index=self.index,
                    original_TRAIN_parent='BOOK_SOURCE_SEED000',
                    actual_state_eef10=self.state.tolist(),student_completed_generations=4,
                    student_prefix_is_expert=False,training_admitted=False))
            if self.expert is not None:
                self.expert.observe(state)
                self.stage=self.expert.stage
                self.candidate_complete=self.expert.candidate_complete

        def plan(self):
            if self.index<TAKEOVER:return super().plan()
            if self.expert is None:raise PermissionError('NO_IMPLICIT_EXPERT_RESTART')
            self.guard()
            return self.expert.plan()

    return Recovery()


def eligible_recovery_anchors(completed_indices,*,full_task_success,native_safe):
    """Eligibility only; this neither exports nor admits a training dataset."""
    if not full_task_success or not native_safe:return ()
    if list(completed_indices)!=list(range(640)):
        raise ValueError('ALL_ACTUAL_COMPLETED_INTERVALS_REQUIRED')
    return tuple(range(TAKEOVER,601))
