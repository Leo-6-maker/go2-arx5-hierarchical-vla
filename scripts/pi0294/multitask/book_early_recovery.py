"""Genuine early student states on original TRAIN layouts, never DEV states."""
from .book_seed_population import plan as population_plan, ORIGINAL_TRAIN_OFFSETS_M
from .libero_book_teacher import route_card, seed_card, RELEASE_REVISION, teacher_for_card

CHECKPOINT = population_plan()['recovery']['checkpoint_manifest_sha256']
KIND = 'V7_BOOK_EARLY_TRAIN_RECOVERY_PANEL'
CONDITION_OUTPUT_BYTES = 288*1024**2
OWNER_OUTPUT_BYTES = 3328*1024**2


def conditions():
    return [dict(id=f'BOOK_M32_RECOVERY{i:03}', TRAIN_parent=p,
                 takeover_index=t, xy_offset_m=list(ORIGINAL_TRAIN_OFFSETS_M[p]))
            for i, (p, t) in enumerate((p, t) for p in range(5) for t in (8, 16))]


def plan():
    return dict(schema='BOOK_M32_EARLY_TRAIN_RECOVERY_PANEL_V1',
        checkpoint_manifest_sha256=CHECKPOINT, conditions=conditions(),
        source_population=population_plan()['schema'], student_generation_queries=15,
        original_TRAIN_only=True, DEV_or_TEST_states_or_labels=False,
        expert_revision=RELEASE_REVISION, full_task_seconds=32, warmup_seconds=5,
        H40=40, prefix=8, student_prefix_is_expert=False,
        expert_uses_actual_takeover_state=True, training_admitted=False,
        condition_output_upper_bytes=CONDITION_OUTPUT_BYTES, owner_output_upper_bytes=OWNER_OUTPUT_BYTES,
        full_task_and_native_safety_and_independent_audit_required_for_suffix=True,
        hard_native_safety_stops_panel=True, automatic_retry=False)


def validate_manifest(m):
    required=dict(optimizer_updates=0, physics_integrations=185000,
                  generation_queries=15, RGB_pairs=6420, source_constructors=10, model_loads=1,
                  output_bytes=OWNER_OUTPUT_BYTES)
    if (m['kind']!=KIND or m.get('recovery_plan')!=plan()
            or m.get('delta_checkpoint_manifest_sha256')!=CHECKPOINT
            or any(m['limits'].get(k)!=v for k,v in required.items())):
        raise PermissionError('DECLARED_EARLY_TRAIN_RECOVERY_PANEL_NO_DEV_OR_RETRIES')


def original_card(condition):
    if condition not in conditions():raise ValueError('EXACT_ORIGINAL_TRAIN_EARLY_RECOVERY_CONDITION')
    p=condition['TRAIN_parent']
    return route_card(RELEASE_REVISION)if p==0 else seed_card(p)


def recovery_card(condition):
    card=original_card(condition)
    card.update(id=condition['id'], study_split='TRAIN_RECOVERY_DIAGNOSTIC',
                original_TRAIN_parent=f"BOOK_SOURCE_SEED{condition['TRAIN_parent']:03}",
                initial_xy_offset_m=list(condition['xy_offset_m']))
    return card


def make_student(base,instance,identifier,record,guard,*,teacher_type=teacher_for_card):
    matches=[c for c in conditions()if c['id']==identifier]
    if len(matches)!=1:raise ValueError('DECLARED_EARLY_TRAIN_RECOVERY_IDENTITY')
    condition=matches[0];takeover=condition['takeover_index'];queries=takeover//8

    class Recovery(base):
        def __init__(self):
            super().__init__(instance,identifier,record,guard,
                             seed_namespace='BOOK_M32_EARLY_TRAIN_RECOVERY')
            self.expert=None

        def observe(self,state):
            super().observe(state)
            if self.index==takeover:
                if (self.queries!=queries or self.completed_generations!=queries
                        or state.get('safety_valid')is not True or state.get('native_grasp')):
                    raise PermissionError('ACTUAL_SAFE_UNGRASPED_EARLY_STUDENT_TAKEOVER')
                self.expert=teacher_type(original_card(condition))
                record('expert_takeover',dict(sequence=self.index//8,index=self.index,
                    original_TRAIN_parent=f"BOOK_SOURCE_SEED{condition['TRAIN_parent']:03}",
                    actual_state_eef10=self.state.tolist(), student_completed_generations=queries,
                    student_prefix_is_expert=False,training_admitted=False))
            if self.expert is not None:
                self.expert.observe(state);self.stage=self.expert.stage
                self.candidate_complete=self.expert.candidate_complete

        def plan(self):
            if self.index<takeover:return super().plan()
            if self.expert is None:raise PermissionError('NO_IMPLICIT_EXPERT_RESTART')
            self.guard();return self.expert.plan()

    return Recovery()


def eligible_recovery_anchors(condition,completed_indices,*,full_task_success,native_safe):
    original_card(condition)
    if not full_task_success or not native_safe:return ()
    if list(completed_indices)!=list(range(640)):
        raise ValueError('COMPLETE_ACTUAL_PREFIX_AND_SUFFIX_REQUIRED')
    return tuple(range(condition['takeover_index'],601))
