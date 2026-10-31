"""Full autonomous TRAIN reproduction and paired feedback diagnostic.

Uses the existing model, capture, engine and task/safety scorer. No expert
takeover, privileged student inputs, changed task or original DEV result.
"""
import hashlib
from .libero_book_teacher import route_card, seed_card, RELEASE_REVISION
from .book_seed_population import ORIGINAL_TRAIN_OFFSETS_M, ADDITIONAL_TRAIN_OFFSETS_M

KIND='V7_BOOK_TRAIN_ROLLOUT_DIAGNOSTIC'
B512='862cabd5657ebfe952fbc200c3162d6a63dcd299784694e0f3e78d3a27c57030'
PARENTS=(0,7,13)


def conditions():
    offsets=ORIGINAL_TRAIN_OFFSETS_M+ADDITIONAL_TRAIN_OFFSETS_M
    return [dict(id=f'BOOK_TRAIN{p:03}',TRAIN_parent=p,xy_offset_m=list(offsets[p]))for p in PARENTS]


def plan(prefix=8):
    if type(prefix)is not int or prefix not in(4,8):raise ValueError('DECLARED_PREFIX8_OR_PREFIX4')
    return dict(schema='BOOK_M32_B512_TRAIN_REPRODUCTION_FEEDBACK_V1',
        checkpoint_manifest_sha256=B512,conditions=conditions(),execution_prefix=prefix,H40=40,
        warmup_seconds=5,task_seconds=32,task_scoring_and_native_safety_unchanged=True,
        original_DEV_or_confirmation=False,expert_takeover=False,training_admitted=False,
        noise='SHA256_BOOK_M32_TRAIN_FEEDBACK_V1_condition_absolute_observation_index',
        common_observation_boundaries_have_identical_seed=True,automatic_retry=False,
        hard_native_safety_stops_panel=True)


def validate_manifest(m):
    p=m['rollout_diagnostic'];prefix=p['execution_prefix']
    if m['kind']!=KIND or p!=plan(prefix)or m['delta_checkpoint_manifest_sha256']!=B512:
        raise PermissionError('FIXED_B512_TRAIN_ONLY_FEEDBACK_DIAGNOSTIC')
    required=dict(model_loads=1,optimizer_updates=0,physics_integrations=55500,
        generation_queries=3*640//prefix,RGB_pairs=1926,source_constructors=3)
    if any(m['limits'].get(k)!=v for k,v in required.items()):
        raise PermissionError('FINITE_THREE_TRAIN_CONDITIONS_NO_DEV_RETRY')


def card(condition):
    if condition not in conditions():raise ValueError('PREDECLARED_TRAIN_REPRODUCTION_ONLY')
    p=condition['TRAIN_parent'];c=route_card(RELEASE_REVISION)if p==0 else seed_card(p)
    c.update(id=condition['id'],study_split='TRAIN_AUTONOMOUS_DIAGNOSTIC',
        original_TRAIN_parent=f'BOOK_SOURCE_SEED{p:03}',expert_takeover=False)
    return c


def student_factory(base,prefix):
    plan(prefix)
    class DiagnosticStudent(base):
        execution_prefix=prefix
        def generation_seed(self,sequence):
            # Seed follows physical observation time, not the number of queries.
            text=f'BOOK_M32_TRAIN_FEEDBACK_V1/{self.identifier}/{self.index}'
            return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],'big')%(2**63)
    return DiagnosticStudent
