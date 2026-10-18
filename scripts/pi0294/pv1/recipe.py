"""PV1 recipe and update identity; legacy source-pair admission stays unchanged."""
from copy import deepcopy

from scripts.pi0294.mp1.train import validate_recipe as validate_legacy
from scripts.pi0294.scene_cards import digest
from .sampler import ARMS


PARENT = '94ad4d0eaaa5bf7df0c55a7b0c9c06701b4039fa2a32a98a29453a1974399c38'
FROZEN = ('optimizer','weight_decay','adam_beta1','adam_beta2','adam_epsilon','max_grad_norm',
    'bf16','tf32','gradient_checkpointing','numerical_configuration','trainable_modules','VLM_frozen','statistics_fit')


def primary_recipe(original,route,arm,*,replica=False):
    validate_legacy(original,'S-CORR')
    if route not in ARMS or arm not in ARMS[route] or type(replica) is not bool:
        raise PermissionError('PV1_FIXED_PRIMARY_PAIR_OR_LOCAL_RECIPE_REPLICATION')
    result={k:deepcopy(original[k]) for k in FROZEN}
    result.update(schema='PI0294_PV1_PRIMARY_LEARNING_RECIPE_V1',route=route,arm=arm,stage='P2',
        mature_source_recipe_canonical_sha256=digest(original),
        initial_parent_checkpoint_manifest_sha256=PARENT,
        initial_optimizer='FRESH_IDENTICAL_IN_EACH_PAIRED_ARM',exact_resume_required_after_interruption=True,
        max_steps=4096,scheduler_horizon_steps=4096,warmup_updates=64,
        seed=20261020 if replica else 20261019,replication_of_local_recipe=replica,
        effective_batch=32,per_device_train_batch_size=2,gradient_accumulation_steps=16,workers=0,
        learning_rate=1e-5,end_learning_rate=1e-6,scheduler='COSINE_MIN_LR_FLOOR',
        checkpoints=[1024,4096],development_candidate_steps=[4096],
        inference_num_steps=4,inference_TF32=False,Teacher_takeover_during_evaluation=False,
        sampling='PV1_MATCHED_ABSOLUTE_CURSOR_LANE_PARENT_WINDOW_V1',sample_visits=131072,
        new_data_frozen_before_training=True)
    return result


def validate_primary(recipe,original):
    expected=primary_recipe(original,recipe['route'],recipe['arm'],replica=recipe['replication_of_local_recipe'])
    if recipe!=expected:
        raise PermissionError('PV1_EXACT_PREFROZEN_PRIMARY_RECIPE')
    return expected


def begin_update(ledger,*,arm,local_step,request_id,recipe):
    if (recipe.get('schema')!='PI0294_PV1_PRIMARY_LEARNING_RECIPE_V1'
            or arm!=recipe['arm'] or type(local_step) is not int or not 0<=local_step<4096
            or ledger['pending_updates'] or not request_id
            or ledger['optimizer_updates']>=16384):
        raise PermissionError('PV1_ONE_KNOWN_AUTHORIZED_LOCAL_UPDATE_WITHIN_CAMPAIGN_CAP')
    value=deepcopy(ledger)
    value['pending_updates']=[dict(arm=arm,local_step_before=local_step,request_id=request_id)]
    return value


def commit_update(ledger,*,arm,local_step_after,request_id):
    pending=ledger['pending_updates']
    if (len(pending)!=1 or pending[0]['arm']!=arm or pending[0]['request_id']!=request_id
            or type(local_step_after) is not int or local_step_after!=pending[0]['local_step_before']+1
            or ledger['optimizer_updates']>=16384):
        raise PermissionError('PV1_COMMIT_ONLY_THE_EXACT_PENDING_NATIVE_OPTIMIZER_STEP')
    value=deepcopy(ledger)
    value['optimizer_updates']+=1
    value['pending_updates']=[]
    return value
