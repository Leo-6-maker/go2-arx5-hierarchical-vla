"""Same-observation frozen baseline input; expert labels never enter the model."""
import base64,hashlib,io,json
from pathlib import Path
import numpy as np
from PIL import Image
from .storage import digest
from .target_gr00t_dataset_v1 import TargetNativeDataset,need
from scripts.pi0292.relay_protocol import unseal

MODE='SEALED_TARGET_GR00T_EXPERT_DECISION_V1'


def recorded_payload(spec):
    root=Path(spec['baseline_dataset']);producer=root.parent
    need(digest(root/'DATASET_MANIFEST.json')==spec['baseline_dataset_sha256'],'BASELINE_SAME_SEALED_DATASET')
    lease=json.loads((producer/'control/LEASE.json').read_bytes())
    worker=json.loads((producer/'WORKER_EXIT.private.json').read_bytes())
    result=json.loads((producer/'RESULT.json').read_bytes())
    need(lease['status']=='CLOSED'and lease['worker_exit_code']==worker['exit_code']==0
         and lease['spec_sha256']==digest(producer/'EXECUTION_SPEC.json')
         and lease['result_sha256']==digest(producer/'RESULT.json')
         and result['status']=='ACTUAL_FOUR_DEMOS_PROCESSOR_AND_PRIVATE_ARCHIVE_SEALED'
         and result['dataset_sha256']==spec['baseline_dataset_sha256'], 'BASELINE_ACTUAL_CORE_EXPORT_CLOSED0')
    bindings=spec['neural_bindings']
    data=TargetNativeDataset(root,bindings['source'],bindings['cosmos'])
    # One predeclared real policy decision, no seed/view selection after inference.
    row,anchor=data.rows[0]
    need(row['id']==spec['baseline_episode'] and anchor==0,'BASELINE_PREREGISTERED_FIRST_TRAIN_ANCHOR')
    state,expert,images,text=data.raw(0)
    proof=row['proof']['capture_index'];path=Path(proof['path'])
    need(digest(path)==proof['sha256'],'BASELINE_ORIGINAL_CAPTURE_INDEX')
    capture=json.loads(path.read_bytes())['captures'][0];path=Path(capture['path'])
    need(capture['atom']==0 and digest(path)==capture['sha256']
         and str(path)==spec['recorded_observation']and capture['sha256']==spec['recorded_observation_sha256'],
         'BASELINE_EXACT_ORIGINAL_POLICY_OBSERVATION')
    body=unseal(json.loads(path.read_bytes()))
    from .fusion_observation import shadow_inference_payload
    payload=shadow_inference_payload(body['original_shadow_ABI'])
    need(np.array_equal(np.asarray(payload['state_eef10']),state)and payload['language']==text,
         'BASELINE_NO_STATE_OR_LANGUAGE_SUBSTITUTION')
    need(len(payload['images'])==2,'BASELINE_TWO_REAL_IMAGES')
    for image,expected,camera in zip(payload['images'],images,('external_rgb','wrist_rgb')):
        raw=base64.b64decode(image['png_base64'],validate=True)
        need(image['camera']==camera and hashlib.sha256(raw).hexdigest()==image['sha256']
             and np.array_equal(np.asarray(Image.open(io.BytesIO(raw))),expected),'BASELINE_SAME_ACTUAL_RGB')
    return payload,dict(observation_mode=MODE,actual_saved_target_images=True,target_online_images=False,
        actual_same_observation_teacher=True,teacher_policy_H40=expert.tolist(),
        teacher_H40_sha256=hashlib.sha256(expert.tobytes()).hexdigest(),dataset_sha256=data.manifest_sha,
        episode=row['id'],native_step=capture['native_step'],task_prompt_matches_target_task=True,
        baseline_pretraining_support_for_this_instruction='NOT_ESTABLISHED',
        expert_labels_or_privileged_geometry_in_student_input=False,target_task_native_qualified=False,
        original_checkpoint_and_original_Processor=True,new_target_statistics_used_for_baseline=False)
