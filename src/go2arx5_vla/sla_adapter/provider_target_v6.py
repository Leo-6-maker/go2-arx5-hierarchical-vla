"""New V6 lineage; reuse unchanged ordinary RGB SHIFT7 decode and sampler."""
import json,shutil
from pathlib import Path
from .checkpoint_v1 import sha
from .provider_time_shift7_v1 import Provider as ShiftProvider
from .target_contract_v6 import RECIPE,SCHEMA,contract
from .gripper_time_shift7_v1 import VERSION,SHIFT,SLOTS
from .gripper_latent_v2 import EPSILON


def export(package,checkpoint,*,complete,active_template,statistics,tokenizer,parent_manifest):
    from .provider_v1 import active_rows
    b=complete['bindings'];checkpoint=Path(checkpoint);package=Path(package)
    if(complete['step']!=500 or complete['cursor']!=16000 or b['recipe']!=RECIPE
       or b['parent_manifest']!=parent_manifest or b['derived_statistics_sha256']!=sha(statistics)
       or b['tokenizer']!=sha(tokenizer)):raise ValueError('FIXED_REAL_V6_500_AND_FROZEN_ASSETS_REQUIRED')
    package.mkdir(exist_ok=False);(package/'adapters').mkdir();rows,_=active_rows(complete,active_template);copied=[]
    if len(rows)!=28:raise ValueError('ALL28_ADAPTERS_MUST_BE_DEPLOYED')
    for row in rows:
        path=checkpoint/row['file']
        if not path.resolve().is_relative_to(checkpoint.resolve())or sha(path)!=row['sha256']:raise ValueError('ACTUAL_ENDPOINT_ARRAY_CHANGED')
        destination=package/'adapters'/row['file'];shutil.copyfile(path,destination);copied.append(dict(row,file='adapters/'+row['file']))
    shutil.copyfile(statistics,package/'STATISTICS.json');shutil.copyfile(tokenizer,package/'paligemma_tokenizer.model')
    m=dict(schema=SCHEMA,codec=VERSION,decode_order='INVERSE_FROZEN_QUANTILES_SIGMOID_PARAMETERS_REFERENCE_CONTINUED_ROBOT_TIME_QUINTIC',
        parameter_slot_shift=SHIFT,parameter_slots=[[int(i)for i in g]for g in SLOTS],label_epsilon=EPSILON,state_gripper_transformed=False,
        step=500,cursor=16000,bindings=b,parent_manifest=parent_manifest,original_complete_sha256=sha(checkpoint/'COMPLETE.json'),adapter_arrays=copied,
        statistics_sha256=sha(package/'STATISTICS.json'),tokenizer_sha256=sha(package/'paligemma_tokenizer.model'),camera_keys=['base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb'],
        image_masks=[True,True,False],image_size=[224,224],sampler_steps=10,action_horizon=40,label_interval_s=.05,previous_reference_required=True,
        teacher_required=False,optimizer_required=False,training_cache_required=False,physical_execution_admitted=False,endpoint_basis_applied=False)
    contract(m);(package/'MANIFEST.private.json').write_text(json.dumps(m,indent=2)+'\n');return m


class Provider(ShiftProvider):
    def validate_package_contract(self,manifest):contract(manifest)
