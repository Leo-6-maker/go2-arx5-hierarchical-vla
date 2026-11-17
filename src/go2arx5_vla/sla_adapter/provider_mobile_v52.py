"""Explicit V5.2 successor lineage; same SHIFT7 ordinary RGB decode/sampling."""
import json
from pathlib import Path
import shutil
from .checkpoint_v1 import sha
from .provider_time_shift7_v1 import Provider as ShiftProvider
from .mobile_contract_v52 import RECIPE,PARENT,PLAN,SCHEMA,ORDER,contract
from .gripper_time_shift7_v1 import VERSION,SHIFT,SLOTS
from .gripper_latent_v2 import VERSION as STATS_CODEC,EPSILON

def export(package,checkpoint,*,complete,active_template,statistics,tokenizer,parent_manifest):
    from .provider_v1 import active_rows
    checkpoint=Path(checkpoint);package=Path(package);b=complete['bindings']
    if (complete['step']!=500 or complete['cursor']!=16000 or b['recipe']!=RECIPE
            or b['parent_manifest']!=parent_manifest or b['derived_statistics_sha256']!=sha(statistics)
            or b['tokenizer']!=sha(tokenizer)):
        raise ValueError('ACTUAL_NEW500_FROZEN_STATS_AND_FOUNDATION_PARENT')
    package.mkdir(exist_ok=False);(package/'adapters').mkdir();rows,_=active_rows(complete,active_template);copied=[]
    for row in rows:
        source=checkpoint/row['file']
        if not source.resolve().is_relative_to(checkpoint.resolve())or sha(source)!=row['sha256']:raise ValueError('ENDPOINT_ADAPTER_BYTES_CHANGED')
        destination=package/'adapters'/row['file'];shutil.copyfile(source,destination);copied.append(dict(row,file='adapters/'+row['file']))
    shutil.copyfile(statistics,package/'STATISTICS.json');shutil.copyfile(tokenizer,package/'paligemma_tokenizer.model')
    m=dict(schema=SCHEMA,codec=VERSION,decode_order=ORDER,parameter_slot_shift=SHIFT,parameter_slots=[[int(i)for i in g]for g in SLOTS],label_epsilon=EPSILON,state_gripper_transformed=False,step=500,cursor=16000,bindings=b,parent_manifest=parent_manifest,original_complete_sha256=sha(checkpoint/'COMPLETE.json'),adapter_arrays=copied,statistics_sha256=sha(package/'STATISTICS.json'),tokenizer_sha256=sha(package/'paligemma_tokenizer.model'),camera_keys=['base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb'],image_masks=[True,True,False],image_size=[224,224],sampler_steps=10,action_horizon=40,label_interval_s=.05,previous_reference_required=True,teacher_required=False,optimizer_required=False,training_cache_required=False,physical_execution_admitted=False,endpoint_basis_applied=False)
    contract(m);(package/'MANIFEST.private.json').write_text(json.dumps(m,indent=2)+'\n');return m


class Provider(ShiftProvider):
    def validate_package_contract(self,manifest):contract(manifest)
