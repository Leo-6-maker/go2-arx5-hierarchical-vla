"""Explicit V5.2 successor lineage; same SHIFT7 ordinary RGB decode/sampling."""
import json
from pathlib import Path
import shutil
from .checkpoint_v1 import sha
from .provider_time_shift7_v1 import Provider as ShiftProvider
from .provider_time_v1 import ORDER
from .mobile_successor_v52 import RECIPE,PARENT,PLAN
from .gripper_time_shift7_v1 import VERSION,SHIFT,SLOTS
from .gripper_latent_v2 import VERSION as STATS_CODEC,EPSILON

SCHEMA='SLA_MOBILE_SHIFT7_SUCCESSOR_RGB_PACKAGE_V52'


def contract(m):
    b=m['bindings']
    if (m['schema']!=SCHEMA or m['codec']!=VERSION or m['decode_order']!=ORDER
            or m['step']!=500 or m['cursor']!=16000 or b['recipe']!=RECIPE
            or b['successor_parent_package']['sha256']!=PARENT
            or b['parent_cumulative_updates']!=1500 or b['parent_cumulative_visits']!=48000
            or b['successor_cumulative_updates']!=2000 or b['successor_cumulative_visits']!=64000
            or b['new_planned_updates']!=500 or b['new_planned_visits']!=16000
            or b['endpoint_lambda']not in(0.,.1)or b['endpoint_margin_m_s2']!=.2
            or b['endpoint_scale_m_s2']!=.25 or b['endpoint_tau_max']!=.5
            or b['goal_plan_sha']!=PLAN or not b['fresh_Adam']
            or b['codec_version']!=VERSION or b['statistics_codec_version']!=STATS_CODEC
            or b['inference_route']!='cached_actual' or m['parameter_slot_shift']!=SHIFT
            or m['parameter_slots']!=[[int(i)for i in g]for g in SLOTS]
            or m['previous_reference_required']is not True
            or any(m[k]is not False for k in ('teacher_required','optimizer_required','training_cache_required','physical_execution_admitted','endpoint_basis_applied'))):
        raise ValueError('EXPLICIT_FIXED_SHIFT7_SUCCESSOR_V52_LINEAGE_REQUIRED')


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
