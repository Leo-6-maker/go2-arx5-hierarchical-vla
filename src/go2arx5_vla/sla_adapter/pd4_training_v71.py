"""Matched PD4 views: exact frozen data, full-true FM32 and common row budget.

No partial supervision, consumed-mask weighting, new loss or dynamic admission.
A native-qualified statistic does not grant a student evaluation condition.
"""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import numpy as np
from .dense_dataset_v4 import DensePilotDataset,member,sha
from .codec_v1 import Quantiles
from .gripper_v2 import GripLogitCodec,VERSION,EPSILON
from .config_v1 import SEED,learning_rates
from .runtime_v4 import DensePilotEngine
from .runtime_v1 import read_pin
from .mixed_visits_v4 import Catalog,MixedVisits

from .pd4_config_v71 import OPERATION,MANIFESTS,STATISTICS,LAYOUTS


def native_qualification(pin):
    value=read_pin(pin)
    if (value['schema']!='SLA_PD4_NEW_STATS_PROSPECTIVE_NATIVE_ACTUAL_V7_1'
            or value['all_new_statistics_qualified']is not True
            or(value['planned_requests'],value['physical_requests'],value['terminal'],value['qualified'])!=(8,8,8,8)
            or value['optimizer_updates']!=0 or value['old_RP8_passed']is not False
            or {(v['view'],v['layout'])for v in value['outcomes']}!={(v,l)for v in STATISTICS for l in LAYOUTS}
            or any(not v['passed']or(v['queries'],v['consumed_points'])!=(61,640)for v in value['outcomes'])):
        raise ValueError('ALL_EIGHT_FRESH_PD4_STATS_NATIVE_QUALIFICATIONS_BEFORE_NEW_TRAINING')
    root=Path(pin['path']).parent;lease=json.loads((root/'control/LEASE.json').read_bytes())
    if lease['status']!='CLOSED' or lease['worker_exit_code']!=0:raise ValueError('TERMINAL_OWNED_PD4_QUALIFICATION_REQUIRED')
    for row in value['outcomes']:
        path=root/'runs'/row['native_run']/'native-01/RESULT.json'
        if sha(path)!=row['native_result_sha256']:raise ValueError('ORIGINAL_QUALIFICATION_NATIVE_RESULT_BYTES')
    return value


class PD4Dataset(DensePilotDataset):
    """Reuse the frame cache, input conversion and item functions, exact PD4 ABI."""
    def __init__(self,root,pin,*,resize_with_pad,max_cache_bytes=512*1024**2):
        self.root=Path(root);self.manifest_sha=pin;assert sha(self.root/'MANIFEST.private.json')==pin
        m=self.manifest=json.loads((self.root/'MANIFEST.private.json').read_bytes());view=m['dataset_id']
        if (view not in MANIFESTS or pin!=MANIFESTS[view]or m['schema']!='SLA_PROSPECTIVE_PD4_DUAL_VIEW_DATASET_V6_1'
                or m['windows']!=244 or len(m['episodes'])!=4 or tuple(e['layout']for e in m['episodes'])!=LAYOUTS
                or m['old_RP8_passed']is not False or m['PD20_complete']is not False
                or m['statistics']['sha256']!=STATISTICS[view]):
            raise ValueError('FIXED_PD4_FOUR_EPISODES_244_WINDOWS_AND_VIEW_STATS')
        p=member(self.root,m['statistics']['file']);assert sha(p)==STATISTICS[view];stats=json.loads(p.read_bytes())
        self.statistics={k:Quantiles(stats[k]['q01'],stats[k]['q99'])for k in ('state','actions')}
        self.resize=resize_with_pad;self.cache_limit=max_cache_bytes;self.cache=OrderedDict();self.cache_bytes=0
        self.labels={};self.frames={};self.codecs={};self.rows=[]
        from go2arx5_vla.world_model.task_learning_grasp_v1 import LANGUAGE
        for descriptor in m['episodes']:
            if descriptor['split']!='TRAIN' or descriptor['task']!='PICK_HOLD':raise ValueError('TRAIN_ONLY_PD4')
            p=member(self.root,descriptor['labels']['file']);assert sha(p)==descriptor['labels']['sha256']
            with np.load(p,allow_pickle=False)as z:data={k:z[k].copy()for k in z.files}
            if (data['state7'].shape!=(61,7)or data['latent7'].shape!=(61,40,7)
                    or data['state_eef10'].shape!=(61,10)or data['nominal_policy_H40'].shape!=(61,40,10)
                    or data['supervision_mask'].dtype!=np.bool_ or data['supervision_mask'].shape!=(61,40)
                    or not data['supervision_mask'].all() or data['consumed_mask'].shape!=(61,40)
                    or not np.array_equal(data['native_step'],5703+np.arange(61)*250)
                    or not np.array_equal(data['frame_index'],np.arange(61)*5)
                    or set(data['phase'])!={'PREGRASP','DESCEND','CLOSE','LIFT','HOLD'}):
                raise ValueError('FULL_TRUE40_PD4_MASKS_AND_ORIGINAL_CAPTURE_CLOCK')
            if any(not np.isfinite(data[k]).all()for k in ('state7','latent7','state_eef10','nominal_policy_H40')):
                raise ValueError('FINITE_PD4_PHYSICAL_AND_LATENT_ARRAYS')
            for shard in descriptor['frame_shards']:
                if sha(member(self.root,shard['file']))!=shard['sha256']:raise ValueError('ORIGINAL_PD4_RGB_BYTES')
            eid=descriptor['id'];episode=dict(descriptor,expert_stage_audit_only=data['phase'].tolist(),language=LANGUAGE)
            self.labels[eid]=data;self.frames[eid]=dict(T_policy_from_mount=descriptor['mount'])
            self.codecs[eid]=GripLogitCodec(descriptor['mount']);self.rows.extend((episode,a)for a in range(61))
        self.original=self


def fixed_probes(dataset):
    result=[];episodes=[r[0]for i,r in enumerate(dataset.rows)if i%61==0]
    def take(ep,a):
        index=next(i for i,(e,k)in enumerate(dataset.rows)if e['id']==ep['id']and k==a)
        if index in result:return False
        result.append(index);return True
    for j,stage in enumerate(('PREGRASP','DESCEND','CLOSE','LIFT','HOLD')):
        for n in range(2):
            ep=episodes[(2*j+n)%4]
            if not any(take(ep,a)for a,s in enumerate(ep['expert_stage_audit_only'])if s==stage):
                raise ValueError('SIXTEEN_FIXED_PD4_PROBE_COVERAGE')
    for j,(left,right)in enumerate(zip(('PREGRASP','DESCEND','CLOSE','LIFT'),('DESCEND','CLOSE','LIFT','HOLD'))):
        ep=episodes[j]
        if not any(take(ep,a)for a in range(60)if ep['expert_stage_audit_only'][a:a+2]==[left,right]):
            raise ValueError('PD4_FIXED_SEAM_COVERAGE')
    if not any(take(episodes[0],a)for a,s in enumerate(episodes[0]['expert_stage_audit_only'])if s=='CLOSE'):raise ValueError('DISTINCT_CLOSE_EVENT')
    if not take(episodes[0],57) or len(result)!=16:raise ValueError('FIXED_16_DISTINCT_PD4_PROBES')
    return result


class PD4Engine(DensePilotEngine):
    def dataset_for_spec(self,spec):
        from openpi_client.image_tools import resize_with_pad
        from .dense_recipe_v4 import microbatch_contract
        microbatch_contract(spec);native_qualification(spec['PD4_native_qualification'])
        audit=read_pin(spec['target_audit']);view=spec['PD4_view']
        if (audit['schema']!='SLA_PD4_TWO_VIEW_STATISTICS_AND_FP32_AUDIT_V6_1' or not audit['passes']
                or audit['views'][view]['windows']!=244 or audit['views'][view]['manifest_sha256']!=MANIFESTS[view]
                or audit['views'][view]['statistics_sha256']!=STATISTICS[view]):raise ValueError('ALL244_FROZEN_PD4_FP32_RAW_COMPILER_AUDIT')
        if spec['dense_manifest_sha256']!=MANIFESTS[spec['PD4_view']]:raise ValueError('EXACT_PD4_VIEW')
        if sha(spec['tokenizer']['path'])!=spec['tokenizer']['sha256']:raise ValueError('SAME_TOKENIZER')
        return PD4Dataset(spec['dataset'],spec['dense_manifest_sha256'],resize_with_pad=resize_with_pad,max_cache_bytes=spec['frame_cache_bytes'])

    def visits_for_dataset(self,dataset):
        # One explicit reference-view catalog supplies the SAME row indices for
        # both arms. Each arm still binds its own physical manifest separately.
        catalog=Catalog('T',MANIFESTS['PD4_EXECUTED_REF'],dataset.catalog_rows());planner=MixedVisits('T100_3K',{'T':catalog},seed=SEED)
        values=np.fromiter((v.row_index for step in range(500)for v in planner.batch(step)),dtype='<i8',count=16000)
        self.visit_bank_binding=dict(sampling='MATCHED_SHARED_REFERENCE_VIEW_TASK_EPISODE_PHASE_ANCHOR_CATALOG',catalog_sha256=catalog.sha256,seed=SEED,planned_target_visits=16000,planned_source_visits=0,student_phase_input=False,consumed_mask_used_for_sampling=False)
        self.selected_indices=fixed_probes(dataset)
        return values

    def learning_rate_values_for_spec(self,spec):return learning_rates()

    def __init__(self,spec,output):
        super().__init__(spec,output)
        self.bindings.update(recipe=OPERATION,PD4_view=spec['PD4_view'],native_statistics_qualification=spec['PD4_native_qualification']['sha256'],
            parent_host_cast_v7=spec['parent_host_cast_v7'],supervision_mask='ALL_TRUE_ORIGINAL_FM32',
            consumed_mask_used_for_loss=False,limited_PICK_HOLD_pilot=False,matched_PD4_fitting_only=True,
            selected_probe_indices=self.selected_indices,statistics_codec_metadata_source='PACKAGE_CODEC_VERSION_AND_EPSILON_WITH_FROZEN_PD4_QUANTILES')
