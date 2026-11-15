"""Variable-length sealed mobile episodes over the qualified FM32 learner."""
from collections import OrderedDict
import json
import math
from pathlib import Path
import numpy as np
from .dense_dataset_v4 import DensePilotDataset, member, sha
from .codec_v1 import Quantiles
from .gripper_v2 import GripLogitCodec, VERSION, EPSILON
from .config_v1 import SEED
from .runtime_v1 import read_pin, probe_indices
from .runtime_v3 import DeviceResidentEngine
from .mixed_visits_v4 import Catalog, MixedVisits

RECIPE='MFREE_NESTED_FINAL_FM1000_V1'


def learning_rates():
    # Same one-based clock as the existing 500-update recipe.
    return [5e-5*j/50 if j<=50 else 1e-5+.5*(5e-5-1e-5)*(1+math.cos(math.pi*(j-50)/950))
        for j in range(1,1001)]


class SnapshotDataset(DensePilotDataset):
    def __init__(self,root,pin,*,resize_with_pad,max_cache_bytes=1024**3):
        self.root=Path(root);self.manifest_sha=pin
        if sha(self.root/'MANIFEST.private.json')!=pin:raise ValueError('SEALED_MOBILE_TRAIN_MANIFEST')
        m=self.manifest=json.loads((self.root/'MANIFEST.private.json').read_bytes())
        if (m['schema']!='MFREE_NESTED_LIVE_TRAIN_DATASET_V1' or m['state']!='committed'
                or m['codec']!=VERSION or m['split']!='TRAIN' or m['phase_input_to_student']
                or m['teacher_memory_input'] or m['BASE_head'] or m['unconsumed_suffix_is_executed']
                or not m['plan_supervision_suffix'] or not 4<=len(m['episodes'])<=100):
            raise ValueError('EXPLICIT_NEW_LIVE_TRAIN_PLAN_SUPERVISION_ONLY')
        stat=m['statistics'];path=member(self.root,stat['file'])
        if sha(path)!=stat['sha256']:raise ValueError('FROZEN_MOBILE_TRAIN_STATS')
        stats=json.loads(path.read_bytes());self.statistics={k:Quantiles(stats[k]['q01'],stats[k]['q99'])for k in ('state','actions')}
        if (stats['codec_version']!=VERSION or stats['epsilon']!=EPSILON or stats['split']!='TRAIN'
                or stats['state_gripper_transformed'] or stats['fit_episodes']!=len(m['episodes'])
                or stats['fit_unique_windows']!=m['windows']):raise ValueError('EXACT_NEW_STATS_TRAIN_POPULATION')
        self.resize=resize_with_pad;self.cache_limit=max_cache_bytes;self.cache=OrderedDict();self.cache_bytes=0
        self.labels={};self.frames={};self.codecs={};self.rows=[]
        for episode in m['episodes']:
            n=episode['windows'];eid=episode['id'];path=member(self.root,episode['labels']['file'])
            if sha(path)!=episode['labels']['sha256'] or episode['split']!='TRAIN':raise ValueError('SEALED_SUCCESSFUL_TRAIN_LABELS')
            with np.load(path,allow_pickle=False)as z:data={k:z[k].copy()for k in z.files}
            if (not 1<=n<=120 or data['state7'].shape!=(n,7) or data['latent7'].shape!=(n,40,7)
                    or data['state_eef10'].shape!=(n,10) or data['nominal_policy_H40'].shape!=(n,40,10)
                    or data['supervision_mask'].shape!=(n,40) or data['supervision_mask'].dtype!=np.bool_
                    or not data['supervision_mask'].all() or data['consumed_mask'].dtype!=np.bool_
                    or not np.array_equal(data['consumed_mask'],np.tile([True]*10+[False]*30,(n,1)))
                    or not np.array_equal(data['actual_consumed_ticks'],np.full(n,250))
                    or not np.array_equal(data['native_step'],5703+np.arange(n)*250)
                    or not np.array_equal(data['frame_index'],np.arange(n))
                    or set(data['phase'])!={'PREGRASP','DESCEND','CLOSE','LIFT','HOLD'}):
                raise ValueError('COMPLETE_VARIABLE_LENGTH_LIVE_H40_P10_AND_FIVE_PHASES')
            for key in ('state7','latent7','state_eef10','nominal_policy_H40'):
                if not np.isfinite(data[key]).all():raise ValueError('FINITE_MOBILE_TRAIN_ARRAYS')
            for shard in episode['frame_shards']:
                if sha(member(self.root,shard['file']))!=shard['sha256']:raise ValueError('SEALED_REAL_RGB_SHARDS')
            self.labels[eid]=data;self.frames[eid]=episode['mount'];self.codecs[eid]=GripLogitCodec(episode['mount']['T_policy_from_mount'])
            self.rows.extend((episode,a)for a in range(n))
        if len(self.rows)!=m['windows']:raise ValueError('TRUE_MOBILE_WINDOW_DENOMINATOR')
        self.original=self


class SnapshotEngine(DeviceResidentEngine):
    def dataset_for_spec(self,spec):
        from openpi_client.image_tools import resize_with_pad
        audit=read_pin(spec['target_audit'])
        if (spec['operation']!=RECIPE or spec['microbatch']!=8 or spec['training_matmul_precision']!='highest'
                or spec.get('action_representation_audit')
                or audit['schema']!='MFREE_TRAIN_FP32_DEPLOYMENT_AUDIT_V1' or not audit['passes']
                or audit['manifest_sha256']!=spec['dense_manifest_sha256']
                or sha(spec['tokenizer']['path'])!=spec['tokenizer']['sha256']):
            raise ValueError('NEW_MOBILE_INIT_FM32_AND_FROZEN_FP32_DATA_AUDIT')
        return SnapshotDataset(spec['dataset'],spec['dense_manifest_sha256'],resize_with_pad=resize_with_pad,max_cache_bytes=spec['frame_cache_bytes'])

    def visits_for_dataset(self,dataset):
        catalog=Catalog('T',dataset.manifest_sha,dataset.catalog_rows())
        planner=MixedVisits('T100_3K',{'T':catalog},seed=SEED)
        self.visit_bank_binding=dict(catalog_sha256=catalog.sha256,seed=SEED,
            sampling='EXISTING_TASK_EPISODE_PHASE_ANCHOR_BANK',planned_target_visits=32000,
            planned_source_visits=0,student_phase_input=False,consumed_mask_used_for_sampling=False)
        self.selected_indices=probe_indices(dataset)
        return np.fromiter((v.row_index for step in range(1000)for v in planner.batch(step)),dtype='<i8',count=32000)

    def learning_rate_values_for_spec(self,spec):return learning_rates()

    def __init__(self,spec,output):
        super().__init__(spec,output)
        self.bindings.update(recipe=RECIPE,codec_version=VERSION,parent_host_cast_v7=True,
            derived_statistics_sha256=self.dataset.manifest['statistics']['sha256'],
            dense_manifest_sha256=self.dataset.manifest_sha,dataset_id=self.dataset.manifest['dataset_id'],
            visit_bank=self.visit_bank_binding,selected_probe_indices=self.selected_indices,
            supervision_mask='ALL_TRUE40_CURRENT_QUERY_CHECKED_FINAL_PLAN',consumed_mask_used_for_loss=False,
            boundary_version='MOBILE_ACCEPTED_BOUNDARY_050_V1',BASE_head=False,
            spatial_loss=False,reference_error_input=False,teacher_memory_input=False,
            state_token_dropout=.25,old_RP8_passed=False,snapshot_id=spec['snapshot_id'],snapshot_descriptor_sha256=spec['snapshot_descriptor']['sha256'])
