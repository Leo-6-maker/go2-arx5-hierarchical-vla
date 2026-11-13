"""Source-only BOOK first fit using the existing resident RGB/FM/Adam engine."""
import numpy as np

from .checkpoint_v1 import sha
from .config_v1 import SEED,learning_rates
from .gripper_v2 import VERSION,EPSILON
from .mixed_visits_v4 import Catalog
from .runtime_v1 import read_pin
from .runtime_v3 import DeviceResidentEngine
from .source_book_v5 import DATASET_ID,SourceBookDataset

RECIPE='SLA_BOOK_FULL21_SOURCE_COMMAND_FM32_500_V5'
OPERATION='SOURCE_BOOK21_FM_500'
PROBE_PHASES=('approach','close','verify_grasp','lift','preinsert','release','withdraw','hold')


def book_probe_indices(dataset):
    result=[]
    for phase in PROBE_PHASES:
        found=next((i for i,(ep,a)in enumerate(dataset.rows)if ep['expert_stage_audit_only'][a]==phase),None)
        if found is None:raise ValueError('COMPLETE_SOURCE_BOOK_GRASP_RELEASE_WITHDRAW_HOLD_PROBES_REQUIRED')
        result.append(found)
    if len(set(result))!=8:raise ValueError('EIGHT_DISTINCT_SOURCE_BOOK_RGB_PROBES')
    return result


def book_probes(engine,name):
    from scripts.sla.train_v1 import probes
    return probes(engine,name,indices=book_probe_indices(engine.dataset))


class SourceBookEngine(DeviceResidentEngine):
    def dataset_for_spec(self,spec):
        from openpi_client.image_tools import resize_with_pad
        audit=read_pin(spec['target_audit'])
        if (audit['schema']!='SLA_BOOK21_ALL_WINDOW_RGB_READBACK_V5' or not audit['passes']
                or audit['read_windows']!=1281 or audit['episodes']!=21 or audit['dataset_id']!=DATASET_ID
                or audit['manifest_sha256']!=spec['dense_manifest_sha256']
                or audit['target_WBC_qualified']is not False or spec.get('action_representation_audit')
                or sha(spec['tokenizer']['path'])!=spec['tokenizer']['sha256']):
            raise ValueError('ALL_SOURCE_BOOK21_RGB_COMMAND_READBACK_REQUIRED')
        dataset=SourceBookDataset(spec['dataset'],spec['dense_manifest_sha256'],resize_with_pad=resize_with_pad)
        stats=read_pin(dict(path=str(dataset.root/dataset.manifest['statistics']['file']),sha256=dataset.manifest['statistics']['sha256']))
        if stats.get('codec_version')!=VERSION or stats.get('epsilon')!=EPSILON:
            raise ValueError('BOOK_NORMALIZER_MUST_CARRY_ORDINARY_PROVIDER_CODEC_METADATA')
        book_probe_indices(dataset)
        return dataset

    def visits_for_dataset(self,dataset):
        catalog=Catalog('S_cmd',dataset.manifest_sha,dataset.catalog_rows())
        values=np.fromiter((catalog.sample(SEED,i).row_index for i in range(16000)),dtype='<i8',count=16000)
        self.visit_bank_binding=dict(catalog_sha256=catalog.sha256,seed=SEED,
            sampling='SOURCE_TASK_EPISODE_PHASE_ANCHOR',student_phase_input=False,planned_target_visits=0,planned_source_visits=16000)
        return values

    def learning_rate_values_for_spec(self,spec):return learning_rates()

    def __init__(self,spec,output):
        super().__init__(spec,output)
        self.bindings.update(recipe=RECIPE,codec_version=VERSION,dataset_id=DATASET_ID,
            dense_manifest_sha256=self.dataset.manifest_sha,
            derived_statistics_sha256=self.dataset.manifest['statistics']['sha256'],
            source_only=True,target_WBC_qualified=False,visit_bank=self.visit_bank_binding)
