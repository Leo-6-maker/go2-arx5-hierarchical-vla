"""Distinct continuous-task data, using the unchanged official target Processor."""
import numpy as np
from . import task_learning_grasp_v1 as task
from .task_learning_dataset_v1 import exposure_schedule
from .task_learning_export_v1 import SCHEMA,DOMAIN,PROFILE
from .target_gr00t_dataset_v1 import TargetNativeDataset


def sample_rows(episodes,split,for_optimization):
    task.need(split in('TRAIN','DEV')and(not for_optimization or split=='TRAIN'),'DEV_NEVER_OPTIMIZER_INPUT')
    result=[]
    for row in episodes:
        if row['split']!=split:continue
        valid=row.get('supervised_anchors');stages=row.get('expert_stage_audit_only')
        task.need(type(valid)is list and bool(valid)and valid==sorted(set(valid))
            and all(type(v)is int and 0<=v<15 for v in valid)
            and type(stages)is list and len(stages)==15,
            'REAL_UNIQUE_CURRENT_DECISIONS_NOT_DENSE_COMMANDS_OR_OVERSAMPLING')
        if split=='TRAIN':
            mask=np.isin(np.arange(15),valid)
            expected=exposure_schedule(dict(valid_window=mask,expert_stage_audit_only=np.asarray(stages)))
            task.need(row.get('optimization_anchors')==expected and len(valid)in(14,15)
                and row.get('real_policy_decisions')==15 and row.get('supervised_policy_decisions')==len(valid),
                'DECLARED_EQUAL_EPISODE32_VISITS_QUARTER_BOUNDARIES_AND_INTERVENTION_EXCLUSION')
        selected=row['optimization_anchors']if for_optimization else valid
        result.extend((row,anchor)for anchor in selected)
    return result


class TaskLearningNativeDataset(TargetNativeDataset):
    MANIFEST_SCHEMA=SCHEMA
    MANIFEST_DOMAIN=DOMAIN
    MANIFEST_PROFILE=PROFILE

    def validate_manifest(self):
        super().validate_manifest();m=self.manifest;s=m.get('population')or{}
        train=[r for r in m.get('episodes',[])if r.get('split')=='TRAIN']
        task.need(m.get('bundle_kind')in('CONTINUOUS_TRAIN_BASE','FORMAL_TASK_LEARNING_V1')
            and s.get('complete')is True and s.get('passes')is True
            and s.get('requested')==s.get('attempted')==100
            and s.get('task_successes')==len(train)>=80
            and s.get('successful_layouts')==len({r['parent']for r in train})>=20
            and s.get('genuine_recovery_successes')==sum(r.get('genuine_recovery_success')is True for r in train)>=20
            and s.get('recovery_layouts')==len({r['parent']for r in train
                if r.get('genuine_recovery_success')is True})>=10
            and s.get('hard_safety_stop_conditions')==s.get('technical_failure_conditions')==0
            and m.get('TRAIN_only_statistics')is True and m.get('statistics_fit_before_oversampling')is True
            and m.get('interventions_masked')is True
            and m.get('construction_or_qualification_relabelled')is False and m.get('TEST_results_read')is False,
            'SEALED_FULL_CONTINUOUS_TRAIN_POPULATION_BEFORE_ANY_TARGET_PROCESSOR')
        task.need(len({r['id']for r in m['episodes']})==len(m['episodes'])
            and all(r['split']in('TRAIN','DEV')for r in m['episodes']),
            'UNIQUE_EPISODES_AND_NO_TEST_DURING_DEVELOPMENT')
        if self.for_optimization:
            dev=[r for r in m['episodes']if r['split']=='DEV']
            task.need(m.get('bundle_kind')=='FORMAL_TASK_LEARNING_V1'
                and len(dev)==4 and len({r['parent']for r in dev})==4
                and not({r['parent']for r in dev}&{r['parent']for r in train})
                and all(r.get('purpose')=='DEV_REFERENCE_NOT_BC'for r in dev),
                'FIXED_NEW_DEV_REFERENCES_AND_ACTUAL_PROCESSOR_REQUIRED_BEFORE_SFT')

    def initialize(self):
        if self.for_optimization:
            import json
            proof=json.loads((self.root/'PROCESSOR_TRAINING_ADMISSION.json').read_bytes())
            task.need(proof.get('dataset_family')=='TASK_LEARNING_CONTINUOUS_V1',
                'ACTUAL_CONTINUOUS_DATASET_PROCESSOR_NOT_LEGACY_SHORT_ADMISSION')
        return super().initialize()

    def select_rows(self):
        return sample_rows(self.manifest['episodes'],self.split,self.for_optimization)

    def raw(self,index):
        row,anchor=self.rows[index]
        result=super().raw(index)
        with np.load(self.root/row['file'],allow_pickle=False)as z:
            valid=np.flatnonzero(z['valid_window']).tolist();stages=z['expert_stage_audit_only'].tolist()
        task.need(valid==row['supervised_anchors']and stages==row['expert_stage_audit_only']
            and result[3]==task.LANGUAGE,'PACKET_MASK_STAGE_AND_ORIGINAL_TASK_LANGUAGE_MATCH_MANIFEST')
        return result
