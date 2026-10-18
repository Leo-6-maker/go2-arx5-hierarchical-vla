"""Keep the mature original/C1 readers; append one frozen admitted PV1 bank."""
from functools import partial

from scripts.pi0294.mp1.dataset import SourceDataset
from scripts.pi0294.native_dataset import NativeDataset
from scripts.pi0294.e1.scope import STATS
from scripts.pi0294.scene_cards import digest
from .sampler import ARMS, MatchedSampler
from .plan import COUNTS

POPULATION = '288be5e878baf1b112190a8ac69c1b507336ca03f550ae9b4f4672f13ef90db1'
TRAIN_GROUPS = {f'PI0294/PV1/TRAIN/{family}/{i:03d}':('-Y' if family=='F1' and i%2 else '+Y')
    for family,count in COUNTS['TRAIN'].items() for i in range(count)}


def admit_new_support(manifest,route):
    if (route not in ARMS or manifest.get('schema')!='PI0294_PV1_ADMITTED_TRAIN_BANK_V1'
            or manifest.get('domain')!='FIXED_BASE_ARX5_TASK_SIM'
            or manifest.get('statistics_sha256')!=STATS or manifest.get('statistics_refit') is not False
            or manifest.get('population_plan_sha256')!=POPULATION
            or manifest.get('elementwise_codec_admission')!='PASS_FIXED_ORIGINAL_V3_0_NO_SILENT_CLIP'
            or manifest.get('evaluation_rows_used_for_training') is not False):
        raise PermissionError('PV1_FROZEN_ADMITTED_BANK_AND_ORIGINAL_CODEC_REQUIRED')
    bank={('T1',.004):{},('T2',.004):{},('T2',.012):{}}
    ids=set()
    for row in manifest['episodes']:
        group=row.get('scene_group','')
        key=(row['task'],row['nominal_clearance_m'])
        if (row['id'] in ids or key not in bank or row['split']!='TRAIN'
                or group not in TRAIN_GROUPS or group in bank[key]
                or row.get('full_task_success') is not True or row.get('independent_raw_admitted') is not True
                or row['windows']!=601 or row['direction']!=TRAIN_GROUPS[group]):
            raise PermissionError('PV1_ONLY_UNIQUE_FULL_SUCCESS_TRAIN_PARENTS_WITH_ACTUAL_H40_LABELS')
        bank[key][group]=row
        ids.add(row['id'])
    t1=bank['T1',.004]
    t2=bank['T2',.004]
    common=sorted(set(t2)&set(bank['T2',.012]))
    if (len(t1)<32 or len(t2)<64
            or any(sum(r['direction']==d for r in t2.values())<16 for d in ('+Y','-Y'))):
        raise PermissionError('PV1_MINIMUM_INDEPENDENT_SUPPORT_NOT_MET')
    if route=='CLEARANCE_PAIR' and len(common)<64:
        raise PermissionError('PV1_MARGIN_PAIR_NEEDS_64_COMMON_SUCCESS_GROUPS')
    if route=='CLEARANCE_PAIR' and any(sum(t2[g]['direction']==d for g in common)<16 for d in ('+Y','-Y')):
        raise PermissionError('PV1_COMMON_T2_SUPPORT_MUST_RETAIN_BOTH_DIRECTION_MINIMA')
    if route=='CLEARANCE_PAIR' and any(t2[g]['direction']!=bank['T2',.012][g]['direction'] for g in common):
        raise PermissionError('PV1_PAIR_MUST_HAVE_SAME_SCENE_DIRECTION')
    return dict(T1_groups=sorted(t1),original_T2_groups=sorted(t2),common_T2_groups=common,
        independent_groups_not_window_counts=True)


class PV1Dataset(SourceDataset):
    def __init__(self,*args,new_root,new_raw_bindings,route,arm,seed=20261019,**kwargs):
        if route not in ARMS or arm not in ARMS[route] or seed not in (20261019,20261020):
            raise PermissionError('PV1_FIXED_ROUTE_ARM_AND_SEED')
        super().__init__(*args,**kwargs)
        self.PV1_route,self.PV1_arm=route,arm
        self.base_rows=len(self.rows)
        base_manifest=self.manifest_sha
        source,cosmos=self.source,self.cosmos
        self.new_reader=NativeDataset(new_root,source,cosmos,for_optimization=True,raw_bindings=new_raw_bindings)
        self.support=admit_new_support(self.new_reader.manifest,route)
        if self.new_reader.statistics_sha!=self.statistics_sha:
            raise PermissionError('PV1_NEW_BANK_CANNOT_REFIT_ORIGINAL_NORMALIZER')
        metadata={}
        original=[]
        for i,(row,anchor) in enumerate(self.rows):
            lane=('original_'+row['task']) if i<self.original_rows else 'C1_T1'
            if i>=self.original_rows and row['task']!='T1':
                raise PermissionError('PV1_ONLY_FROZEN_EXISTING_T1_C1')
            key=(lane,row['id'])
            metadata.setdefault(key,dict(row,PV1_lane=lane))
            original.append((metadata[key],anchor))
        parents={r.get('source_student_episode') for r,_ in original if r['PV1_lane']=='C1_T1'}
        if len(parents)!=4 or None in parents:
            raise PermissionError('PV1_REPLAY_EXACT_EXISTING_FOUR_C1_STUDENT_PARENTS')
        new=[]
        self.new_indices=[]
        for index,(row,anchor) in enumerate(self.new_reader.rows):
            if row['task']=='T2':
                margin=.012 if arm=='B_12MM' else .004
                if row['nominal_clearance_m']!=margin:
                    continue
                if route=='CLEARANCE_PAIR' and row['scene_group'] not in self.support['common_T2_groups']:
                    continue
            key=('new_'+row['task'],row['id'])
            metadata.setdefault(key,dict(row,PV1_lane=key[0]))
            new.append((metadata[key],anchor))
            self.new_indices.append(index)
        self.rows=original+new
        self.manifest_sha=digest(dict(original_and_C1=base_manifest,new_bank=self.new_reader.manifest_sha,
            route=route,arm=arm,support=self.support,statistics_sha256=self.statistics_sha))
        self.make_sampler=partial(MatchedSampler,seed=seed)

    def raw(self,index):
        if index<self.base_rows:return super().raw(index)
        return self.new_reader.raw(self.new_indices[index-self.base_rows])

    def identity(self,index):
        value=super().identity(index)
        row,_=self.rows[index]
        value.update(PV1_lane=row['PV1_lane'],scene_group=row.get('scene_group'),
            source_student_episode=row.get('source_student_episode'),nominal_clearance_m=row.get('nominal_clearance_m'))
        return value
