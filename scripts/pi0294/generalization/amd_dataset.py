"""GEN1 reader plus unchanged original126/C1 and C50 numerical sampling.

No old PV1 arm/population is granted by this reader. Native callers must bind
the explicit GEN1 request and original independent CORR2048 initialization.
"""
from functools import partial
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from . import amd_bank as bank,amd_pilot_scope as scope
from scripts.pi0294.mp1.dataset import SourceDataset
from scripts.pi0294.native_dataset import NativeDataset
from scripts.pi0294.pv1.sampler import MatchedSampler
from scripts.pi0294.pv1.mix1024 import lane_at as numerical_lane
from scripts.pi0294.scene_cards import digest

VISITS=1024*32


class Sampler(MatchedSampler):
    def __init__(self,dataset,*,start=0,stop=VISITS,seed=20261019,positioned=True):
        scope.need(type(start)is int and type(stop)is int and 0<=start<=stop<=VISITS
            and seed==20261019,'GEN_SAMPLER_ORIGINAL_SEED_AND_EXACT_CURSOR')
        # The mature constructor only indexes lanes/parents/legal anchors.
        # These transient names do not leave it as a request or recipe.
        support=SimpleNamespace(rows=dataset.rows,PV1_route='COVERAGE_FIRST',
            PV1_arm='B_COVERAGE',manifest_sha=dataset.manifest_sha)
        super().__init__(support,start=start,stop=stop,seed=seed,positioned=positioned)
        self.dataset=dataset;self.route='GEN1_LAYOUT_V0';self.arm=bank.IDENTITY

    def selection_at(self,cursor):
        if type(cursor)is not int or not self.start<=cursor<self.stop:raise IndexError('GEN1_EXACT_SAMPLE_CURSOR')
        task,lane=numerical_lane(cursor,'C50')
        lane_seed=int.from_bytes(hashlib.sha256(lane.encode()).digest()[:4],'big')
        rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([self.seed,cursor,lane_seed])))
        parents=sorted(self.bank[lane]);parent=parents[int(rng.integers(len(parents)))];windows=self.bank[lane][parent]
        anchor,index=windows[int(rng.integers(len(windows)))]
        return dict(task=task,lane=lane,parent=parent,anchor=anchor,index=index)

    def state_dict(self,consumed):
        scope.need(type(consumed)is int and self.start<=consumed<=self.stop and consumed%32==0,
            'GEN_SAMPLER_COMMITTED_CURSOR_ONLY')
        return dict(algorithm='GEN1_FROZEN_C50_NUMERICAL_LANE_PARENT_WINDOW_V1',
            training_identity=bank.IDENTITY,route=self.route,seed=self.seed,cursor=consumed,
            stop=self.stop,manifest_sha256=self.dataset.manifest_sha,support_sha256=self.support_sha256,
            metadata_not_model_input=True)


class Dataset(SourceDataset):
    def __init__(self,*args,new_root,new_raw_bindings,packet_bytes,**kwargs):
        super().__init__(*args,**kwargs);self.base_rows=len(self.rows);base_manifest=self.manifest_sha
        self.new_reader=NativeDataset(new_root,self.source,self.cosmos,for_optimization=True,raw_bindings=new_raw_bindings)
        self.support=bank.validate(self.new_reader.manifest,packet_bytes)
        scope.need(type(new_raw_bindings)is dict
            and set(new_raw_bindings)==set(self.support['admitted']),
            'GEN_READER_EXPLICIT_ALL_REGISTERED_RAW_PARENTS')
        from scripts.pi0292.durable import read
        from .amd_export import exact_card
        for row in self.new_reader.manifest['episodes']:
            raw=Path(new_raw_bindings[row['id']])
            scope.need(raw.is_absolute()and raw.is_dir()and not raw.is_symlink()
                and raw.stat().st_uid==os.getuid(),'GEN_READER_OWN_REAL_RAW_INPUT_DIRECTORY')
            card=exact_card(read(raw/'CARD.json'),packet_bytes)
            scope.need(digest(card)==row['source_card_sha256'],'GEN_READER_SAME_SEALED_NATIVE_CARD')
        scope.need(self.new_reader.statistics_sha==self.statistics_sha==scope.a.STATS,
            'GEN_READER_SAME_FROZEN_R0')
        metadata={};original=[]
        for i,(row,anchor)in enumerate(self.rows):
            lane='original_'+row['task']if i<self.original_rows else'C1_T1'
            scope.need(i<self.original_rows or row['task']=='T1','GEN_ORIGINAL_C1_T1_ONLY')
            key=(lane,row['id']);metadata.setdefault(key,dict(row,PV1_lane=lane));original.append((metadata[key],anchor))
        parents={r.get('source_student_episode')for r,_ in original if r['PV1_lane']=='C1_T1'}
        scope.need(len(parents)==4 and None not in parents,'GEN_SAME_FOUR_ORIGINAL_C1_STUDENT_PARENTS')
        new=[]
        for row,anchor in self.new_reader.rows:
            key=('new_'+row['task'],row['id']);metadata.setdefault(key,dict(row,PV1_lane=key[0]));new.append((metadata[key],anchor))
        self.rows=original+new;self.manifest_sha=digest(dict(identity=bank.IDENTITY,
            original_and_C1=base_manifest,new_GEN_bank=self.new_reader.manifest_sha,
            support=self.support,statistics_sha256=self.statistics_sha))
        self.make_sampler=partial(Sampler)

    def raw(self,index):
        return super().raw(index)if index<self.base_rows else self.new_reader.raw(index-self.base_rows)

    def identity(self,index):
        value=super().identity(index);row,_=self.rows[index]
        value.update(PV1_lane=row['PV1_lane'],study_identity=bank.IDENTITY,
            scene_group=row.get('scene_group'),source_student_episode=row.get('source_student_episode'),
            nominal_clearance_m=row.get('nominal_clearance_m'))
        return value
