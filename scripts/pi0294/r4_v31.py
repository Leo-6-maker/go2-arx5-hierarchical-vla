"""Existing V31-only TRAIN view with original V30 statistics, never refitted."""
import argparse
import copy
import collections
import os
from pathlib import Path
import sys
import time

from scripts.pi0292.durable import read,write
from .native_dataset import sha,official_import
from .cpu_pipeline import verify_frozen_bank
from .r4 import cpu_stage


def run(args):
    root=Path(args.root);v30,v31=Path(args.v30),Path(args.v31)
    with cpu_stage(root,'V31_INCREMENTAL_VIEW',cores=4,cpu_s_cap=7200):
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('R4_CPU_CUDA_HIDDEN_REQUIRED')
        before=time.time();banks=[]
        for folder in (v30,v31):
            manifest=read(folder/'DATASET_MANIFEST.json')
            banks.append(verify_frozen_bank(folder,sha(folder/'BANK_SEAL.json'),sha(folder/'DATASET_MANIFEST.json'),sha(folder/'statistics.json')))
        qualification=v31.parents[2]/'evidence/bank-qualification-v3_1-02/CPU_QUALIFICATION.json'
        qualified=read(qualification)
        if qualified['status']!='PASS_PI0294_TRAIN_DEV_AND_TRAINER_LOADER_CPU' or qualified['manifest_sha256']!=sha(v31/'DATASET_MANIFEST.json'):
            raise ValueError('ACTUAL_V31_PRIOR_CPU_ADMISSION_REQUIRED')
        old={r['id']:r for r in banks[0]['episodes']}
        for r in banks[1]['episodes']:
            if r['id'] in old:
                previous=old[r['id']]
                if r['sha256']!=previous['sha256'] or r['split']!=previous['split']:raise ValueError('V31_OLD_EPISODE_BYTES_OR_SPLIT_CHANGED')
        novel=[r for r in banks[1]['episodes'] if r['split']=='TRAIN' and r['id'] not in old]
        counts=dict(collections.Counter(r['task'] for r in novel))
        if len(novel)!=122 or counts!={'T1':61,'T2':61}:raise ValueError('ACTUAL_V31_INCREMENTAL_TRAIN_POPULATION_CHANGED')
        output=root/'data/v31_incremental_v30_stats';output.mkdir(exist_ok=False)
        episodes=[];members=[];total_bytes=0
        for row in novel:
            p=v31/row['file'];assert p.is_file() and not p.is_symlink() and sha(p)==row['sha256']
            target=output/row['file'];target.parent.mkdir(parents=True,exist_ok=True);os.link(p,target)
            episodes.append(copy.deepcopy(row));members.append(dict(relative=row['file'],bytes=p.stat().st_size,sha256=row['sha256']))
            total_bytes+=p.stat().st_size
        (output/'statistics.json').write_bytes((v30/'statistics.json').read_bytes())
        manifest=copy.deepcopy(banks[1]);manifest.update(version='V31_ONLY_TRAIN_V30_STATS_R4',episodes=episodes,
            statistics_sha256=sha(output/'statistics.json'),statistics_fit_count=0,purpose='R4_INCREMENTAL_TRAIN_CONSUMER_VIEW',
            source_v31_manifest_sha256=sha(v31/'DATASET_MANIFEST.json'),source_v30_manifest_sha256=sha(v30/'DATASET_MANIFEST.json'),
            original_banks_untouched=True,DEV_refit=False,not_original_V31_statistics=True)
        write(output/'DATASET_MANIFEST.json',manifest)
        write(root/'evidence/V31_INCREMENTAL_IDENTITY.json',dict(status='122_TRAIN_EPISODES_EXACT_ORIGINAL_BYTES',counts=counts,
            view_manifest_sha256=sha(output/'DATASET_MANIFEST.json'),members=members,numeric_payload_bytes=total_bytes,
            original_V30_and_V31_seals_verified=True,original_stats_sha256=sha(v30/'statistics.json'),new_samples=0,new_physics=0))
        official_import(args.source)
        import torch
        torch.set_num_threads(1)
        from scripts.pi0292.train_adapter import build_processor
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi0291_arx5_data'))
        from scripts.pi0291_arx5_data.qualify_loader import codec_all_windows
        processor=build_processor(args.source,args.cosmos,read(output/'statistics.json'))
        try:
            codec=codec_all_windows(output,processor,splits=('TRAIN',))
        except ValueError as error:
            write(root/'evidence/V31_V30_STATS_CODEC.json',dict(status='DATA_VIEW_BLOCKED_ORIGINAL_CODEC_GATE',error=str(error),
                original_statistics_unchanged=True,new_neural_queries=0,new_physics=0,stat_refit=0,action_correction=0))
            print('V31_INCREMENTAL_CODEC_BLOCKED_C0_INDEPENDENT',flush=True)
            return
        if torch.cuda.is_initialized():raise PermissionError('CPU_QUALIFICATION_INITIALIZED_CUDA')
        write(root/'evidence/V31_V30_STATS_CODEC.json',dict(status='PASS_ORIGINAL_V30_STATS_INCREMENTAL_TRAIN_CODEC',codec=codec,
            views_created_without_refitting=True,new_neural_queries=0,new_physics=0,actual_elapsed_s=time.time()-before))
        print('V31_INCREMENTAL_122_TRAIN_CODEC_PASS_PENDING_CONSUMER_DELIVERY',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','v30','v31','source','cosmos'):p.add_argument('--'+n,required=True)
    run(p.parse_args())
