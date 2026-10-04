"""Freeze the authorized fitting and evaluation membership; no neural calls."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

BASE = Path('/home/liuyu-nfs/Projects/go2-arx5')
ROOT = BASE/'PI0272'
RAW = Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0272')
HOPPER = Path('/scratch/e1192548/Projects/go2-arx5/experiments')
RELEASE = 'e62e6a469dce1f4d3df5d49b38bad4a3c627e969'
LATE_SHA = '2899f730ef8f3fd1a76350662ec284237439ab7104192d9d6dfe2c30c503528e'
NEWFIT_SHA = '553d254cea5ad5c32d0d8acf615f2abfe222459ac10e9d0bb0e09b786884d226'
STATS_SHA = '0ea96eaf9ba6f2bfb5c16681542723e9ee0e82281dd07c4e33ed9189799580e9'


def seed_for(update):
    assert 1 <= update <= 1024
    return int.from_bytes(hashlib.sha256(f'PI0272:TRAIN:27202:{update}'.encode()).digest()[:8], 'big') % 2**31


def schedule(rows):
    ordered = sorted(rows, key=lambda r: (r['source_mapping']['unit'], r['source_mapping']['micro']))
    assert len(ordered) == len({r['input_id'] for r in ordered}) == 296
    stream = [dict(ordinal=n, unit=n//2+1, micro=n%2, input_id=ordered[n%296]['input_id'],
                   seed=seed_for(n//2+1)) for n in range(2048)]
    counts = Counter(r['input_id'] for r in stream)
    assert Counter(counts.values()) == {7:272, 6:24}
    return ordered, stream


def main():
    sys.path.insert(0, str(BASE/'PI0270/code'))
    from admission import read, sha, write
    closed = BASE/'PI0271/evidence/final-publication/FINAL_CLOSEOUT_RECEIPT.json'
    closeout = read(closed)
    assert closeout['execution'] == 'COMPLETE' and closeout['missing_scientific_requirements'] == []
    assert closeout['canonical']['id'] == 5978753654
    assert read(BASE/'PI0271/evidence/FINAL_PUBLICATION_HF_READBACK.json')['status'] == 'PASS_FIXED_FINAL_PUBLICATION_ALL_FILES_READBACK'
    late_path = BASE/'PI0271/design/LATE_SUPPORT_CARD.json'
    newfit_path = BASE/'PI0270/design/NEW_FIT192.json'
    assert sha(late_path) == LATE_SHA and sha(newfit_path) == NEWFIT_SHA
    late, newfit = read(late_path), read(newfit_path)
    assert late['statistics_sha256'] == newfit['statistics_sha256'] == STATS_SHA
    ordered, stream = schedule(late['rows'])
    counts = Counter(r['source_mapping']['bucket'] for r in ordered)
    assert counts == dict(ORIGINAL_Q2=96, ORIGINAL_Q3=96, M1_TERMINAL=32,
                          M3_ENTRY=24, M3_MIDARC=24, M3_TERMINAL=24)
    assert Counter(r['source_mapping']['task'] for r in ordered) == dict(M1=128, M3=168)
    assert len({r['source_mapping']['family'] for r in ordered}) == 24
    for row in ordered:
        m = row['source_mapping']
        assert row['actual_AGG_consumption']['actual_seen']
        assert (m['unit'],m['micro']) == (row['actual_AGG_consumption']['unit'],row['actual_AGG_consumption']['micro'])
        assert sha(row['input']) == row['input_sha256'] == m['sample_sha256']
        assert sha(m['source_codec_path']) == m['source_codec_sha256']
        codec = read(m['source_codec_path'])
        assert codec['eligible'] and codec['roundtrip']['PASS']
        assert row['language'] == m['language'] and row['condition_sha256'] == m['condition_sha256']
        assert row['labels_masks_sha256'] == m['labels_masks_sha256']
        row['hopper_input'] = str(HOPPER/'PI0270/source/recovery'/m['source_uid']/'sample.npz')
    by_id = {r['input_id']:r for r in ordered}
    exposure = Counter((by_id[r['input_id']]['source_mapping']['task'],
                        by_id[r['input_id']]['source_mapping']['command'],
                        by_id[r['input_id']]['source_mapping']['bucket']) for r in stream)
    membership = {(r['condition_sha256'],r['labels_masks_sha256']) for r in ordered}
    overlap = [r['input_id'] for r in newfit['rows'] if (r['condition_sha256'],r['labels_masks_sha256']) in membership]
    card = dict(schema='PI0272_FROZEN_FITTING296_V1', release=RELEASE, source_late_card_sha256=LATE_SHA,
                rows=ordered, stream=stream, training_seed=27202, updates_per_arm=1024,
                actual_Hopper_encoding_still_required=True, actual_training_consumption_not_yet_observed=True,
                NEW_FIT192_exact_processed_condition_and_label_overlap=overlap,
                exposure=[dict(task=k[0],command=k[1],bucket=k[2],microbatches=v) for k,v in sorted(exposure.items())],
                source_files=[dict(path=str(p),sha256=sha(p)) for p in (closed,late_path,newfit_path,Path(__file__))])
    write(ROOT/'design/FITTING_CARD.json',card)
    shared = dict(schema='PI0272_READOUT_CONTRACT_V1',release=RELEASE,initial_weights_revision='f862abb26041c8e7d355470d6e9d48899f78b9ce',
                  initial_weights_prefix='arms/PI0270/AGG/checkpoint-4096',statistics_sha256=STATS_SHA,
                  policy_inputs=['body_rgb','wrist_rgb','state_v1','language'],native_num_inference_timesteps=4,
                  state_is_actual=True,normalized_query_shape=['B',40,132],query_dtype='torch.bfloat16',
                  native_loss_retained=True,loss_coefficients=dict(native=1.,position=.4,increment=.1,rotation=.1,peak_speed=.1),
                  temporal_pose=dict(prefix_points=9,prefix_mass=.8,suffix_points=31,suffix_mass=.2),
                  temporal_increment=dict(prefix_transitions=8,prefix_mass=.8,suffix_transitions=31,suffix_mass=.2),
                  new_parameters=0,expected_trainable_tensors=471,expected_frozen_tensors=559,
                  parameter_name_shape_identity_check_required=True,decoder='UNCHANGED_ROT_UNCLAMPED_RELATIVE_SE3_ROW_ROT6D_FP32',
                  selected_checkpoint='NONE',deployment_selected_checkpoint='NONE')
    for mode in ('S4FIT','DIRECT'):
        write(ROOT/'design'/f'{mode}_READOUT.json',dict(**shared,mode=mode,
              generated_nodes=4 if mode=='S4FIT' else 1,
              query='ACTUAL_NATIVE_EPSILON' if mode=='S4FIT' else 'ALL_ZERO_LATENT',
              timestep_buckets='NATIVE_0_0.25_0.5_0.75' if mode=='S4FIT' else [0],
              interpretation='EULER_DT_ONE_QUARTER' if mode=='S4FIT' else 'RAW_READOUT_IS_NORMALIZED_RELATIVE_H40',
              generation_consumes_RNG=False,generated_path_dropout=False,teacher_conditioning=False))
    queries = []
    def add(point,panel,row,noise=None,repeat=False):
        query = dict(ordinal=len(queries)+1,point=point,panel=panel,input_id=row['input_id'],
                     input=row['input'],input_sha256=row['input_sha256'],language=row['language'],
                     expected_condition_sha256=row['condition_sha256'],
                     readout='S4FIT' if point=='S4FIT1024' else 'DIRECT',noise=noise,
                     output=str(RAW/'quality'/point/panel/row['input_id']/(f"j{noise['j']}.npz" if noise else 'direct.npz')),
                     repeatability_only=repeat)
        queries.append(query)
    for row in ordered:
        for noise in row['noise']:add('S4FIT1024','support',row,noise)
        for point in ('DIRECT0','DIRECT1024'):add(point,'support',row)
    repeat_ids=[]
    for task in ('M1','M3'):
        for command in (0,1):
            row=next(r for r in ordered if (r['source_mapping']['task'],r['source_mapping']['command'],r['source_mapping']['bucket'])==(task,command,'ORIGINAL_Q3'))
            repeat_ids.append(row['input_id']);add('DIRECT1024','repeatability',row,repeat=True)
    for row in newfit['rows']:
        assert sha(row['input'])==row['input_sha256'] and sha(row['noise_file'])==row['noise_file_sha256']
        noise=dict(file=row['noise_file'],sha256=row['noise_file_sha256'],noise_sha256=row['noise_sha256'],j='ORIGINAL_NEWFIT')
        for point in ('S4FIT1024','DIRECT1024'):add(point,'early_guard',row,noise if point=='S4FIT1024' else None)
    assert len(queries)==2148 and len({q['output'] for q in queries})==2148
    assert sum(4 if q['readout']=='S4FIT' else 1 for q in queries)==6228
    write(ROOT/'design/EVALUATION_CARD.json',dict(schema='PI0272_FROZEN_QUALITY2148_V1',release=RELEASE,rows=queries,
          fitting_card_sha256=sha(ROOT/'design/FITTING_CARD.json'),NEW_FIT192_sha256=NEWFIT_SHA,
          repeatability_input_ids=repeat_ids,run_only_after_both_complete_finals=True,
          AGG_reuse=dict(support_predictions=1168,early_guard_predictions=192),
          expected_queries=2148,expected_heads=6228,hard_attempts=2560,hard_heads=7168,
          binding_complete_new_final_policy_weights_required=True,no_quality_queries_in_preparation=True))
    write(ROOT/'evidence/SOURCE_AND_PLAN_ADMISSION.json',dict(status='PASS_FROZEN_296_SOURCE_MEMBERSHIPS_AND_2048_MICRO_PLAN',
          fitting_card_sha256=sha(ROOT/'design/FITTING_CARD.json'),evaluation_card_sha256=sha(ROOT/'design/EVALUATION_CARD.json'),
          source_rows=296,source_files_verified=True,microbatches_per_arm=2048,exposure_histogram={7:272,6:24},
          early_guard_overlap=overlap,model_queries=0,teacher_calls=0,renders=0,optimizer_updates=0,
          actual_Hopper_processor_and_runtime_and_real_canaries_still_required=True))
    print('FROZEN_296_INPUTS_2048_MICRO_2148_QUALITY_PLAN',dict(exposure), 'guard_overlap',overlap)


def self_check():
    rows=[dict(input_id=str(i),source_mapping=dict(unit=i//2+1,micro=i%2)) for i in range(296)]
    a,s=schedule(list(reversed(rows)))
    assert a==rows and s[0]['unit']==1 and s[-1]['unit']==1024 and s[-1]['micro']==1
    assert s[0]['seed']==s[1]['seed'] and s[2]['seed']!=s[0]['seed']
    assert [s[i]['input_id'] for i in (0,295,296,2047)]==['0','295','0','271']
    print('PASS_FIXED_ORDER_EXPOSURE_AND_UNIT_SEED_CPU_ONLY')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--self-check',action='store_true')
    if parser.parse_args().self_check:self_check()
    else:main()
