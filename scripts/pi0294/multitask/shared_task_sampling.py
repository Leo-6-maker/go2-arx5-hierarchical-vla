"""Positioned task/episode/window sampling for the fixed raw shared baseline.

Only TRAIN row identities are used. No BOOK phase, task score or privileged
metadata becomes model input. This module neither admits training nor launches
models; its finite sequence is consumed by the existing native dataset/Trainer.
"""
import hashlib
import json

import numpy as np

TASKS = ('T1','T2','BOOK','DOOR')
ALGORITHM = 'SHARED4_TASK8_EPISODE_BALANCED_WINDOW_CYCLES_V1'
SEED = 2026110204
HORIZON = 8192*32
TASK_COUNTS = {'T1':(10,6010),'T2':(10,6010),'BOOK':(34,19538),'DOOR':(5,3005)}
SCHEMA = 'SHARED4_RAW_M32_TASK8_8192_V1'


def recipe(manifest_sha256,statistics_sha256,population_sha256,task_DEV_sha256,*,steps=8192):
    from .raw_parent_delivery import REPO,REVISION,PARENT_ID
    pins=(manifest_sha256,statistics_sha256,population_sha256,task_DEV_sha256)
    if any(type(p)is not str or len(p)!=64 or any(c not in'0123456789abcdef'for c in p)for p in pins):
        raise ValueError('EXACT_SHARED_DATA_STATISTICS_POPULATION_AND_DEV_BINDINGS')
    if type(steps)is not int or steps not in(10,8192):raise ValueError('RAW_SHARED_NATIVE10_OR_FINITE8192')
    return dict(schema=SCHEMA,pretrained_repo=REPO,pretrained_revision=REVISION,
        raw_parent_files_identity=PARENT_ID,initialization='ORIGINAL_RAW_CATEGORY10_NOT_E1_BOOK128_OR_B248',
        dataset_manifest_sha256=manifest_sha256,statistics_sha256=statistics_sha256,
        TRAIN_population_sha256=population_sha256,task_DEV_population_sha256=task_DEV_sha256,
        TRAIN_entries=59,full_expert_episodes=46,expert_suffix_entries=13,TRAIN_windows=34563,
        effective_batch_size=32,microbatch=1,gradient_accumulation_steps=32,
        trainable_numel=77847684,target_category=10,limited_DiT_blocks=2,
        actual_DiT_block_indices=[30,31],backbone_frozen=True,
        sampling=ALGORITHM,seed=SEED,task_slots_per_batch32=dict.fromkeys(TASKS,8),
        episode_sampling='BALANCED_EPISODE_THEN_WITHOUT_REPLACEMENT_WINDOW_CYCLES',
        task_specific_stage_weighting=False,privileged_metadata_student_input=False,
        state_history_length=1,H40=40,execution_prefix=8,denoise_steps=4,
        loss='UNCHANGED_OFFICIAL_MASKED_FLOW_MATCHING',continuous_gripper_decode=True,
        optimizer='FRESH_ADAMW_FROM_RAW',learning_rate=1e-4,weight_decay=1e-5,
        scheduler_horizon_steps=8192,warmup_updates=256,
        max_steps=steps,TRAIN_visits=steps*32,capacity_qualification_only=steps==10,
        checkpoint_steps=[10]if steps==10 else[2048,4096,6144,8192],
        primary_endpoint=steps,generation_queries=32 if steps==10 else 80,
        native_diagnostic_loss_forwards=32 if steps==10 else 80,
        diagnostic_task_windows=dict.fromkeys(TASKS,4),
        checkpoint='RAW_PARENT_BOUND_MODEL_OPTIMIZER_SCHEDULER_RNG_SAMPLER',
        new_STATISTICS_or_augmentation_fit=False,DEV_TEST_in_optimizer=False,
        WBC_changed=False,physics_integrations=0,
        original_E1_E2_and_book_experiments_changed=False)


def is_recipe(value):
    try:
        return value==recipe(value['dataset_manifest_sha256'],value['statistics_sha256'],
            value['TRAIN_population_sha256'],value['task_DEV_population_sha256'],steps=value['max_steps'])
    except(KeyError,TypeError,ValueError):return False


def row_identity(row,anchor):
    return dict(id=row['id'],task=row['task'],anchor=anchor,numeric_sha256=row['sha256'])


def pools(rows):
    bank={t:{}for t in TASKS};seen=set();entries={}
    for index,(row,anchor)in enumerate(rows):
        if (row.get('split')!='TRAIN'or row.get('task')not in bank
                or type(anchor)is not int or type(row.get('windows'))is not int
                or not 0<=anchor<row['windows']<=601
                or not isinstance(row.get('sha256'),str)or len(row['sha256'])!=64):
            raise PermissionError('ONLY_FROZEN_TRAIN_EXPERT_WINDOW_IDENTITIES')
        key=(row['id'],anchor)
        if key in seen:raise PermissionError('DUPLICATE_SHARED_TRAIN_WINDOW')
        seen.add(key)
        identity=(row['task'],row['windows'],row['sha256'])
        if entries.setdefault(row['id'],identity)!=identity:
            raise PermissionError('INCONSISTENT_SHARED_EPISODE_IDENTITY')
        bank[row['task']].setdefault(row['id'],[]).append((anchor,index))
    for task,episodes in bank.items():
        for name,values in episodes.items():
            windows=entries[name][1]
            if sorted(a for a,i in values)!=list(range(windows)):
                raise PermissionError('COMPLETE_ORIGINAL_VALID_WINDOWS_WITHOUT_PADDING')
            episodes[name]=np.asarray([i for a,i in sorted(values)],dtype=np.uint32)
        if not episodes:raise PermissionError('ALL_FOUR_TASKS_REQUIRE_ACTUAL_EXPERT_SUPPORT')
    return bank


def identity_digest(rows):
    value=[row_identity(row,a)for row,a in rows]
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def seed_for(seed,task,episode,stream):
    body=json.dumps([ALGORITHM,seed,task,episode,stream],separators=(',',':')).encode()
    return int.from_bytes(hashlib.sha256(body).digest()[:8],'big')


def sequence(rows,*,seed=SEED,visits=HORIZON):
    if type(seed)is not int or seed<0 or type(visits)is not int or not 0<=visits<=HORIZON:
        raise ValueError('FINITE_SHARED_TASK_SAMPLING_RANGE')
    bank=pools(rows);result=np.empty(visits,dtype=np.uint32)
    episode_rng={t:np.random.Generator(np.random.PCG64(seed_for(seed,t,'','EPISODE')))for t in TASKS}
    window_rng={(t,n):np.random.Generator(np.random.PCG64(seed_for(seed,t,n,'WINDOW')))
        for t in TASKS for n in sorted(bank[t])}
    episode_order={};episode_cursor={t:0 for t in TASKS};window_order={};window_cursor={}
    for cursor in range(visits):
        task=TASKS[cursor%4];names=sorted(bank[task]);position=episode_cursor[task]
        if position%len(names)==0:episode_order[task]=episode_rng[task].permutation(len(names))
        name=names[int(episode_order[task][position%len(names)])];episode_cursor[task]+=1
        key=(task,name);position=window_cursor.get(key,0);indices=bank[task][name]
        if position%len(indices)==0:window_order[key]=window_rng[key].permutation(len(indices))
        result[cursor]=indices[window_order[key][position%len(indices)]];window_cursor[key]=position+1
    return result


def validate_fixed59(rows):
    bank=pools(rows)
    if {t:(len(bank[t]),sum(map(len,bank[t].values())))for t in TASKS}!=TASK_COUNTS:
        raise PermissionError('FIXED59_ACTUAL_46_FULL_PLUS13_EXPERT_SUFFIX_POPULATION')
    full=sum(len(v)==601 for t in bank.values()for v in t.values())
    suffix=sum(len(v)<601 for t in bank.values()for v in t.values())
    if (full,suffix)!=(46,13):raise PermissionError('FULL_EPISODES_AND_SUFFIXES_MUST_BE_REPORTED_SEPARATELY')
    return bank


class TaskBalancedSampler:
    def __init__(self,dataset,*,seed=SEED,start=0,stop=HORIZON,positioned=True):
        if (type(start)is not int or type(stop)is not int or not 0<=start<=stop<=HORIZON
                or type(seed)is not int or seed<0):
            raise ValueError('FINITE_SHARED_COMMITTED_CURSOR')
        self.rows=dataset.rows;self.manifest_sha256=dataset.manifest_sha
        validate_fixed59(self.rows)
        self.start,self.stop,self.seed,self.positioned=start,stop,seed,positioned
        self.row_identity_sha256=identity_digest(self.rows)
        self.values=sequence(self.rows,seed=seed)
        self.sequence_sha256=hashlib.sha256(self.values.astype('<u4',copy=False).tobytes()).hexdigest()

    def __len__(self):return self.stop-self.start

    def index_at(self,cursor):
        if type(cursor)is not int or not self.start<=cursor<self.stop:raise IndexError('SHARED_FINITE_CURSOR')
        return int(self.values[cursor])

    def __iter__(self):
        for cursor in range(self.start,self.stop):
            index=int(self.values[cursor]);yield(cursor,index)if self.positioned else index

    def state_dict(self,consumed):
        if type(consumed)is not int or not self.start<=consumed<=self.stop:
            raise ValueError('SHARED_ACTUALLY_CONSUMED_CURSOR')
        return dict(algorithm=ALGORITHM,seed=self.seed,cursor=consumed,stop=self.stop,
            manifest_sha256=self.manifest_sha256,row_identity_sha256=self.row_identity_sha256,
            sequence_sha256=self.sequence_sha256,horizon_visits=HORIZON,
            task_slots_per_batch32=dict.fromkeys(TASKS,8),episode_balance='CYCLIC_PERMUTATIONS',
            window_balance='WITHOUT_REPLACEMENT_WITHIN_EACH_EPISODE_CYCLE')


def exposure(rows,values):
    bank=validate_fixed59(rows);counts=np.bincount(values,minlength=len(rows));by_task={}
    for task in TASKS:
        episodes=[]
        for name,indices in sorted(bank[task].items()):
            visits=counts[indices]
            episodes.append(dict(id=name,windows=len(indices),visits=int(visits.sum()),
                unique_windows=int(np.count_nonzero(visits)),minimum_window_visits=int(visits.min()),
                maximum_window_visits=int(visits.max()),full_episode=len(indices)==601))
        by_task[task]=dict(visits=sum(r['visits']for r in episodes),entries=len(episodes),
            windows=sum(r['windows']for r in episodes),unique_windows=sum(r['unique_windows']for r in episodes),episodes=episodes)
    if len(values)%32:raise ValueError('COMPLETE_EFFECTIVE_BATCHES_FOR_EXPOSURE_REPORT')
    task_values=np.asarray([TASKS.index(row['task'])for row,a in rows],dtype=np.uint8)[values]
    if any(not np.all((task_values.reshape(-1,32)==i).sum(1)==8)for i in range(4)):
        raise PermissionError('EACH_EFFECTIVE_BATCH_HAS_EIGHT_FROM_EACH_TASK')
    return dict(algorithm=ALGORITHM,visits=len(values),effective_batch=32,
        every_batch_exact_task8=True,unique_windows=int(np.count_nonzero(counts)),tasks=by_task,
        sequence_sha256=hashlib.sha256(values.astype('<u4',copy=False).tobytes()).hexdigest(),
        new_models_updates_physics=0,training_admitted=False)
