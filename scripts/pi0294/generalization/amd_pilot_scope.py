"""CPU-only GEN1 pilot/training contracts using the existing resource vectors.

No native admission, registration, ledger mutation, model or rollout occurs.
"""
from copy import deepcopy
import hashlib
import json
import math

from .static_preview import PACKET_SHA,specifications
from .spec import design_cards
from scripts.pi0294.pv1 import mix_accounting as a,successor_cd as s,mix_vanda_first as first

TASK='GEN1_AMD_LAYOUT_PILOT_CORR1024_V1'
GIB=1024**3
PARENT=a.PARENT


def need(ok,why):
    if not ok:raise PermissionError('GEN1_AMD_SCOPE_'+why)


def pilot_plan(packet_bytes):
    need(type(packet_bytes)is bytes and hashlib.sha256(packet_bytes).hexdigest()==PACKET_SHA,
         'EXACT_EXISTING_SIX_TRAIN_V2_PREVIEW_PACKET')
    groups=specifications(json.loads(packet_bytes));attempts=[]
    for geometry,pair in sorted(groups.items()):
        spec=pair['V0']
        for card in design_cards(spec):
            attempts.append(dict(attempt_id='GEN1-PILOT-'+spec['design_hash'][:16]+'-'+card['task'],
                geometry_id=geometry,lineage_id=spec['lineage_id'],design_hash=spec['design_hash'],
                visual_profile='V0',study_split='G_TRAIN',task=card['task'],
                layout_stratum=spec['layout']['direction']+'/'+spec['layout']['distance_bin'],
                reset='LEGACY_MATCHED',instruction=card['instruction']))
    need(len(attempts)==12 and len({x['attempt_id']for x in attempts})==12,'TWELVE_FIXED_FIRST_ATTEMPTS')
    return dict(task=TASK,packet_sha256=PACKET_SHA,attempts=attempts,first_attempts_only=True,
        replacement_attempts=0,preview_images=24,preview_visuals=['V0','V1'],
        training_visuals=['V0'],colors_used_as_new_training_data=False,
        execution_sites=dict(Teacher='CORELAB_AMD_CPU_MESA',export='VANDA_CPU_PBS',train='VANDA_A40_PBS'),
        repair=dict(maximum_extra_PBS=1,known_zero_model_query_constructor_physics_update_only=True,
                    GPU_s=3600,site_CPU_s=3600,LOCAL_CPU_s=600,metadata_bytes_per_site=64*1024**2,
                    Teacher_valid_failures_replayed=False,unknown_writes_replayed=False),
        physics='ORIGINAL_ARX5_SOURCE',R0_unchanged=True,Teacher_source='EXISTING_CAPTURE_RUN_CELL',
        export_source='EXISTING_EXPORT_COMMITTED_CASE_AND_CODEC_PACKET',
        dataset_gate='At least one independently audited R0-admitted success per task in each of four frozen layout strata; otherwise no training',
        train=dict(initial_checkpoint_manifest_sha256=PARENT,independent_fresh_optimizer=True,
            updates=1024,reference_recipe='C50_FROZEN_OPTIMIZER_AND_EXPERT_LANE_RATIO',
            new_expert_fraction=.5,entire_branch_new_fraction=.375,statistics_sha256=a.STATS,
            inference_num_steps=4,new_id='GEN1_LAYOUT_V0_CORR1024',old_C_D_endpoints_unchanged=True),
        native_runtime_adapter='NOT_DEPLOYED',native_admission=False)


def training_support(plan,attempts):
    expected={r['attempt_id']:r for r in plan['attempts']}
    need(type(attempts)is list and len(attempts)==12 and len({r['attempt_id']for r in attempts})==12
         and {r['attempt_id']for r in attempts}==set(expected),'ALL_FIRST_ATTEMPTS_AND_FAILURES_RETAINED')
    support=set();admitted=[]
    for row in attempts:
        original=expected[row['attempt_id']]
        need(row['unknown_effects']==[] and row['terminal']is True
             and type(row['admitted_for_training'])is bool,'KNOWN_TERMINAL_DATA_EFFECTS')
        need(type(row['native_safety_failures'])is int and row['native_safety_failures']==0,
             'ORIGINAL_NATIVE_SAFETY_FAIL_STOPS_PACKAGE_NOT_ONLY_ONE_LABEL')
        if not row['admitted_for_training']:continue
        need(row['full_task_success']is True and row['independent_raw_audit']=='PASS_INDEPENDENT_RAW_AUDIT'
             and row['codec_status']=='PASS_FIXED_ORIGINAL_V3_0_NO_SILENT_CLIP'
             and row['statistics_sha256']==a.STATS and row['H40_windows']==601
             and row['task']==original['task'] and row['geometry_id']==original['geometry_id']
             and row['design_hash']==original['design_hash'] and row['study_split']=='G_TRAIN'
             and row['visual_profile']=='V0' and row['Teacher_takeover_of_student']is False,
             'ACTUAL_SUCCESS_RAW_LANGUAGE_R0_AND_EXACT_TRAIN_IDENTITY')
        support.add((original['task'],original['layout_stratum']));admitted.append(row['attempt_id'])
    required={(task,d+'/'+b)for task in('T1','T2')for d in('+Y','-Y')for b in('NEAR','FAR')}
    return dict(status='DATA_SUPPORT_READY_FOR_FRESH_TRAINING_ADMISSION'if support==required else 'DATA_SUPPORT_INCOMPLETE_NO_TRAINING',
        missing=sorted(required-support),admitted=sorted(admitted),all_first_attempts_retained=True,
        physics_or_training_authorized=False)


def full_forecast(global_bytes,historical_LOCAL_bytes):
    need(type(global_bytes)is bytes and type(historical_LOCAL_bytes)is bytes,'FULL_ORIGINAL_CAPTURED_BYTES')
    g,l=json.loads(global_bytes),json.loads(historical_LOCAL_bytes)
    base=s.recompute(g,l,Core_storage_upper_bytes=first.CORE_STORAGE,local_storage_upper_bytes=first.LOCAL_STORAGE)
    # The original installed accounting capsule predates GEN1. Read the four
    # real adopted dimensions from their exact validated grant, rather than
    # assuming that capsule already knows a future scope. No other cap moves.
    adopted={}
    if 'GEN1_AMD_layout_CORR1024_authority' in g:
        from .amd_adoption import validate as adopted_GEN1
        adopted=adopted_GEN1(g)['caps']
    need(g[a.PACKAGE]['completed_stages']==['TRAIN:C50:1024','TRAIN:D25:1024']
         and g['pending_updates']==g['unknown_effects']==[],'BOTH_CLOSED_TRAININGS_NO_CURRENT_EFFECTS')
    # The conservative historical LOCAL projection already retains the two
    # original Eval GPU/Core-CPU escrows. Never count that as a fresh read.
    need(base['dimensions']['H200_s']['reserved']>=2*28860
         and base['dimensions']['Core_CPU_s']['reserved']>=2*173900,
         'EXACT_ORIGINAL_C_D_COMPUTE_HOLDS_RETAINED')
    add={k:0 for k in base['dimensions']}
    def plus(values):
        for k,x in values.items():add[k]+=x
    # AMD source:36 paired-task/repeat+6 home static+2 parity constructions.
    # Preserve two complete raw envelopes rather than assume hardlink credit.
    # Keep81 task/shadow starts, the original four zero-integration install
    # negatives, and up to3 readonly raw-audit reconstructions separately.
    # The original safety procedures and all campaign caps are unchanged.
    plus(dict(episode_starts=12,physics_ticks=12*32000,engine_starts=12*(81+4+3)+44,
        Core_CPU_s=1200+12*3600+1800,local_CPU_s=1800,
        Core_storage_bytes=20*GIB+64*1024**2,Hopper_storage_bytes=10*GIB,
        transfer_bytes=12*GIB))
    failed=g.get('GEN1_AMD_layout_CORR1024_authority',{}).get('teacher_pilot',{})
    if failed.get('status')=='NATIVE_ENGINEERING_TERMINAL':
        # Real failed costs remain in addition to a complete future twelve;
        # no scientific negative is removed or used to choose another layout.
        actual=failed['actual']
        need(failed['all_old_costs_and_files_preserved']is True
             and actual['unknown_effects']==[]and actual['admitted_demonstrations']==0,
             'CLOSED_ENGINEERING_EFFECTS_RETAINED')
        plus(dict(episode_starts=actual['episode_starts'],
            physics_ticks=actual['live_ticks']+actual['shadow_ticks'],
            engine_starts=actual['engine_starts'],Core_CPU_s=actual['Core_CPU_s']+600,
            local_CPU_s=600,Core_storage_bytes=320*1024**2,
            Hopper_storage_bytes=64*1024**2,transfer_bytes=64*1024**2))
        repair=failed.get('engineering_recovery',{})
        if repair.get('status')=='NATIVE_AUDIT_ENGINEERING_TERMINAL':
            # Its two full trajectories are part of the twelve logical cells,
            # so never reserve twelve new Teacher executions on top of them.
            # Separate saved-state audit and next cold preparation are finite
            # engineering work; all old native and failed-audit costs remain.
            plus(dict(engine_starts=4,Core_CPU_s=1200,local_CPU_s=900))
    # Export in an independent CPU PBS allocation, using Vanda's existing
    # official processor environment. Train receives a closed immutable bank.
    plus(dict(PBS_submissions=1,Hopper_CPU_s=36600,local_CPU_s=600))
    plus(a.stage_vector('TRAIN'))
    # Re-invoice future C output fully, not transfer its consumed old request's
    # paidIO marker. Its original compute/paid control remain already held.
    ev=a.stage_vector('EVAL',Core_GPU_projection=True)
    plus({k:ev[k]for k in ('episode_starts','physics_ticks','engine_starts','neural_queries',
                          'Hopper_storage_bytes','Core_storage_bytes','transfer_bytes')})
    # Keep one new endpoint's eventual original24 Core Eval in the complete
    # forecast. No current Eval dispatch and no new Vanda Eval PBS are implied.
    plus(ev)
    add['PBS_submissions']-=1  # original D Vanda Eval is now a Core task
    # One bounded cold engineering replacement, only with actual terminal
    # proof of zero model/query/constructor/physics/update effects. It never
    # replays a Teacher valid failure or an UNKNOWN write.
    plus(dict(PBS_submissions=1,H200_s=3600,Hopper_CPU_s=3600,local_CPU_s=600,
        Hopper_storage_bytes=64*1024**2,Core_storage_bytes=64*1024**2,transfer_bytes=64*1024**2))
    if 'GEN1_AMD_layout_CORR1024_authority'in g:
        from .amd_export_accounting import forecast_credit
        for key,amount in forecast_credit(g).items():add[key]-=amount
        if 'official_export_git_repair'in g['GEN1_AMD_layout_CORR1024_authority']:
            from .amd_export_repair_accounting import forecast_credit as repaired_export_credit
            for key,amount in repaired_export_credit(g).items():add[key]-=amount
    rows={};required={}
    for key,v in base['dimensions'].items():
        cap=adopted.get(key,v['cap']);projected=v['projected']+add[key];shortfall=max(0,projected-cap)
        requested=math.ceil(projected/GIB)*GIB if shortfall and key.endswith('bytes')else projected if shortfall else cap
        rows[key]=dict(v,cap=cap,additional=add[key],projected=projected,shortfall=shortfall,proposed_cap=requested)
        if shortfall:required[key]=dict(before=cap,after=requested,
            after_GiB=requested//GIB if key.endswith('bytes')else None)
    return dict(task=TASK,GLOBAL_sha256=hashlib.sha256(global_bytes).hexdigest(),
        LOCAL_is_historical=True,native_admission=False,scope_adopted=False,
        dimensions=rows,cap_changes_required=required,old_costs_refunded=False,
        original_C_D_Eval_and_new_GEN_Eval_forecast_retained=True,
        matrix3_and_final20GPUh40GiB_retained=True,T_CD=g[s.FIELD]['T_CD'],
        deadline=g[s.FIELD]['deadline_epoch'],stop_new_training=g[s.FIELD]['stop_new_training_epoch'])
