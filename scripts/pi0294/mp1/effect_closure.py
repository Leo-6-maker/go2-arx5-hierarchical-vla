"""Read-only closure of recorded effects, including known unexecuted negatives."""
from pathlib import Path
import hashlib

from scripts.pi0292.durable import read
from scripts.pi0292.relay_protocol import digest, unseal
from scripts.pi0294.native_dataset import sha


def rejected_prefix(folder,row,name,bundle):
    import numpy as np
    from scripts.pi0292.eval_contract import matrix10
    if (row.get('status')!='VALID_TASK_FAIL' or row.get('failure_stage')!='S1_PRE_EXECUTION_REJECT'
            or row.get('pipeline_trustworthy') is not True or row.get('S1_rejections')!=1
            or row.get('live_native_failures')):
        raise PermissionError('UNKNOWN_INSTALLED_PREFIX_NOT_AN_AUDITED_S1_REJECTION')
    folder=Path(folder);relay=folder/'relay';sequence=int(name.removesuffix('.json'))
    query=unseal(bundle['request']);command=unseal(bundle['command']);action=unseal(bundle['action'])
    intent=read(relay/'intents'/name);forecast=read(folder/'S1_PREFLIGHTS.json')[-1]
    s1=read(folder/'S1_RESULT.json');worker=read(folder/'WORKER_RECEIPT.json')
    audit_path=folder.parent.parent/'evidence/audits'/folder.name/'AUDIT.private.json';audit=read(audit_path)
    trajectory=np.asarray(action['payload']['trajectory_eef10'],dtype=float)
    action_hash=hashlib.sha256(matrix10(trajectory)[:8].tobytes()+trajectory[:8,9].tobytes()).hexdigest()
    if (row['status']!='VALID_TASK_FAIL' or row['failure_stage']!='S1_PRE_EXECUTION_REJECT'
            or row['pipeline_trustworthy'] is not True or row['S1_rejections']!=1
            or row['main_integrations']!=sequence*200 or row['live_native_failures']
            or query['sequence']!=sequence or command['sequence']!=sequence or action['sequence']!=sequence
            or intent['tick_before']!=sequence*200 or intent['query_uuid']!=query['query_uuid']
            or intent['command_sha256']!=bundle['command']['sha256'] or intent['requested_integrations']!=200
            or command['prefix_sha256']!=digest(trajectory[:8].tolist())
            or forecast['decision']!='REJECT' or forecast['live_integrations']!=0
            or forecast['live_state_unchanged'] is not True or forecast['actions_modified'] is not False
            or forecast['sequence']!=sequence*8 or forecast['action_sha256']!=action_hash
            or s1['pre_execution_rejections']!=1 or s1['live_native_failures']!=0
            or worker['physics_steps']!=sequence*200
            or audit['status']!='PASS_INDEPENDENT_ACTUAL_EVIDENCE_AUDIT' or audit['adjudication']!=row
            or audit['actual_native_safety_flag_rows']!=0
            or (relay/f'physics-{sequence:03}.npz').exists() or (relay/'acks'/name).exists()):
        raise PermissionError('UNACKED_INSTALL_REQUIRES_EXACT_AUDITED_S1_ZERO_LIVE_REJECTION')
    return dict(attempt_id=folder.name,sequence=sequence,request_sha256=bundle['request']['sha256'],
        action_sha256=bundle['action']['sha256'],command_sha256=bundle['command']['sha256'],
        original_intent_sha256=sha(relay/'intents'/name),S1_forecasts_sha256=sha(folder/'S1_PREFLIGHTS.json'),
        raw_audit_sha256=sha(audit_path),live_integrations=0,physical_ACK_created=False,
        classification='ORIGINAL_S1_REJECTED_INSTALLED_ZERO_LIVE_PREFIX')


def core(root):
    root=Path(root);ledger=read(root/'RESOURCE_LEDGER.json')
    if ledger['reservations'] or ledger.get('CPU_reserved_s',0):
        raise PermissionError('CORE_EFFECTS_STILL_RESERVED')
    episodes={};observations={};installed={};rejected={};rows=[]
    for folder in sorted((root/'episodes').iterdir()):
        if not folder.is_dir():continue
        row=read(folder/'MP1_ADJUDICATION.json');rows.append(row)
        if row['live_native_failures']:raise PermissionError('NATIVE_SAFETY_STOP_NOT_ENGINEERING_CLOSURE')
        worker=read(root/'worker_processes'/(folder.name+'.json'))
        stat=Path('/proc',str(worker['pid']),'stat')
        if stat.exists():
            fields=stat.read_text().split()
            if fields[2]!='Z' and fields[21]==str(worker['proc_start']):
                raise PermissionError('CORE_PHYSICAL_WORKER_STILL_LIVE')
        relay=folder/'relay';acks={};actual=0
        commands={p.name:p for p in (relay/'commands').glob('*.json')}
        intents={p.name:p for p in (relay/'intents').glob('*.json')}
        ack_paths={p.name:p for p in (relay/'acks').glob('*.json')}
        if (set(commands)!=set(intents) or not set(ack_paths)<=set(commands)
                or set(commands)!={f'{i:03}.json' for i in range(len(commands))}
                or set(ack_paths)!={f'{i:03}.json' for i in range(len(ack_paths))}
                or len(commands)-len(ack_paths) not in [0,1]):
            raise PermissionError('UNKNOWN_INSTALLED_OR_EXECUTED_PREFIX_PRESERVE_INTENT')
        for name,path in sorted(commands.items()):
            bundle=read(path);query=unseal(bundle['request']);action=unseal(bundle['action']);command=unseal(bundle['command'])
            if query['query_uuid'] in installed or query['query_uuid'] in rejected:
                raise PermissionError('UNIQUE_CORE_EXECUTED_OR_REJECTED_QUERY_IDENTITY')
            if name not in ack_paths:
                value=rejected_prefix(folder,row,name,bundle)
                rejected[query['query_uuid']]=value
                continue
            ack_packet=read(ack_paths[name]);ack=unseal(ack_packet);sequence=query['sequence']
            if (query['query_uuid'] in installed or name!=f'{sequence:03}.json'
                    or query['sequence']!=action['sequence'] or command['sequence']!=sequence
                    or command['prefix_sha256']!=digest(action['payload']['trajectory_eef10'][:8])
                    or ack['payload']['command_sha256']!=bundle['command']['sha256']
                    or ack['payload']['query_uuid']!=query['query_uuid']
                    or ack['payload']['integrations']!=200 or ack['payload']['tick_after']!=(sequence+1)*200
                    or ack['payload']['physics_receipt_sha256']!=sha(relay/f'physics-{sequence:03}.npz')):
                raise PermissionError('ACTUAL_CORE_COMMAND_EXECUTED_ACK_AND_PHYSICS_CHAIN')
            installed[query['query_uuid']]=dict(attempt_id=folder.name,sequence=sequence,
                request_sha256=bundle['request']['sha256'],action_sha256=bundle['action']['sha256'],
                ACK_sha256=ack_packet['sha256'])
            acks[sequence]=ack_packet['sha256'];actual+=200
        if actual!=row['main_integrations']:
            raise PermissionError('CORE_ACTUAL_PREFIX_COUNT_MUST_EQUAL_ADJUDICATED_PHYSICS')
        for path in sorted((relay/'observations').glob('*.json')):
            packet=read(path);observation=unseal(packet);sequence=observation['sequence'];episode=observation['episode_uuid']
            key=episode+'/'+str(sequence)
            if key in observations or observation['tick']!=sequence*200:
                raise PermissionError('UNIQUE_FRESH_CORE_OBSERVATION_SEQUENCE')
            for image,camera in zip(observation['payload']['images'],['body_rgb','wrist_rgb']):
                if sha(folder/camera/f'{sequence*8:04}.png')!=image['sha256']:
                    raise PermissionError('CORE_RECORDED_RGB_MUST_EQUAL_ACTUAL_SAVED_FRAME')
            observations[key]=dict(attempt_id=folder.name,observation_sha256=packet['sha256'],
                payload_sha256=digest(observation['payload']),
                previous_ACK_sha256=acks.get(sequence-1) if sequence else None)
            if episode in episodes and episodes[episode]!=folder.name:
                raise PermissionError('UNIQUE_CORE_EPISODE_TO_ATTEMPT_IDENTITY')
            episodes[episode]=folder.name
    return dict(status='PASS_ALL_CLOSED_CORE_EFFECTS_RECORDED',rows=rows,observations=observations,
        installed=installed,rejected_installs=rejected,episodes=episodes,core_reservations_closed=True,core_CPU_reserved_s=0,
        actual_executed_prefixes=len(installed),new_physics=0,new_model_queries=0)


def hopper(root,core_proof):
    import numpy as np
    from .fixed_probes import status as probe_status
    root=Path(root);ledger=read(root/'HOPPER_RESOURCE_LEDGER.json')
    if ledger['GPU_reserved_s'] or ledger['pending_updates'] or ledger.get('CPU_reserved_s',0):
        raise PermissionError('NATIVE_EFFECT_CLOSURE_REQUIRES_SETTLED_ALLOCATION')
    if any(r['status'] not in ['NATIVE_TERMINAL','NATIVE_QSUB_REJECTED_NO_JOB'] for r in ledger['PBS_submissions']):
        raise PermissionError('EVERY_OLD_NATIVE_ALLOCATION_MUST_BE_TERMINAL')
    classified=[];generated=0;all_keys=set();executed=set();rejected_keys=set()
    rows={r['attempt_id']:r for r in core_proof['rows']}
    for folder in sorted((root/'evidence').glob('*/resident')):
        state_path=folder/'QUERY_STATE.json';state=read(state_path) if state_path.exists() else dict(queries=[],pending=None)
        if state.get('pending') is not None:raise PermissionError('UNKNOWN_MODEL_QUERY_RESERVATION')
        draws={r['query_uuid']:r for r in state['queries']}
        if len(draws)!=len(state['queries']):raise PermissionError('DUPLICATE_NATIVE_DRAW_IDENTITY')
        native=read(folder/'STATE.json')
        if native['queries']!=len(draws):raise PermissionError('NATIVE_DRAW_COUNT_AND_RESERVATION_DISAGREE')
        inbox={p.stem:p for p in (folder/'inbox').glob('*.json')}
        outbox={p.stem:p for p in (folder/'outbox').glob('*.json')}
        receipts={p.stem:p for p in (folder/'receipts').glob('*.json')}
        arrays={p.stem:p for p in (folder/'receipts').glob('*.npz')}
        if set(draws)!=set(outbox) or set(draws)!=set(receipts) or set(draws)!=set(arrays) or not set(draws)<=set(inbox):
            raise PermissionError('UNKNOWN_DRAW_OR_RESPONSE_OR_ARRAY_EFFECT')
        for key,path in sorted(inbox.items()):
            if key in all_keys:raise PermissionError('QUERY_UUID_REUSED_ACROSS_NATIVE_SESSIONS')
            all_keys.add(key);request=read(path);query=unseal(request);sequence=query['sequence']
            observation=core_proof['observations'][query['episode_uuid']+'/'+str(sequence)]
            attempt=observation['attempt_id'];row=rows[attempt]
            previous=query['previous_ack']
            if (query['query_uuid']!=key or query['observation_sha256']!=observation['observation_sha256']
                    or digest(query['payload'])!=observation['payload_sha256']
                    or (None if previous is None else previous['sha256'])!=observation['previous_ACK_sha256']
                    or query['identity']!=native['identity']['relay_identity']):
                raise PermissionError('EVERY_NATIVE_QUERY_MUST_JOIN_PREVIOUS_EXECUTION_AND_ACTUAL_NEW_OBSERVATION')
            if key not in draws:
                if (key in core_proof['installed'] or key in core_proof.get('rejected_installs',{})
                        or row['status']!='ENGINEERING_INCOMPLETE'
                        or row['main_integrations']!=sequence*200):
                    raise PermissionError('INBOX_ONLY_ORPHAN_CANNOT_HAVE_AN_ACTION_OR_POLICY_SUCCESS')
                classified.append(dict(query_uuid=key,label=folder.parent.name,attempt_id=attempt,
                    classification='TERMINAL_INBOX_ONLY_NO_GENERATION_NO_INSTALL',request_sha256=request['sha256']))
                continue
            record=draws[key];action_packet=read(outbox[key]);action=unseal(action_packet);receipt=read(receipts[key])
            if ('error' in record or record['response_sha256']!=action_packet['sha256']
                    or receipt['action_sha256']!=action_packet['sha256'] or sha(arrays[key])!=receipt['arrays_sha256']
                    or action['payload']['query_uuid']!=key or action['sequence']!=sequence
                    or record['sequence']!=sequence or record['episode_uuid']!=query['episode_uuid']):
                raise PermissionError('EVERY_ACTUAL_DRAW_REQUIRES_ORIGINAL_RAW_ACTION_ARRAY_PROOF')
            with np.load(arrays[key],allow_pickle=False) as values:
                if not np.array_equal(values['physical'],np.asarray(action['payload']['trajectory_eef10'])):
                    raise PermissionError('ORIGINAL_NATIVE_ARRAYS_MUST_MATCH_PUBLISHED_ACTION')
            installed=core_proof['installed'].get(key)
            if installed is None and row['main_integrations']!=sequence*200:
                raise PermissionError('UNEXECUTED_QUERY_MUST_BE_THE_NEXT_PREFIX_AFTER_ACTUAL_FINAL_ACK')
            if installed is not None:
                if installed['request_sha256']!=request['sha256'] or installed['action_sha256']!=action_packet['sha256']:
                    raise PermissionError('ACTUAL_NATIVE_DRAW_AND_INSTALLED_ACTION_MISMATCH')
                classification='EXECUTED_ACKED_PREFIX';executed.add(key)
            elif key in core_proof.get('rejected_installs',{}):
                rejected=core_proof['rejected_installs'][key]
                if rejected['request_sha256']!=request['sha256'] or rejected['action_sha256']!=action_packet['sha256']:
                    raise PermissionError('S1_REJECTED_INSTALLED_BYTES_MUST_MATCH_ORIGINAL_NATIVE_DRAW')
                classification='ORIGINAL_S1_REJECTED_INSTALLED_ZERO_LIVE_PREFIX'
                rejected_keys.add(key)
            elif row['status']=='VALID_TASK_FAIL' and row['S1_rejections']:
                raise PermissionError('S1_REJECTION_REQUIRES_ITS_EXACT_ZERO_LIVE_CORE_PROOF')
            elif row['status']=='ENGINEERING_INCOMPLETE':
                classification='TERMINAL_GENERATED_BUT_NEVER_INSTALLED_ACTION'
            else:raise PermissionError('UNCLASSIFIED_UNINSTALLED_NATIVE_ACTION')
            classified.append(dict(query_uuid=key,label=folder.parent.name,attempt_id=attempt,
                classification=classification,request_sha256=request['sha256'],action_sha256=action_packet['sha256'],
                arrays_sha256=receipt['arrays_sha256']))
        generated+=len(draws)
    if executed!=set(core_proof['installed']):raise PermissionError('EVERY_EXECUTED_PREFIX_REQUIRES_ITS_ACTUAL_NATIVE_DRAW')
    if rejected_keys!=set(core_proof.get('rejected_installs',{})):
        raise PermissionError('EVERY_INSTALLED_S1_REJECTION_REQUIRES_ITS_ACTUAL_NATIVE_DRAW')
    probes=[]
    for branch in ['S-BC','S-CORR']:
        for step in [512,2048]:
            checkpoint=root/'training'/branch/f'checkpoint-{step}'
            complete=read(checkpoint/'CHECKPOINT_COMPLETE.json')
            files={n:s for n,s in complete['files'].items() if n.endswith('.safetensors')
                or n in ['config.json','model.safetensors.index.json','statistics.json'] or n.startswith('processor/')}
            value=probe_status(root,f'{branch}-{step}-TRAIN4',files)
            if not value['complete'] or value['unknown_effects'] or value['new_queries_required']:
                raise PermissionError('FIXED_PROBE_DRAWS_NOT_COMPLETELY_RECONCILED')
            probes.append(dict(label=f'{branch}-{step}-TRAIN4',draws=len(value['completed_draw_ids']),
                checkpoint_manifest_sha256=sha(checkpoint/'CHECKPOINT_COMPLETE.json')))
    from .endpoint_anchor_ni import closed_draws,LABEL as NI_LABEL
    diagnostic_draws=closed_draws(root)
    total=generated+sum(p['draws'] for p in probes)+diagnostic_draws
    if total!=ledger['generation_queries']:
        raise PermissionError('ALL_RESIDENT_AND_FIXED_PROBE_DRAWS_MUST_MATCH_CUMULATIVE_LEDGER')
    return dict(status='PASS_ALL_NATIVE_DRAWS_CLASSIFIED_AND_ARRAYS_VERIFIED',
        cumulative_generation_queries=total,resident_generations=generated,fixed_probes=probes,
        separately_bound_current_anchor_endpoint_diagnostic_draws=diagnostic_draws,
        endpoint_diagnostic_result_sha256=sha(root/'evidence'/NI_LABEL/'RESULT.json') if diagnostic_draws else None,
        classified=classified,actual_executed_prefixes=len(executed),
        native_draws_all_classified=True,all_action_arrays_verified=True,pending_queries=0,
        new_model_queries=0,new_physics=0)
