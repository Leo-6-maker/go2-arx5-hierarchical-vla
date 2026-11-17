"""Finite CPU-only reuse of two sealed R3 inputs/outputs. No model or physics."""
import argparse
import json
from pathlib import Path
import time

from scripts.pi0292.durable import write
from go2arx5_vla.world_model.hybrid_nonphysical_v3 import read
from go2arx5_vla.world_model.sim_paused_contract_v5 import (
    PausedMockSession, PROFILE_SHA256, SELECTED_SHIFT7_MANIFEST_SHA256,
)
from go2arx5_vla.world_model.storage import digest


def run(evidence_root, output_root, source_sha):
    start, cpu = time.monotonic(), time.process_time()
    evidence_root, output_root = Path(evidence_root), Path(output_root)
    expected = {
        'RESULT.private.json': '28a2869e4ab0169382c9c640432d4ced5aea7b421a9635e09744ec41e354a5dc',
        'REQUEST_RECORDS.private.json': '5416b229ffd0887fe2df5b74b7b980a62959dd2ff44f19c4ab9ba7aa28bcc831',
    }
    for name, sha in expected.items():
        if digest(evidence_root/name) != sha:
            raise ValueError('SEALED_R3_EVIDENCE_IDENTITY_CHANGED')
    if len(source_sha) != 40:
        raise ValueError('IMMUTABLE_CANDIDATE_SOURCE_REQUIRED')
    output_root.mkdir()  # New CPU output identity; no overwrite/replay.
    rows = []
    recorded = read(evidence_root/'REQUEST_RECORDS.private.json')
    for index in (1, 17):  # First hot F1, first hot F2. No outcome-based switch.
        saved = evidence_root/'requests'/('request-%02d' % index)
        original = read(saved/'REQUEST.private.json')
        response = read(saved/'RESPONSE.json')
        inp = dict(path=str(saved/'INPUT.npz'), sha256=original['input']['sha256'])
        action = dict(path=str(saved/'SHIFT7_OUTPUT.npz'), sha256=response['action']['sha256'])
        if digest(inp['path']) != inp['sha256'] or digest(action['path']) != action['sha256']:
            raise ValueError('SEALED_INPUT_OR_H40_CHANGED')
        seed = original['previous_evidence']
        if digest(seed['bank']['path']) != seed['bank']['sha256']:
            raise ValueError('RECORDED_SEED_EVIDENCE_CHANGED')
        session = PausedMockSession(output_root/('cached-%02d' % index),
            'V5-CACHED-ONLY-R3-%02d' % index, original['native_step'],
            original['previous_reference'], seed,
            dict(architecture='SHIFT7_SINGLE',
                 SHIFT7_package_manifest_sha256=SELECTED_SHIFT7_MANIFEST_SHA256,
                 candidate_source_sha=source_sha, saved_R3_request_id=original['request_id']),
            service_timeout_s=10.)
        query = session.begin_query(inp, original['language'], original['noise_id'],
            native_step=original['native_step'], observation_native_step=original['native_step'],
            observation_available_monotonic=time.monotonic())
        ack = session.install(dict(request_id=query['request_id'], bindings=query['bindings'], action=action),
                              native_step=original['native_step'])
        token = session.prepare_mock_resume(ack, native_step=original['native_step'])
        session.resume_mock(token, native_step=original['native_step'])
        partial = session.consume_mock(3, native_step=original['native_step'])
        if partial['full_P10'] or partial['consumed_atoms'] != 3:
            raise RuntimeError('PARTIAL_HISTORY_NOT_THREE')
        final = session.consume_mock(7, native_step=session.receiver.state['native_step'])
        session.finish()
        rows.append(dict(saved_R3_index=index, family=recorded[index]['family'],
            input_sha256=inp['sha256'], action_sha256=action['sha256'],
            original_H40_compiled=ack['full_H40_compiled'],
            partial_atoms=partial['consumed_atoms'], full_mock_P10=final['full_P10'],
            mock_atoms=final['consumed_atoms'], native_requests=0, native_ticks=0,
            model_generations=0, camera_acquisitions=0,
            prior_recorded_native_step=original['native_step'],
            final_mock_native_step=session.receiver.state['native_step']))
    result = dict(schema='MFREE_V5_SAVED_PAIR_CPU_NONPHYSICAL_RESULT', source_sha=source_sha,
        profile_sha256=PROFILE_SHA256, SHIFT7_manifest_sha256=SELECTED_SHIFT7_MANIFEST_SHA256,
        rows=rows, cached_request_attempts=2, full_mock_P10=sum(r['full_mock_P10']for r in rows),
        mock_atoms=sum(r['mock_atoms']for r in rows),
        wall_s=time.monotonic()-start, process_cpu_s=time.process_time()-cpu,
        strict_realtime_qualified=False, independent_native_readmission=False,
        actual_native_grasp=None, actual_native_lift20mm_hold1s=None,
        effects=dict(new_model_generations=0, new_training_updates=0, new_optimization_visits=0,
                     new_native_requests=0, new_native_ticks=0, real_camera_acquisitions=0))
    write(output_root/'RESULT.private.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--source-sha', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.evidence_root, args.output_root, args.source_sha)))
