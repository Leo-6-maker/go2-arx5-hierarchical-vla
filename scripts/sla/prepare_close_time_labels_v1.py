"""Finite original-C20 label reconstruction, no images/model/optimizer/physics."""
import json
from pathlib import Path
import time
import traceback
import numpy as np
from scripts.pi0292.durable import write
from go2arx5_vla.sla_adapter.gripper_time_data_v1 import build
from go2arx5_vla.sla_adapter.gripper_time_v1 import VERSION,BLOCKS
from go2arx5_vla.world_model.mfree_unattended_ledger_v1 import read,sha


def run(spec):
    root=Path(spec['registered_root']);started=time.time()
    if spec['new_optimizer_updates_budget']or spec['new_physical_requests_budget']or spec['new_complete_generations_budget']:
        raise ValueError('LABEL_PREPARATION_EFFECT_BUDGET_ZERO')
    targets,contexts,detail=build(spec['dataset'],spec['dense_manifest_sha256'])
    with(root/'TIME_TARGETS.private.npz').open('xb')as handle:np.savez(handle,normalized_gripper_parameters=targets)
    write(root/'REFERENCE_CONTEXTS.private.json',dict(contexts=contexts,source_pins=detail['source_pins']))
    write(root/'ALL_LABEL_RECORDS.private.json',detail)
    output=dict(schema='SLA_REFERENCE_CONTINUED_TIME_CODEC_C20_LABEL_AUDIT_V1',exit_code=0,passes=True,
        source_sha=spec['source_sha'],codec_version=VERSION,parameter_slots=BLOCKS,
        dataset_manifest_sha256=spec['dense_manifest_sha256'],statistics_sha256=detail['statistics_sha256'],
        original_stats_refitted=False,original_files_written=False,windows=len(targets),
        maximum_width_reconstruction_error_m=detail['maximum_width_error_m'],
        maximum_speed_with_seam_m_s=detail['maximum_speed_with_seam_m_s'],
        annotated_crossings=detail['annotated_crossings'],annotated_crossings_within100ms=detail['annotated_crossings_within100ms'],
        target_bank=dict(path=str(root/'TIME_TARGETS.private.npz'),sha256=sha(root/'TIME_TARGETS.private.npz')),
        reference_contexts=dict(path=str(root/'REFERENCE_CONTEXTS.private.json'),sha256=sha(root/'REFERENCE_CONTEXTS.private.json')),
        all_records=dict(path=str(root/'ALL_LABEL_RECORDS.private.json'),sha256=sha(root/'ALL_LABEL_RECORDS.private.json')),
        new_optimizer_updates=0,new_training_visits=0,new_model_generations=0,new_physical_requests=0,new_integrations=0,
        task_success_proven=False,physical_admission=False,wall_s=time.time()-started,CPU_s=time.process_time())
    return output


if __name__=='__main__':
    import sys
    s=read(sys.argv[2]);root=Path(s['registered_root'])
    try:result=run(s)
    except BaseException:result=dict(exit_code=1,status='LABEL_RECONSTRUCTION_FAILED_NO_TRAINING',failure=traceback.format_exc(),new_optimizer_updates=0,new_physical_requests=0)
    write(root/'RESULT.private.json',result)
    from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close
    close(s,result['exit_code']);raise SystemExit(result['exit_code'])
