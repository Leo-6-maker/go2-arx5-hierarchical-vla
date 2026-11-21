"""Export one independently qualified terminal; keep validation data separate."""
from pathlib import Path
import os

import numpy as np

from go2arx5_vla.sla_adapter.codec_v1 import Quantiles
from go2arx5_vla.sla_adapter.multitask_data_v6 import C20_STATISTICS
from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
from go2arx5_vla.world_model.storage import digest
from scripts.pi0292.durable import write
from scripts.sla.export_multitask_g7_v6 import fp32_audit
from scripts.sla.multitask_g7_capture_v6 import copy_shards, read_capture


def admit(root, condition, case, native, result_sha, stats_path):
    split = condition['split']
    folder = root/'admission'/split/condition['id']
    raw = read_pin(dict(path=stats_path,sha256=C20_STATISTICS))
    stats = {k:Quantiles(raw[k]['q01'],raw[k]['q99']) for k in ('state','actions')}
    capture, manifest, frame, codec, arrays = read_capture(condition,case,native)
    audit = fp32_audit([(condition['id'],codec,arrays,arrays['previous_actual_command'])],stats)
    if any(g['over_original_gate'] or (g['codec_path']=='fp32_codec' and g['over_margin'])
           for g in audit['label_margin_by_domain_codec_phase']):
        raise ValueError('Original G7 FP32 margin')
    copy_shards(capture,manifest,arrays)
    tmp = folder.parent/(folder.name+'.incoming')
    tmp.mkdir(parents=True)
    rgb = copy_shards(capture,manifest,arrays,tmp)
    for row in rgb:
        row['file'] = str(Path(folder.name)/Path(row['file']).name)
    labels = tmp/'LABELS.npz'
    with labels.open('xb') as out:
        np.savez(out,**arrays)
        out.flush()
        os.fsync(out.fileno())
    episode = dict(id=condition['id'],split=split,task=condition['instruction'],family=condition['family'],
                   language=case['language'],windows=manifest['queries'],mount=dict(T_policy_from_mount=frame.tolist()),
                   scene_sha256=case['frozen_files'][case['model_path']],reset=condition['reset'],
                   label_kind='expert_query_plan_H40',expert_stage_audit_only=arrays['phase'].tolist(),
                   labels=dict(file=str(Path(folder.name)/labels.name),sha256=digest(labels)),
                   frame_shards=rgb,native_result_sha256=result_sha,individual_pass=True)
    write(tmp/'EPISODE.private.json',episode)
    tmp.rename(folder)
    return dict(path=str(folder/'EPISODE.private.json'),sha256=digest(folder/'EPISODE.private.json'),
                windows=manifest['queries'],confirmed_P10_atoms=10*manifest['queries'],split=split)


if __name__ == '__main__':
    import json
    import sys
    filename, sha = sys.argv[1:]
    request = read_pin(dict(path=filename,sha256=sha))
    case = read_pin(request['case'])
    native = read_pin(request['native'])
    try:
        ref = admit(Path(case['campaign_root']), request['condition'], case, native,
                    request['native']['sha256'], request['statistics'])
        result = dict(admitted=True, episode=ref)
    except (KeyError,ValueError) as error:
        result = dict(admitted=False, first_failure=str(error))
    print(json.dumps(result))
