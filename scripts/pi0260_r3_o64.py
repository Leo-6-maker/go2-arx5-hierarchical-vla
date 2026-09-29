"""Bind the original O64 source identities under the new frozen processor."""
import json
from pathlib import Path

import numpy as np

from pi0260_r2 import ROOT,read,sha
from pi0260_r3_inputs import batch,build,identity

BASE=Path('/home/liuyu-nfs/Projects/go2-arx5')
OLD=BASE/'PI0254'
LANGUAGE='move the end effector to the red target'


def main():
    cohort=read(OLD/'design/O64.json')['rows']
    manifest=read(OLD/'source640/manifest.json')
    assert manifest['status']=='PASS_ORIGINAL_FILES_LABEL_STATE_TIME_BINDING' and len(cohort)==64
    by_source={r['source_episode_id']:r for r in manifest['files']}
    processor=build(read(ROOT/'dataset/statistics.json'),BASE/'PI0250/inputs/frozen_v1/cosmos_snapshot')
    records=[]
    for row in cohort:
        bound=by_source[row['source_episode_id']]
        path=OLD/'source640'/bound['path']
        assert sha(path)==bound['sha256'] and path.stat().st_size==bound['bytes']
        with np.load(path,allow_pickle=False) as z:
            assert z['body_rgb'].shape==z['wrist_rgb'].shape==(96,96,3)
            assert z['state'].shape==(10,) and z['expert'].shape==(40,10) and np.isfinite(z['expert']).all()
            packet=dict(body_rgb=z['body_rgb'],wrist_rgb=z['wrist_rgb'],state_v1=z['state'],language=LANGUAGE)
            key=identity(batch(processor,packet)['inputs'])
        records.append(dict(source_episode_id=row['source_episode_id'],episode_index=row['episode_index'],
                            source_path=str(path),source_sha256=bound['sha256'],target_family=row['target_family'],
                            phase=row['phase'],observation_timestamp=row['observation_timestamp'],condition_sha256=key,
                            original_language=LANGUAGE))
    assert len(records)==len({r['source_episode_id'] for r in records})==64
    p=ROOT/'R3/O64_eval_cases.json';assert not p.exists()
    p.write_text(json.dumps(dict(status='O64_FIXED_ORIGINAL_64_READY_UNDER_R3_PROCESSOR',rows=records,
                                 original_design_sha256=sha(OLD/'design/O64.json'),
                                 original_manifest_sha256=sha(OLD/'source640/manifest.json'),
                                 statistics_sha256=sha(ROOT/'dataset/statistics.json'),
                                 model_loads=0,model_queries=0),indent=2)+'\n')
    print(json.dumps(dict(status='O64_FIXED_ORIGINAL_64_READY_UNDER_R3_PROCESSOR',cases=len(records),sha256=sha(p))))


if __name__=='__main__':main()
