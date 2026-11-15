"""Fresh matched-budget N20 reference-label fit;old negative endpoint is retained."""
import json
from pathlib import Path
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.reference_dataset_v6 import OPERATION,ORIGINAL_MANIFEST_SHA
from go2arx5_vla.sla_adapter.reference_labels_v6 import VERSION


def admission(spec):
    if (spec['operation']!=OPERATION or spec.get('resume')or spec.get('continuation')
            or spec['reference_label_semantics']!=VERSION
            or spec['microbatch']!=8 or spec['training_matmul_precision']!='highest'
            or spec['closeout_reserve_s']<120
            or(spec['planned_candidate_updates'],spec['planned_target_visits'],spec['planned_source_visits'])!=(3000,96000,0)):
        raise ValueError('NEW_REFERENCE_LABEL_CLEAN_BASE_FIXED3K_NOT_EXTRA_OLD_REPRESENTATION_UPDATES')
    from scripts.sla.train_named_target_v5 import reusable_core_qualifications
    reusable_core_qualifications(spec)
    before=read_pin(spec['original_N20_complete3K'])
    if (before['schema']!='SLA_CORE_FRESH_N20_FM3K_RGB_ACTUAL_V5' or not before['passes']
            or not before['complete3K']or before['dataset_manifest_sha256']!=ORIGINAL_MANIFEST_SHA
            or before['step']!=3000 or before['target_visits']!=96000 or before['source_visits']!=0):
        raise ValueError('ACTUAL_OLD_N20_3K_NEGATIVE_CONTROL_REQUIRED')


def run(spec):
    from scripts.sla import train_dense_pilot_v4 as dense
    from scripts.sla.train_v1 import probes
    from go2arx5_vla.sla_adapter.runtime_reference_v6 import ReferenceTargetEngine
    def checked_probes(engine,name):
        if name=='step0-RGB':
            old=read_pin(spec['original_N20_initial_complete'])
            current=json.loads((engine.output/'step0000/COMPLETE.json').read_bytes())
            expected={r['key']:r for r in old['arrays']}
            if len(expected)!=87 or len(current['arrays'])!=87 or any(
                    any(row[k]!=expected[row['key']][k]for k in('sha256','shape','dtype'))for row in current['arrays']):
                raise ValueError('ALL87_MODEL_ADAM_RNG_INITIAL_ARRAYS_MUST_MATCH_ORIGINAL_N20')
            from scripts.pi0292.durable import write
            write(Path(spec['registered_root'])/'COMMON_INITIAL_ACTUAL.private.json',dict(
                passes=True,arrays_compared=87,bit_exact=True,optimizer_updates=0,
                original_complete_sha256=spec['original_N20_initial_complete']['sha256']))
        return probes(engine,name)
    result=dense.run(spec,engine_class=ReferenceTargetEngine,admission_check=admission,
        endpoint_updates=3000,candidate_run=True,expected_resume_cursor=(0,0),probe_function=checked_probes)
    result.update(schema='SLA_N20_REFERENCE_LABEL_FINITE_FM3K_ACTUAL_V6',reference_label_semantics=VERSION,
        same_original20_observations=True,original_3K_control_retained=True,
        complete3K=result['step']==3000 and result['exit_code']==0,D100_new=False,D300_new=False)
    if result['complete3K']:
        from go2arx5_vla.sla_adapter.provider_reference_v6 import export
        from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
        package=Path(spec['registered_root'])/'RGB_PACKAGE'
        export(package,result['last_checkpoint']['path'],active_template=dense.ACTIVE_ENGINE.active,
            dataset=spec['dataset'],tokenizer=spec['tokenizer']['path'],parent_manifest=spec['parent_manifest'])
        result['package']=dict(path=str(package),sha256=sha(package/'MANIFEST.private.json'))
    return result


if __name__=='__main__':
    from scripts.sla.train_dense_pilot_v4 import main
    raise SystemExit(main(runner=run))
