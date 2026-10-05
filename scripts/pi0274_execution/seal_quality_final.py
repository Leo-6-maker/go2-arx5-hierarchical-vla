"""Extend the original immutable field seal before the sole new endpoint queries."""
from pathlib import Path
from quality_gate import BASE,ROOT,read,sha,admitted
from quality_worker import write

def main():
    admitted('EQUIL5120');original=read(ROOT/'design/QUALITY_EXECUTION_FILESET.json')
    for r in original['files']:assert sha(r['path'])==r['sha256']
    fixtures=read(ROOT/'evidence/REPORT_CPU_FIXTURES.json');assert fixtures['status']=='PASS_PI0274_ORIGINAL_BOUNDARIES_AND_NEW_FIELD_REPORT_WRAPPERS'
    for r in fixtures['files']:assert sha(r['path'])==r['sha256']
    assert read(ROOT/'evidence/HISTORICAL_TABLE_REUSE.json')['status']=='PASS_CLOSED_AGG_BROAD_ALL_ORIGINAL_PANELS_AND_S4FIT_CONTEXT'
    native=read(BASE/'PI0273/evidence/NATIVE_CPU_FIXTURES.json');assert native['status']=='PASS_CONSTRUCTED_NATIVE_E0_E1_AND_GATE'
    for r in native['source_files']:assert sha(r['path'])==r['sha256']
    audit=read(ROOT/'evidence/MAIN_TRAINING_AUDIT.json');binding=read(ROOT/'model_bindings/EQUIL5120.json')
    receipt=read(ROOT/'evidence/EQUIL/checkpoint-5120.archive_receipt.json');checkpoint=next(c for c in audit['complete_checkpoints'] if c['step']==5120)
    assert receipt['status']=='UPLOADED' and receipt['entries']==checkpoint['entries']
    assert binding['source_revision']==f"{receipt['repo_id']}@{receipt['revision']}/{receipt['prefix']}" and binding['checkpoint_tree_sha256']==receipt['tree_sha256']
    assert binding['view_type']=='model-only' and binding['point']=='EQUIL5120' and binding['readout_contract_sha256']==sha(ROOT/'design/EQUIL_READOUT_R1.json')
    paths={Path(r['path']) for r in original['files']}|set((ROOT/'code').glob('*.py'))
    paths.update(ROOT/p for p in ('design/QUALITY_EXECUTION_FILESET.json','evidence/MAIN_TRAINING_AUDIT.json','evidence/EQUIL5120/FROZEN_SCOPE.json',
        'evidence/EQUIL/checkpoint-5120.archive_receipt.json','evidence/REPORT_CPU_FIXTURES.json','evidence/HISTORICAL_TABLE_REUSE.json','model_bindings/EQUIL5120.json'))
    paths.add(BASE/'PI0273/evidence/NATIVE_CPU_FIXTURES.json')
    write(ROOT/'design/QUALITY_FINAL_EXECUTION_FILESET.json',dict(status='FROZEN_PI0274_FINAL_DRIVER_BEFORE_EQUIL_QUERIES',
        files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(paths)],original_field_seal_unchanged=True,
        formal_final_only=True,old_neural_queries_repeated=0,new_model_queries=0))
    print('FROZEN_PI0274_FINAL_ENDPOINT_DRIVER',len(paths),flush=True)
if __name__=='__main__':main()
