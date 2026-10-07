"""Reuse the qualified stdlib byte comparison for this point's559 frozen tensors."""
import importlib.util,json,sys
from pathlib import Path
sys.path.insert(0,'/home/liuyu-nfs/Projects/go2-arx5/PI0278/R2/code')
from evaluation_card import COMMON,ROOT,RAW,read,sha,freeze
source=COMMON.with_name('PI0277')/'R3/code/frozen_bytes.py'
spec=importlib.util.spec_from_file_location('qualified_frozen',source);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
if __name__=='__main__':
    point=sys.argv[1];scope=COMMON/'evidence/COVER93_PARAMETER_SCOPE.json'
    binding=read(ROOT/'model_bindings'/f'{point}.json');assert point.startswith('COVER93')
    model=Path(binding['view'])
    for name,h in binding['model_files_sha256'].items():assert sha(model/name)==h
    parent=Path(read(COMMON.with_name('PI0277')/'R2/model_bindings/FOCUS4096.json')['view'])
    result=old.compare(parent,model,read(scope));result.update(point=point,scope_sha256=sha(scope),comparison_source_sha256=sha(source))
    freeze(ROOT/'evidence'/f'FROZEN_BYTES_{point}.json',result);print(result['status'])
