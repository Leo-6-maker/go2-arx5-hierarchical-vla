"""Restore the qualified import paths for the unchanged saved-only PI0270 report."""
import argparse
import os
import runpy
import sys
from pathlib import Path

os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',OPENBLAS_CORETYPE='Prescott')
ROOT=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0270')
sys.path.insert(0,str(ROOT/'code'))
import quality_worker  # Its imports establish the actual inference/codec paths; main is never called.
from quality_gate import read,sha,admitted


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    admitted()
    assert sha(ROOT/'code/quality_report.py')=='f7b2760b11d25ba6173b2b00b880f615f35c4b777cc869a87d7f6ab7a15c071b'
    import pi0260_r2_codec as codec
    processor=codec.processor(read(ROOT/'contracts/ROT/source/dataset/statistics.json'))
    if args.check_only:print('PASS_ACTUAL_PROCESSOR_IMPORT_WITH_QUALIFIED_PATHS_NO_MODEL_QUERY',type(processor).__name__)
    else:
        import quality_report
        quality_report.main()
        runpy.run_path(str(ROOT/'code/video.py'),run_name='__main__')
