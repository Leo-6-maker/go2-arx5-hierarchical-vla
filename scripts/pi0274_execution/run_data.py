"""Reuse the qualified native environment for the finite zero-physics producer."""
import os,subprocess,sys
from pathlib import Path
ROOT=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0274')
sys.path.insert(0,str(ROOT.parent/'PI0273/code'))
from quality_evaluate import native_environment

if __name__=='__main__':
    python,env=native_environment();env['PYTHONPATH']=str(ROOT/'code')+':'+env['PYTHONPATH'];env['PYTHONDONTWRITEBYTECODE']='1'
    subprocess.run([str(python),'-u',str(ROOT/'code/data.py')],env=env,check=True)
