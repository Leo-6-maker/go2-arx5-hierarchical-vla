"""Reuse the original measured-state/full-H40 witness under the isolated secondary root."""
import importlib.util
import sys
from pathlib import Path

BASE=Path('/home/liuyu-nfs/Projects/go2-arx5')
sys.path.insert(0,str(BASE/'PI0266/code'))
spec=importlib.util.spec_from_file_location('pinned_pi0266_geometry_worker',BASE/'PI0266/code/pi0266_geometry_worker.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.ROOT=BASE/'PI0267/R1'
module.RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0267/R1')
if __name__=='__main__':module.main()
