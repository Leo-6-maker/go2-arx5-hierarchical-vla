"""Compile an isolated native startup patch; never edit the delivered workspace."""
import base64
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi0287_target_sim'))
from owner import remote

if __name__=='__main__':
    worktree=Path(__file__).resolve().parents[2]
    files=['src/go2arx5_vla/target_mujoco/native_home.hpp','src/go2arx5_vla/target_mujoco/audit_home.cpp','src/go2arx5_vla/target_mujoco/compile_model.cpp','scripts/pi0288_mujoco/run.py']
    payload={Path(f).name:base64.b64encode((worktree/f).read_bytes()).decode() for f in files}
    body=(Path(__file__).with_name('initialization_build_remote.py')).read_text()
    print(remote('payload='+repr(payload)+'\n'+body))
