"""Run the single authorized physical startup with the isolated native binary."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi0287_target_sim'))
from owner import remote
root='/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0288'
candidate='/var/cache/users/liuyu-nfs/Projects/go2-arx5/executor-candidates/wbc-mujoco-6dd13098'
env=f"""unset CONDA_PREFIX PYTHONHOME PYTHONPATH LD_PRELOAD
export WBC_ROS_DOMAIN_ID=228 ROS_LOCALHOST_ONLY=1 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST CUDA_VISIBLE_DEVICES=0
export TMPDIR={root}/runtime/tmp ROS_LOG_DIR={root}/runtime/ros-logs
source {candidate}/wbc_mujoco_minimal/env.sh
export PATH={candidate}/dependency-root/usr/bin:$PATH
export LD_LIBRARY_PATH={candidate}/dependency-root/opt/ros/jazzy/lib/x86_64-linux-gnu:{candidate}/dependency-root/opt/ros/jazzy/lib:{candidate}/dependency-root/usr/lib/x86_64-linux-gnu:{candidate}/wbc_mujoco_minimal/install/lib:/opt/ros/jazzy/lib:/opt/ros/jazzy/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export PYTHONPATH={root}/project-src/src:$PYTHONPATH
exec /usr/bin/python3 {root}/initialization-repair/run.py --config {root}/initialization-repair/config.json --root {root} --mode interface-tests"""
if __name__=='__main__':
    print(remote(f"""import subprocess
p=subprocess.run(['/bin/bash','-c',{env!r}],capture_output=True,text=True)
print(p.stdout);print(p.stderr);print('EXIT',p.returncode)
"""))
