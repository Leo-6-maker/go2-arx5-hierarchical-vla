"""Two distinct registered physical controls, serial and never quality-retried."""
import argparse
from .run_episode import run

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source-sha',required=True)
    a=p.parse_args()
    for identifier,mode in [('C29101','empty_grasp'),('C29102','single_side')]:
        run(a.root,identifier,a.source_sha,mode,'mesa_cpu')
