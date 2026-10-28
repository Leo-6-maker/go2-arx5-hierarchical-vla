"""Original fixed192, with actual EQUIL consumption audited after training."""
import argparse
from compat import load
runtime=load('newfit');globals().update({k:v for k,v in vars(runtime).items() if not k.startswith('__')})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--point',choices=('EQUIL5120',),required=True);main(p.parse_args().point)
