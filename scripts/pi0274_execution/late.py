"""Original296 support states at original j0/j1."""
import argparse
from compat import load
runtime=load('late');globals().update({k:v for k,v in vars(runtime).items() if not k.startswith('__')})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--point',choices=('EQUIL5120',),required=True);runtime.newfit.evaluate=evaluate;runtime.newfit.main(p.parse_args().point)
