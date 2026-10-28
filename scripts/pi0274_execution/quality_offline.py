"""Original offline metrics and evaluator, registered EQUIL endpoint only."""
import argparse
from compat import load
runtime=load('quality_offline');globals().update({k:v for k,v in vars(runtime).items() if not k.startswith('__')})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--point',choices=('EQUIL5120',),required=True);main(p.parse_args().point)
