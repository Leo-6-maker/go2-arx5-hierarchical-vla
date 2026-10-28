"""Original fixed component starts; no added task qualification."""
import argparse
from compat import load
runtime=load('quality_components');globals().update({k:v for k,v in vars(runtime).items() if not k.startswith('__')})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--point',choices=('EQUIL5120',),required=True);p.add_argument('--family',required=True);a=p.parse_args();main(a.point,a.family)
