"""Same native S4 worker; closed control is restricted to the registered field."""
import argparse
from compat import load
import quality_gate
runtime=load('quality_worker')
BASE,ROOT,R1,read,sha,write,admitted=runtime.BASE,runtime.ROOT,runtime.R1,runtime.read,runtime.sha,runtime.write,runtime.admitted
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--point',choices=('BROAD5120','EQUIL5120'),required=True);a=p.parse_args();quality_gate.ACTIVE_POINT=a.point;runtime.main(a.point)
