"""Bounded file RPC; no model or physics runs in the login-side endpoint."""
import argparse
import json
from pathlib import Path
import time

from scripts.pi0292.durable import read
from .endpoint import Endpoint
from .journal import Journal


def call(root,role,label,args,*,endpoint=None,adjudication_name='E1_ADJUDICATION.json'):
    root=Path(root);relative='plans/'+label+'/REGISTRY.json'
    from scripts.pi0294.native_dataset import sha
    if endpoint is None:endpoint=Endpoint(root,relative,sha(root/relative),role)
    elif endpoint.root.resolve()!=root.resolve() or endpoint.role!=role:raise PermissionError('EXACT_PREVALIDATED_ENDPOINT_ROLE')
    if role=='core' and args['operation'] in ['observation','ack']:
        case=endpoint.case(args['episode']);folder=root/'episodes'/case['attempt_id']
        takeover=folder/'relay/INTERVENTION.json'
        if takeover.exists():return dict(known_intervention=read(takeover))
        p=folder/adjudication_name
        if p.exists() and read(p)['failure_stage'] not in ['OBJECT_TASK_RESULT','COMPLETE_SUCCESS']:
            return dict(known_terminal=read(p))
    invoke=lambda:getattr(endpoint,role)(**args)
    if args['operation'] in ['query','install']:
        message=args['query']['body']['query_uuid'] if role=='hopper' else args['bundle']['command']['body']['query_uuid']
        return Journal(root/'reservations'/('RPC-'+role),root.name).once(message,args,invoke)
    return invoke()


if __name__=='__main__':
    import sys
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--role',required=True);p.add_argument('--label',required=True)
    a=p.parse_args();print(json.dumps(call(a.root,a.role,a.label,json.load(sys.stdin))),flush=True)
