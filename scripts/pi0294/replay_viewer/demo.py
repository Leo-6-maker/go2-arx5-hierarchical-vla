"""Generate four unmistakably synthetic UI fixtures, never scientific episodes."""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jWZkAAAAASUVORK5CYII=')


def generate(destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    config = {'episodes': []}
    for eid, kind, task, outcome in [('synthetic-t1-teacher','teacher','T1','FIXTURE_SUCCESS'),
                                      ('synthetic-t2-teacher','teacher','T2','FIXTURE_SUCCESS'),
                                      ('synthetic-t1-student','student_eval','T1','FIXTURE_SUCCESS'),
                                      ('synthetic-s1-student','student_eval','T1','FIXTURE_S1_REJECT')]:
        root=destination/eid;root.mkdir();rows=[]
        for i in range(40):
            t=i*.05
            def pose(x,y,z):return [[1,0,0,x],[0,1,0,y],[0,0,1,z],[0,0,0,1]]
            tcp=pose(.2+.1*math.sin(t),.05*math.cos(t),.25+.04*math.sin(t*2))
            cam={}
            for c in ('body_rgb','wrist_rgb'):
                p=root/c/f'{i:04d}.png';p.parent.mkdir(exist_ok=True);p.write_bytes(PNG)
                cam[c]=dict(path=p.relative_to(root).as_posix(),sha256=hashlib.sha256(PNG).hexdigest(),timestamp_s=t)
            rows.append(dict(policy_obs=dict(timestamp_s=t,actual_tcp=tcp,actual_gripper=.5,instruction='SYNTHETIC UI FIXTURE ONLY',**cam),
                execution={},privileged=dict(object_pose=pose(.3,.03,.2),teacher_stage='SYNTHETIC'),
                supervision=dict(desired=pose(.3,.05,.3),committed=tcp,actually_committed=True)))
        samples=root/'samples.json';samples.write_text(json.dumps(rows))
        audit=root/'audit.json';audit.write_text(json.dumps(dict(status=outcome,fixture=True,scientific_result=False)))
        config['episodes'].append(dict(id=eid,source_kind=kind,task=task,fixture=True,format='fixed_base',root=str(root),
            samples_sha256=hashlib.sha256(samples.read_bytes()).hexdigest(),audit_path='audit.json',audit_sha256=hashlib.sha256(audit.read_bytes()).hexdigest(),
            provenance=dict(origin='synthetic_ui_test',actual_native_episode=False)))
    (destination/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    return destination/'config.json'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    print(generate(p.parse_args().output))
