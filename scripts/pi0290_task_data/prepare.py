from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
from go2arx5_vla.task_data.config import PARAMETERS as P,AUTHORITY,CANONICAL,COMMISSIONING,PILOTS,layout,canonical_hash
from go2arx5_vla.task_data.scene import freeze_description,xml_scene
from go2arx5_vla.task_data.io import write,file_hash
from go2arx5_vla.task_data.resource import Budget


def seal_code(root,source):
    code=root/'code';files={p.relative_to(code).as_posix():file_hash(p) for folder in ['src','scripts','tests'] for p in (code/folder).rglob('*.py')}
    write(root/'CODE_SEAL.json',dict(source_sha=source,files=files,authority=AUTHORITY,canonical=CANONICAL))
    return files


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source-sha',required=True)
    p.add_argument('--pilot-seal',action='store_true');a=p.parse_args();root=Path(a.root)
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
    if a.pilot_seal:
        budget=Budget(root)
        if (root/'TEACHER_AND_LAYOUT_SEAL.json').exists():raise RuntimeError('Pilot freeze already exists; do not overwrite')
        assert not any(x['id'].startswith('P') for x in json.loads((root/'RESOURCE_ACCOUNTING.json').read_text())['starts'])
        results=[json.loads((root/'episodes'/x/'RESULT.json').read_text()) for x in COMMISSIONING]
        assert results[0]['score']['history']['sustained_grasp'] or results[3]['score']['history']['sustained_grasp'],'D1 lacks real grasp'
        assert not results[1]['score']['history']['sustained_grasp'] and not results[2]['score']['history']['sustained_grasp'],'Negative contact check failed'
        audit=json.loads((root/'evidence/D1_DATA_AUDIT.json').read_text())
        assert audit['status']=='PASS_STATIC_DATA_AND_NATIVE_CODEC_ONLY'
        assert audit['episodes'][2]['contact_samples']['left']>0 and audit['episodes'][2]['contact_samples']['right']==0
        assert json.loads((root/'evidence/CAMERA_QUALIFICATION_X_SIDE.json').read_text())['status']=='PASS'
        assert json.loads((root/'evidence/LAYOUT_PREFLIGHT.json').read_text())['status']=='PASS'
        def physics_xml(text):
            tree=ET.fromstring(text)
            for parent in tree.iter():
                for child in list(parent):
                    if child.tag=='camera':parent.remove(child)
                parent.text=None;parent.tail=None
            return ET.tostring(tree)
        old_xml=json.loads((root/'episodes/C2903/scene.xml.json').read_text())['xml']
        assert physics_xml(old_xml)==physics_xml(xml_scene(layout('C2903'))),'Optics repair changed physical XML'
        files=seal_code(root,a.source_sha)
        value=dict(schema='PI0290_TEACHER_AND_LAYOUT_SEAL_V1',authority=AUTHORITY,source_sha=a.source_sha,
            parameter_sha256=canonical_hash(P),parameters=P,teacher_version=P['version'],maximum_teacher_versions=2,
            layouts={x:layout(x) for x in COMMISSIONING+PILOTS},sampling='ID SHA256 first8 bytes -> PCG64; fixed legal ranges in config.py',
            pilot_seen=list(PILOTS[:4]),pilot_heldout=list(PILOTS[4:]),sealed_before_first_pilot=True,code_files=files)
        write(root/'TEACHER_AND_LAYOUT_SEAL.json',value)
        write(root/'LAYOUT_SEAL.json',dict(source_sha=a.source_sha,layouts=value['layouts'],sampling=value['sampling']))
        write(root/'SCENE_AND_EXECUTOR_FREEZE.json',dict(source_sha=a.source_sha,**freeze_description(),
            scene_xml_sha256={x:__import__('hashlib').sha256(xml_scene(layout(x)).encode()).hexdigest() for x in COMMISSIONING+PILOTS},
            scorer_source_sha256=file_hash(root/'code/src/go2arx5_vla/target_sim/scorer.py'),
            score='existing engine-independent T1 physical predicates; no numerical changes',
            clock='MuJoCo sim0..32s, dt2ms, sample every25 integrations, zero cross-camera skew',
            render_backend='Mesa llvmpipe CPU, not other users GPU',
            preflight_sha256=file_hash(root/'evidence/LAYOUT_PREFLIGHT.json'),
            camera_qualification_sha256=file_hash(root/'evidence/CAMERA_QUALIFICATION_X_SIDE.json'),
            commissioning_camera_caveat='Original wrist palm occlusion; optics-only repair qualified on saved states before pilots; no original RGB replaced'))
        budget.value['optics_repair_physical_xml_unchanged']=True
        budget.finish_cpu('D2_PREPILOT_SEAL')
        print(json.dumps({'status':'PILOT_PRESEALED','layout_seal_sha256':file_hash(root/'TEACHER_AND_LAYOUT_SEAL.json')}))
    else:
        budget=Budget(root);files=seal_code(root,a.source_sha)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
        result=subprocess.run([str(root/'runtime/venv/bin/python'),'-B','-m','pytest','-q',str(root/'code/tests/task_data')],capture_output=True,text=True,env=env)
        write(root/'evidence/CPU_TESTS.json',dict(status='PASS' if result.returncode==0 else 'FAIL',source_sha=a.source_sha,
            exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr,source_files=files,physics_integrations=0,GR00T=0,WBC=0))
        budget.finish_cpu('CPU_TESTS')
        print(result.stdout,result.stderr,flush=True)
        raise SystemExit(result.returncode)


if __name__=='__main__':main()
