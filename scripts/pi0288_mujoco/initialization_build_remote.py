"""Executed on Corelab with a source-bound public payload supplied by owner."""
from pathlib import Path
import base64
import difflib
import hashlib
import json
import os
import shlex
import subprocess
import shutil
import time

root=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0288')
repair=root/'initialization-repair'
private=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/executor-candidates/wbc-mujoco-6dd13098/wbc_mujoco_minimal')
source=private/'src/ctrl/rl_sar/src/rl_sar'
build=private/'build/mujoco'
overlay=repair/'private-source';overlay.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
config=json.loads((root/'private/config.json').read_text())
for name,expected in config['core_hashes'].items():assert sha(private/name)==expected,name
for p in Path('/proc').iterdir():
    if not p.name.isdigit() or int(p.name)==os.getpid():continue
    try:
        cmd=(p/'cmdline').read_bytes()
        assert b'rl_sim_mujoco\x00' not in cmd and not (b'/PI0288/' in cmd and b'run.py' in cmd),'Owned runtime active'
    except (FileNotFoundError,PermissionError):pass
prior=repair/'precompiled-loader-prior'
if not prior.exists():
    prior.mkdir()
    for name in ['BUILD_AND_CPU_RECEIPT.json','config.json','HOME_SHARED_RESET_CPU_AUDIT.json','audit_home.cpp','audit_home','native_home.hpp','rl_sim_mujoco','run.py','PRIVATE_NATIVE_INITIALIZATION.diff']:
        if (repair/name).exists():shutil.copy2(repair/name,prior/name)
for name,data in payload.items():(repair/name).write_bytes(base64.b64decode(data))
(overlay/'native_home.hpp').write_bytes((repair/'native_home.hpp').read_bytes())
names={'mujoco_utils.hpp':'library/thirdparty/mujoco_simulate/mujoco_utils.hpp','rl_sim_mujoco.cpp':'src/rl_sim_mujoco.cpp','simulate.cc':'library/thirdparty/mujoco_simulate/simulate.cc','rl_sim_mujoco.hpp':'include/rl_sim_mujoco.hpp'}
expected={'mujoco_utils.hpp':'009347e0f323de261b86ecc759d699b889e444cdce0d636d58e2bf60e7e0f476','rl_sim_mujoco.cpp':'e05de9ee87d3030871743445a50b2e69e4565e3a786429b8566f24b012263d29','simulate.cc':'d423e8a611a74356a53c80fe65e41056eede3cad9904c50d3c06f9dde0db6a82'}
identities=[];diffs=[]
for name,relative in names.items():
    original=source/relative
    if name in expected:assert sha(original)==expected[name],name
    before=original.read_text();after=before
    if name=='mujoco_utils.hpp':
        assert before.count('if (mnew) dnew = mj_makeData(mnew);')==2
        assert before.count('d = mj_makeData(m);')==1
        after=after.replace('#include <mujoco/mujoco.h>','#include <mujoco/mujoco.h>\n#include "native_home.hpp"')
        after=after.replace('if (mnew) dnew = mj_makeData(mnew);','if (mnew) { dnew = mj_makeData(mnew); PI0288ResetHome(mnew, dnew); }')
        after=after.replace('d = mj_makeData(m);','d = mj_makeData(m);\n      PI0288ResetHome(m, d);')
    elif name=='rl_sim_mujoco.cpp':
        assert before.count('mj_resetData(this->mj_model, this->mj_data);')==1
        after=after.replace('mj_resetData(this->mj_model, this->mj_data);','PI0288ResetHome(this->mj_model, this->mj_data);')
        assert before.count('    // start physics thread')==1
        after=after.replace('    // start physics thread','    if (const char* compiled = std::getenv("PI0288_COMPILED_MODEL")) filename = compiled;\n\n    // start physics thread')
    elif name=='simulate.cc':
        assert before.count('mj_resetData(m_, d_);')==1 and before.count('mj_resetDataKeyframe(m_, d_, this->key);')==1
        after=after.replace('#include "simulate.h"','#include "simulate.h"\n#include "native_home.hpp"')
        after=after.replace('mj_resetData(m_, d_);','PI0288ResetHome(m_, d_);')
        after=after.replace('mj_resetDataKeyframe(m_, d_, this->key);','PI0288ResetHome(m_, d_);')
    (overlay/name).write_text(after)
    identities.append(dict(path=relative,original_sha256=sha(original),patched_sha256=sha(overlay/name)))
    diffs.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=relative,tofile=relative))
(repair/'PRIVATE_NATIVE_INITIALIZATION.diff').write_text(''.join(diffs))
flags={}
for line in (build/'CMakeFiles/rl_sim_mujoco.dir/flags.make').read_text().splitlines():
    if ' = ' in line:
        k,v=line.split(' = ',1);flags[k]=shlex.split(v)
commands=[];logs=[]
def execute(cmd,label,cwd=None):
    started=time.time();p=subprocess.run(cmd,cwd=cwd,capture_output=True,text=True)
    (repair/(label+'.log')).write_text(p.stdout+p.stderr)
    commands.append(dict(label=label,argv=cmd,exit_code=p.returncode,elapsed_s=time.time()-started))
    print(label,p.returncode,p.stderr[-1600:]);p.check_returncode()
for name in ['rl_sim_mujoco.cpp','simulate.cc']:
    cmd=['/usr/bin/c++','-I'+str(overlay)]+flags['CXX_DEFINES']+flags['CXX_INCLUDES']+flags['CXX_FLAGS']+['-o',str(repair/(name+'.o')),'-c',str(overlay/name)]
    execute(cmd,'compile-'+name)
link=shlex.split((build/'CMakeFiles/rl_sim_mujoco.dir/link.txt').read_text())
for i,arg in enumerate(link):
    if arg=='CMakeFiles/rl_sim_mujoco.dir/src/rl_sim_mujoco.cpp.o':link[i]=str(repair/'rl_sim_mujoco.cpp.o')
    if arg=='CMakeFiles/rl_sim_mujoco.dir/library/thirdparty/mujoco_simulate/simulate.cc.o':link[i]=str(repair/'simulate.cc.o')
    if arg=='bin/rl_sim_mujoco':link[i]=str(repair/'rl_sim_mujoco')
execute(link,'link',build)
lib=private/'src/ctrl/rl_sar/library/mujoco'
execute(['g++','-O2','-std=c++17','-I'+str(lib/'include'),str(repair/'audit_home.cpp'),'-L'+str(lib/'lib'),'-Wl,-rpath,'+str(lib/'lib'),'-lmujoco','-o',str(repair/'audit_home')],'compile-home-audit')
scene=private/'src/ctrl/rl_sar/src/rl_sar_zoo/go2_x5_description/mjcf/scene.xml'
execute(['g++','-O2','-std=c++17','-I'+str(lib/'include'),str(repair/'compile_model.cpp'),'-L'+str(lib/'lib'),'-Wl,-rpath,'+str(lib/'lib'),'-Wl,--export-dynamic','-lmujoco','-ldl','-o',str(repair/'compile_model')],'compile-model-cache-tool')
cache=repair/'private-scene.mjb'
c=subprocess.run([str(repair/'compile_model'),str(scene),str(cache)],capture_output=True,text=True);c.check_returncode()
(repair/'COMPILED_MODEL_PARITY.json').write_text(c.stdout)
(repair/'XML_COMPILER_SELF_TEST.jsonl').write_text(c.stderr)
parity=json.loads(c.stdout);assert parity['compiled_model_serialization_byte_equal'] and parity['compiler_self_test_steps']==1
a=subprocess.run([str(repair/'audit_home'),str(cache)],capture_output=True,text=True)
(repair/'HOME_SHARED_RESET_CPU_AUDIT.json').write_text(a.stdout);a.check_returncode()
audit=json.loads(a.stdout);assert audit['reset_exact'] and not audit['invalid_joint_count'] and not audit['illegal_touching_contacts']
manifest=dict(status='CPU_PASS_NOT_NATIVE_STARTUP_PASS',runtime_integrations=0,compiler_self_test_steps=parity['compiler_self_test_steps'],authorization='USER_DELEGATED_HOME_QUALIFICATION',initialization='DELIVERED_HOME_ONLY',source_identities=identities,commands=commands,original_core_hashes=config['core_hashes'],pinned_files={str(p):sha(p) for p in [repair/'native_home.hpp',repair/'audit_home.cpp',repair/'audit_home',repair/'rl_sim_mujoco',repair/'run.py',repair/'HOME_SHARED_RESET_CPU_AUDIT.json',cache,repair/'COMPILED_MODEL_PARITY.json']},limitations=['home is airborne at t=0','x5_joint2/3 have zero lower margin','native controller/gains/safety unchanged','live reset does not reset controller history; episodes use fresh whole stack','XML compiler self-test violation retained separately; identical MJB uses native loader without compiler self-test'])
(repair/'BUILD_AND_CPU_RECEIPT.json').write_text(json.dumps(manifest,indent=2)+'\n')
config['initialization']=dict(authorization=manifest['authorization'],pinned_files=manifest['pinned_files'],audit_binary=str(repair/'audit_home'),sim_binary=str(repair/'rl_sim_mujoco'),compiled_model=str(cache))
(repair/'config.json').write_text(json.dumps(config,indent=2)+'\n')
for name,expected_hash in manifest['original_core_hashes'].items():assert sha(private/name)==expected_hash,name
print(json.dumps(dict(status=manifest['status'],source_identities=identities,pinned_files=manifest['pinned_files'],reset_exact=audit['reset_exact'])))
