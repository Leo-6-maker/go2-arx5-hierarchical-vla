"""Reuse qualified packet/file helpers with isolated PI0278 roots."""
import importlib.util
from pathlib import Path
COMMON=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0278');ROOT=COMMON/'R2'
OLD=COMMON.with_name('PI0276')
RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0278/R2');OLD_RAW=RAW.parent.with_name('PI0276')
RELEASE='83af9099f3bab9a4ac08eaa85380c2cfd6351f34'
spec=importlib.util.spec_from_file_location('qualified_card',COMMON.with_name('PI0277')/'R2/code/evaluation_card.py')
qualified=importlib.util.module_from_spec(spec);spec.loader.exec_module(qualified)
read,sha,write,freeze,packet=qualified.read,qualified.sha,qualified.write,qualified.freeze,qualified.packet
