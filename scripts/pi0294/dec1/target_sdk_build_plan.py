"""Pure cold build plan for OUR companion against the installed original SDK.

Never executes a compiler, changes vendor CMake, builds vendor sources or
produces admission. The current admitted owner must seal/hash every returned
input and run the two argv lists within its original CPU/output bounds.
"""
import hashlib
import posixpath
from pathlib import PurePosixPath
import shlex

FLAGS_SHA256 = 'b9d8d124223c89216d5401b16455f55db5c9247b8d21a870827f4f7aef55f371'
LINK_SHA256 = '242b8c4088cb5f5dc3854a5432f3c4748afa15fbd5145124040931e1cc7c3a5f'
NODE_OBJECT = 'CMakeFiles/mobile_manipulator_mpc_node.dir/src/MobileManipulatorMpcNode.cpp.o'
DEPENDENCY_SHARED_SHA256 = {
    'liboctomap.so.1.9': 'ec8bfd6cce31e7fd3dced8d5489109f6bca3d066c8ac3035b032940a8eb9ef50',
    'liboctomath.so.1.9': 'efd1dcfc111d271e13173f02153c0a950bbce9ccd53d1319990a5e6ff56c8dca',
}


def need(ok, why):
    if not ok: raise PermissionError('DEC1_SDK_BUILD_' + why)


def absolute(value):
    need(type(value) is str and value.startswith('/') and '\x00' not in value, 'EXPLICIT_NATIVE_ABSOLUTE_PATH')
    need(str(PurePosixPath(value)) == value and '..' not in PurePosixPath(value).parts, 'NORMALIZED_BOUND_NATIVE_PATH')
    return value


def plan(flags_bytes, link_bytes, *, sdk_root, sealed_client_directory, registered_build_directory):
    need(type(flags_bytes) is bytes and hashlib.sha256(flags_bytes).hexdigest() == FLAGS_SHA256
         and type(link_bytes) is bytes and hashlib.sha256(link_bytes).hexdigest() == LINK_SHA256,
         'EXACT_ACTUAL_INSTALLED_SDK_BUILD_RECORDS')
    sdk = absolute(sdk_root); source = absolute(sealed_client_directory); out = absolute(registered_build_directory)
    need(source != out and not (out == sdk or out.startswith(sdk+'/')),
         'NEW_REGISTERED_OUTPUT_NOT_VENDOR_TREE')
    build = sdk+'/build/ocs2_mobile_manipulator_ros'
    variables = {}
    for line in flags_bytes.decode().splitlines():
        if not line.startswith(('CXX_DEFINES =', 'CXX_INCLUDES =', 'CXX_FLAGS =')): continue
        key, value = line.split('=', 1); key = key.strip()
        need(key not in variables, 'UNIQUE_BUILD_VARIABLES')
        variables[key] = shlex.split(value)
    need(set(variables) == {'CXX_DEFINES', 'CXX_INCLUDES', 'CXX_FLAGS'}, 'ORIGINAL_ABI_FLAGS_AND_INCLUDES')
    link = shlex.split(link_bytes.decode())
    need(link[0] == '/usr/bin/c++' and link.count('-o') == 1 and link.count(NODE_OBJECT) == 1
         and sum(x.endswith('.o') for x in link) == 1
         and link[link.index('-o')+1] == 'mobile_manipulator_mpc_node', 'EXACT_ORIGINAL_NODE_LINK_SHAPE')
    need(not any(x.startswith('@') or x.startswith('-Wl,@') for x in link), 'NO_IMPLICIT_RESPONSE_FILE')
    obj = out+'/target_sdk_client.o'; binary = out+'/target_sdk_client'
    link[link.index(NODE_OBJECT)] = obj; link[link.index('-o')+1] = binary
    inputs = []
    for i, value in enumerate(link):
        if value.endswith(('.a', '.so')) or '.so.' in value and not value.startswith('-'):
            if not value.startswith('/'): value = posixpath.normpath(build+'/'+value)
            link[i] = absolute(value);inputs.append(value)
    need(inputs and all(not x.endswith('.cpp') for x in link), 'LINK_EXISTING_SDK_ONLY')
    compile_argv = ['/usr/bin/c++', *variables['CXX_DEFINES'], *variables['CXX_INCLUDES'],
                    *variables['CXX_FLAGS'], '-pthread', '-I'+source, '-MMD', '-MF', out+'/target_sdk_client.d',
                    '-c', source+'/target_sdk_client.cpp', '-o', obj]
    # Own reader uses std::thread ownership; do not rely on implicit pthread
    # linkage in an unrelated original ROS target.
    link.append('-pthread')
    # The original ROS node relied on its launch environment for these two
    # transitive libcoal dependencies. Our independent companion must bind
    # the exact delivered libraries itself, including runtime resolution.
    dependency_directory = str(PurePosixPath(sdk).parent/'dependency-root/usr/lib/x86_64-linux-gnu')
    need(not any(c in dependency_directory for c in (',', ':', '\n')), 'UNAMBIGUOUS_DEPENDENCY_SEARCH_DIRECTORY')
    dependencies = {dependency_directory+'/'+name: pin for name,pin in DEPENDENCY_SHARED_SHA256.items()}
    link.extend(['-Wl,--push-state,--no-as-needed', *dependencies, '-Wl,--pop-state',
                 '-Wl,-rpath-link,'+dependency_directory, '-Wl,-rpath,'+dependency_directory])
    inputs.extend(dependencies)
    # RUNPATH only applies to direct dependencies. The installed original SDK
    # also has indirect dependencies (including HPIPM); retain every pinned
    # shared-library directory as transitive RPATH on OUR binary only.
    runtime_directories = sorted({str(PurePosixPath(p).parent) for p in inputs
                                  if p.endswith('.so') or '.so.' in PurePosixPath(p).name})
    need(all(not any(c in p for c in (',', ':', '\n')) for p in runtime_directories),
         'UNAMBIGUOUS_ALL_RUNTIME_DIRECTORIES')
    link.extend(['-Wl,--disable-new-dtags', '-Wl,-rpath,'+':'.join(runtime_directories)])
    return dict(status='PURE_COMPANION_BUILD_PLAN_NOT_NATIVE_ADMISSION',
                compile_argv=compile_argv, link_argv=link, cwd=out,
                new_sources=[source+'/target_sdk_client.cpp', source+'/target_sdk_constraints.hpp'],
                existing_link_inputs=inputs, dependency_file=out+'/target_sdk_client.d', binary=binary,
                dependent_shared_library_sha256=dependencies,
                runtime_library_directories=runtime_directories,
                vendor_sources_compiled=[], vendor_binaries_modified=False,
                model_constructors_during_build=0, native_action=False)
