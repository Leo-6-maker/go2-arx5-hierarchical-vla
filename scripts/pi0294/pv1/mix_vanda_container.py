"""Original source physics container, with explicit current PBS propagation.

Command assembly only. It grants no allocation, native import or physics; the
sealed entry validates its own real qstat/request/lease before native imports.
"""
import re
from pathlib import Path

PHYSICS_ENV = dict(PYTHONDONTWRITEBYTECODE='1', LIBGL_ALWAYS_SOFTWARE='1',
    CUDA_VISIBLE_DEVICES='', __EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/50_mesa.json',
    MUJOCO_GL='egl', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2',
    NUMEXPR_NUM_THREADS='1',LP_NUM_THREADS='1')


def need(ok, why):
    if not ok: raise PermissionError('V5_VANDA_CONTAINER_'+why)


def physics_environment(environ, *, code_root, worker_manifest=None,audit_codec_bin=None):
    job = environ.get('PBS_JOBID')
    need(type(job) is str and re.fullmatch(r'[0-9]+\.stdct-mgmt-02',job), 'INHERIT_CURRENT_PBS_NOT_CALLER_ID')
    need(Path(code_root).is_absolute() and ':' not in code_root, 'EXACT_SEALED_SOURCE_ROOT')
    critical = set(PHYSICS_ENV)|{'PBS_JOBID','PYTHONPATH','MUJOCO_EGL_DEVICE_ID','PV1_RUN_MANIFEST','USER','PATH',
                               'LD_LIBRARY_PATH','LD_PRELOAD','PYTHONHOME'}
    need(not any(k in environ for prefix in ('APPTAINERENV_','SINGULARITYENV_')
                 for k in (prefix+n for n in critical)), 'NO_HIDDEN_CONTAINER_ENVIRONMENT_OVERRIDE')
    env=dict(PHYSICS_ENV, PBS_JOBID=job, USER='e1192548',PYTHONPATH=code_root+':'+code_root+'/src')
    if worker_manifest is not None:
        need(worker_manifest==str(Path(code_root).parent/'WORKER_RUN_MANIFEST.json'), 'EXACT_REGISTERED_WORKER_MANIFEST_NOT_ANOTHER_ROLE')
        env['PV1_RUN_MANIFEST']=worker_manifest
    if audit_codec_bin is not None:
        need(audit_codec_bin==str(Path(code_root).parents[2]/'audit-codec/bin'),'ONLY_REGISTERED_ROLE_CODEC_BIN')
        env['PATH']=audit_codec_bin+':/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
    return env


def physics_command(environ, *, code_root, container, readonly, writable, module, args=(),worker_manifest=None,audit_codec_bin=None):
    need(module in ('scripts.pi0294.pv1.mix_vanda_render','scripts.pi0294.pv1.mix_vanda_eval_render','scripts.pi0294.core',
                    'scripts.pi0294.relay_file_rpc','scripts.pi0294.episode'),
         'ONLY_EXISTING_SOURCE_WORKERS_AND_FINITE_RENDER_HOOK')
    env = physics_environment(environ,code_root=code_root,worker_manifest=worker_manifest,audit_codec_bin=audit_codec_bin)
    command = ['/cm/shared/apps/apptainer/1.4.1/bin/apptainer','exec','--cleanenv']
    for path,mode in [(p,'ro')for p in readonly]+[(p,'rw')for p in writable]:
        need(type(path) is str and Path(path).is_absolute() and ':' not in path,
             'EXACT_REGISTERED_BIND_PATH_NO_MOUNT_SYNTAX_INJECTION')
        command += ['--bind',path+':'+path+':'+mode]
    for k,v in env.items(): command += ['--env',k+'='+v]
    need(Path(container).is_absolute(), 'ORIGINAL_PINNED_CONTAINER_PATH')
    return command+[container,'/opt/wbc-python/bin/python','-u','-m',module,*args]
