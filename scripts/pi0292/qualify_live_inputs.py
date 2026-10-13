"""Real bank RGB comparison of observation-only official CPU processing.

Recorded data is used only for an input-equivalence test. It is not a new live
observation and cannot satisfy physical relay qualification.
"""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import signal
import uuid

from PIL import Image

from .durable import read,write
from .eval_contract import FRAME,TCP
from .live_inputs import prepare
from .train_adapter import NativeDataset,ProcessedCollator,sha
from .qualify_adapter import inventory


def run(args):
    import torch
    torch.set_num_threads(1)
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_CPU,(300,310))
    def expired(signum,frame):
        raise TimeoutError('OBSERVATION_INPUT_CPU_300_SECOND_LIMIT')
    signal.signal(signal.SIGALRM,expired);signal.alarm(300)
    root=Path(args.root);bindings=read(args.bindings);records=[]
    for split in ('TRAIN','DEV'):
        dataset=NativeDataset(bindings['dataset'],bindings['source'],bindings['cosmos'],split,raw_bindings=read(bindings['raw_bindings']))
        dataset.initialize();before=sha(dataset.root/'statistics.json')
        for index in (0,len(dataset)//2,len(dataset)-1):
            state,unused_expert,images,text=dataset.raw(index)
            images_payload=[]
            for camera,pixels in zip(('external_rgb','wrist_rgb'),images):
                output=io.BytesIO();Image.fromarray(pixels).save(output,format='PNG');raw=output.getvalue()
                images_payload.append(dict(camera=camera,png_base64=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest()))
            payload=dict(snapshot_uuid=str(uuid.uuid4()),language=text,state_eef10=state.tolist(),images=images_payload,frame=FRAME,tcp=TCP)
            reference=ProcessedCollator(dataset.processor)([dataset[index]])
            del reference['inputs']['action'];del reference['inputs']['action_mask']
            actual=prepare(dataset.processor,payload)
            expected=inventory(reference);observed=inventory(actual)
            if expected!=observed:
                raise ValueError('OBSERVATION_ONLY_NATIVE_INFERENCE_TENSOR_MISMATCH')
            records.append(dict(split=split,index=index,identity=dataset.identity(index),tensor_inventory=observed,
                future_action_present=False,privileged_task_coordinates_present=False))
        if before!=sha(dataset.root/'statistics.json'):
            raise ValueError('FROZEN_STATISTICS_CHANGED')
    if torch.cuda.is_initialized():
        raise ValueError('CPU_ONLY_PROCESSOR_NO_CUDA')
    u=resource.getrusage(resource.RUSAGE_SELF);c=resource.getrusage(resource.RUSAGE_CHILDREN)
    result=dict(status='PASS_RECORDED_RGB_OBSERVATION_ONLY_CPU_EQUIVALENCE_NOT_LIVE_PHYSICS',records=records,
        model_constructed=False,CUDA_initialized=False,live_observation=False,physics_integrations=0,
        statistics_sha256=sha(Path(bindings['dataset'])/'statistics.json'),CPU_s=u.ru_utime+u.ru_stime+c.ru_utime+c.ru_stime)
    if Path(args.output).exists():
        raise FileExistsError('Preserve the original input qualification')
    write(args.output,result)
    signal.alarm(0)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('root','bindings','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NO_ALBUMENTATIONS_UPDATE='1',TOKENIZERS_PARALLELISM='false')
    print(json.dumps(run(args)))
