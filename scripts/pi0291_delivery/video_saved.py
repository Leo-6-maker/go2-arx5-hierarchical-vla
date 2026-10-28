"""Encode already-saved RGB; no renderer, physics, or physical-owner lock."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def main(root, identifier):
    os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[-2:]))
    begin = time.monotonic()
    folder = Path(root) / 'episodes' / identifier
    result = json.loads((folder/'RESULT.json').read_text())
    assert result['status'] != 'STARTED'
    samples = json.loads((folder/'samples.json').read_text())
    output = folder/'ACTUAL_DUAL_RGB.mp4'
    if output.exists():
        raise FileExistsError('Preserve the existing original video')
    with imageio.get_writer(output, fps=20, codec='libx264', quality=7,
            macro_block_size=16, ffmpeg_params=['-threads','2','-pix_fmt','yuv420p']) as writer:
        for index, sample in enumerate(samples):
            images = []
            for name in ('body_rgb','wrist_rgb'):
                row = sample['policy_obs'][name]
                assert row['frame_index'] == index and row['snapshot_id'] == sample['snapshot_id']
                path = folder/row['path']
                assert sha(path) == row['sha256']
                images.append(Image.open(path).convert('RGB'))
            canvas = Image.new('RGB',(512,288),'#202528')
            canvas.paste(images[0],(0,32));canvas.paste(images[1],(256,32))
            draw = ImageDraw.Draw(canvas)
            draw.text((6,2),f'PI0291 {identifier}  t={index/20:05.2f}s  SIM_ONLY / NOT_WBC',fill='white')
            draw.text((6,18),'BODY RGB',fill='white');draw.text((262,18),'WRIST RGB',fill='white')
            writer.append_data(np.asarray(canvas))
    with imageio.get_reader(output) as reader:
        decoded = sum(1 for _ in reader)
    assert decoded == len(samples)
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    receipt = dict(status='PASS_ACTUAL_SAVED_RGB_HASHED_FULL_DECODE', id=identifier,
        frames=decoded, fps=20, first_sim_time_s=0, last_sim_time_s=(decoded-1)/20,
        sha256=sha(output), bytes=output.stat().st_size, CPU_s=own.ru_utime+own.ru_stime+children.ru_utime+children.ru_stime,
        elapsed_s=time.monotonic()-begin, new_render=0, new_physics=0,
        source_sha=result['source_sha'], delivery_code_sha256=sha(__file__), domain=result['domain'])
    (folder/'VIDEO_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    paths = sorted(path for path in folder.rglob('*') if path.is_file() and path.name != 'FILE_MANIFEST.json')
    manifest = [dict(path=path.relative_to(folder).as_posix(), bytes=path.stat().st_size,sha256=sha(path)) for path in paths]
    (folder/'FILE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--id',required=True)
    args = parser.parse_args()
    main(args.root,args.id)
