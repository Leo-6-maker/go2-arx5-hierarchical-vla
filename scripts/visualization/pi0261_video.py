"""Fixed R3 semantic cases, comparing PI0261 saved arrays without model calls."""
import json
import math
from pathlib import Path

import cv2
import numpy as np

import pi0260_r3_video as vis
from pi0260_r3_eval import R3, read, sha, source

ROOT = Path('/home/liuyu-nfs/Projects/go2-arx5/PI0261')
POINTS = {'Donor': R3 / 'eval/Base_6000/cells',
          'C2000': ROOT / 'eval/C2000/cells',
          'P2000': ROOT / 'eval/P2000/cells'}
vis.COLORS.update({'Donor': (70, 225, 245), 'C2000': (225, 170, 70),
                   'P2000': (100, 220, 95)})


def chapters():
    fixed = read(R3 / 'S4_FIXED_DISPLAY_SELECTION.json')
    cases = read(R3 / 'motion2_eval_cases.json')['cases']
    result = []
    for spec in fixed['semantic_cases']:
        assert spec['noise'] == 'B0'
        first, second = spec['case_indices']
        assert [cases[i]['command'] for i in (first, second)] == [0, 1]
        for index, opposite in ((first, second), (second, first)):
            packet, reference, case = source(dict(cohort='Motion2', case_index=index), cases, [])
            _, opposite_reference, _ = source(dict(cohort='Motion2', case_index=opposite), cases, [])
            paths = {'reference': reference, 'opposite': opposite_reference}
            metrics = {}
            for point, cells in POINTS.items():
                cell = cells / f'Motion2-{index:03d}-B0'
                record = read(cell / 'record.json')
                assert record['status'] == 'VALID' and sha(cell / 'tensors.npz') == record['tensor_sha256']
                with np.load(cell / 'tensors.npz', allow_pickle=False) as saved:
                    assert np.array_equal(saved['expert'], reference)
                    paths[point] = saved['decoded'][0].copy()
                metrics[point] = (record['metrics']['xyz_rmse_m'],
                                  record['metrics']['so3_rmse_rad'], record['gate']['decision'])
            result.append(dict(index=index, case=case, packet=packet, paths=paths, metrics=metrics))
    assert len(result) == 4
    return fixed, result


def render():
    fixed, items = chapters()
    extent = math.ceil(max(float(np.max(np.abs(path[:, :3] - item['packet']['state_v1'][:3])))
                           for item in items for path in item['paths'].values()) * 100) / 100 + .01
    out = ROOT / 'eval/media'
    out.mkdir(parents=True, exist_ok=True)
    video = out / 'PI0261_fixed_semantic_Donor_C_P.mp4'
    assert not video.exists()
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 20., (1280, 720))
    assert writer.isOpened()
    for chapter_index, item in enumerate(items, 1):
        packet = item['packet']
        for frame in range(60):
            tick = round(frame * 39 / 59)
            canvas = np.full((720, 1280, 3), (28, 25, 22), np.uint8)
            vis.line(canvas, 'KINEMATIC RENDERED REFERENCE / OFFLINE PREDICTION / NO PHYSICS OR WBC',
                     18, 28, (100, 205, 255), .56)
            vis.line(canvas, f'Frozen R3 semantic case {chapter_index}/4 | {item["case"]["task"]} '
                     f'command {item["case"]["command"]} | B0 | H40 dt=.05s | t={tick * .05:.2f}s',
                     18, 57, size=.48)
            for image, x in ((packet['body_rgb'], 20), (packet['wrist_rgb'], 275)):
                canvas[100:340, x:x + 240] = cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                    (240, 240), interpolation=cv2.INTER_NEAREST)
            vis.line(canvas, 'Saved BODY 96x96', 20, 365, size=.43)
            vis.line(canvas, 'Saved WRIST 96x96', 275, 365, size=.43)
            vis.line(canvas, item['case']['language'][:70], 20, 400, size=.43)
            vis.plot(canvas, (595, 125), 'XY', (0, 1), item['paths'],
                     packet['state_v1'][:3], extent, tick)
            vis.plot(canvas, (930, 125), 'XZ', (0, 2), item['paths'],
                     packet['state_v1'][:3], extent, tick)
            for line_number, name in enumerate(item['paths']):
                y = 445 + line_number * 34
                cv2.line(canvas, (600, y - 5), (630, y - 5), vis.COLORS[name], 3)
                detail = item['metrics'].get(name)
                label = name if detail is None else (
                    f'{name}: XYZ {detail[0]:.4f}m, SO3 {detail[1]:.3f}rad, gate {detail[2]}')
                vis.line(canvas, label, 640, y, size=.42)
            vis.line(canvas, 'Fixed synthetic entry comparisons; no deployment or physical task claim.',
                     18, 690, size=.44)
            writer.write(canvas)
    writer.release()
    capture = cv2.VideoCapture(str(video))
    frames = 0
    while capture.read()[0]:
        frames += 1
    capture.release()
    assert frames == 240
    receipt = dict(schema='PI0261_FIXED_SAVED_ARRAY_VIDEO_V1', status='PASS_VIDEO_DECODE',
                   video=video.name, sha256=sha(video), frames=frames, fps=20,
                   selection_sha256=sha(R3 / 'S4_FIXED_DISPLAY_SELECTION.json'),
                   chapters=[item['index'] for item in items], model_queries=0, physics_steps=0)
    (out / 'video_receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    render()
