"""Fixed-case, saved-array-only PI0260-R3 comparison video."""
import json
import math
from pathlib import Path

import cv2
import numpy as np

from pi0260_r3_eval import R3, read, sequence, sha, source

POINTS = ('Base_0', 'Base_6000', 'G_0', 'G_6000')
COLORS = {'opposite': (95, 95, 95), 'reference': (245, 245, 245),
          'Base_0': (60, 145, 245), 'Base_6000': (70, 225, 245),
          'G_0': (225, 170, 70), 'G_6000': (100, 220, 95),
          'alias t0': (245, 245, 245), 'alias t0.8': (245, 120, 180)}


def chapter(case, opposite, index, packet):
    paths = {'reference': source(dict(cohort='Motion2', case_index=index),
                                 read(R3 / 'motion2_eval_cases.json')['cases'], [])[1],
             'opposite': source(dict(cohort='Motion2', case_index=opposite),
                                read(R3 / 'motion2_eval_cases.json')['cases'], [])[1]}
    details = {}
    for point in POINTS:
        cell = R3 / 'eval' / point / 'cells' / f'Motion2-{index:03d}-B0'
        record = read(cell / 'record.json')
        tensor = cell / 'tensors.npz'
        assert sha(tensor) == record['tensor_sha256']
        with np.load(tensor, allow_pickle=False) as saved:
            assert np.array_equal(saved['expert'], paths['reference'])
            paths[point] = saved['decoded'][0].copy()
        details[point] = (record['metrics']['xyz_rmse_m'], record['metrics']['so3_rmse_rad'],
                          record['gate']['decision'])
    return dict(title=f"{case['task']} command {case['command']} | B0",
                instruction=case['language'], packet=packet, origin=packet['state_v1'][:3],
                paths=paths, details=details, source_case=index, sequence=case['sequence'])


def chapters():
    fixed = read(R3 / 'S4_FIXED_DISPLAY_SELECTION.json')
    cases = read(R3 / 'motion2_eval_cases.json')['cases']
    result = []
    for spec in fixed['semantic_cases']:
        first, second = spec['case_indices']
        assert spec['noise'] == 'B0' and [cases[i]['command'] for i in (first, second)] == [0, 1]
        for index, opposite in ((first, second), (second, first)):
            packet, _, case = source(dict(cohort='Motion2', case_index=index), cases, [])
            result.append(chapter(case, opposite, index, packet))
    alias = fixed['alias_example']
    record, states, labels = sequence(alias['sequence'])
    a, b = alias['indices']
    assert record['rows'][a]['time'] == alias['times_s'][0]
    assert record['rows'][b]['time'] == alias['times_s'][1]
    assert record['rows'][a]['index'] == a and record['rows'][b]['index'] == b
    assert record['rows'][a]['frame_sha256'] == record['rows'][b]['frame_sha256']
    assert np.array_equal(states[a], states[b]) and np.max(np.abs(labels[a] - labels[b])) > 0
    packet, _, _ = source(dict(cohort='Motion2', case_index=fixed['semantic_cases'][0]['case_indices'][0]), cases, [])
    assert record['rows'][a]['frame_sha256'] == cases[fixed['semantic_cases'][0]['case_indices'][0]]['frame_sha256']
    result.append(dict(title='Measured hidden-time alias | same observable M1 input',
                       instruction=record['sequence']['instruction'], packet=packet, origin=states[a, :3],
                       paths={'alias t0': labels[a], 'alias t0.8': labels[b]}, details={},
                       source_case='KIN-VAL alias indices 0,16', sequence=alias['sequence']))
    return fixed, result


def line(canvas, words, x, y, color=(230, 235, 240), size=.48):
    cv2.putText(canvas, words, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def plot(canvas, center, title, axes, paths, origin, extent, tick):
    x, y, side = *center, 305
    cv2.rectangle(canvas, (x, y), (x + side, y + side), (82, 88, 98), 1)
    cv2.line(canvas, (x + side // 2, y), (x + side // 2, y + side), (55, 60, 70), 1)
    cv2.line(canvas, (x, y + side // 2), (x + side, y + side // 2), (55, 60, 70), 1)
    line(canvas, f'{title} | +/-{extent:.2f} m from initial TCP', x, y - 10, size=.40)
    for name, values in paths.items():
        positions = np.asarray(values[:tick + 1, :3], np.float64)
        points = np.rint(np.stack([x + side / 2 + (positions[:, axes[0]] - origin[axes[0]]) * side / (2 * extent),
                                  y + side / 2 - (positions[:, axes[1]] - origin[axes[1]]) * side / (2 * extent)], axis=1)).astype(np.int32)
        color = COLORS[name]
        if len(points) > 1:
            cv2.polylines(canvas, [points], False, color, 2 if name != 'opposite' else 1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(points[-1]), 3, color, -1)


def render():
    fixed, items = chapters()
    assert len(items) == 5 and fixed['points'] == list(POINTS)
    extent = math.ceil(max(float(np.max(np.abs(path[:, :3] - item['origin'])))
                           for item in items for path in item['paths'].values()) * 100) / 100 + .01
    output = R3 / 'eval' / 'media'
    output.mkdir(parents=True, exist_ok=True)
    video = output / 'PI0260_R3_Base_G_before_after.mp4'
    assert not video.exists()
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 20., (1280, 720))
    assert writer.isOpened()
    for chapter_index, item in enumerate(items, 1):
        body, wrist = item['packet']['body_rgb'], item['packet']['wrist_rgb']
        assert body.shape == wrist.shape == (96, 96, 3)
        for frame in range(60):
            tick = round(frame * 39 / 59)
            canvas = np.full((720, 1280, 3), (28, 25, 22), np.uint8)
            line(canvas, 'KINEMATIC RENDERED REFERENCE / OFFLINE PREDICTION / NO PHYSICS OR WBC', 18, 28, (100, 205, 255), .56)
            line(canvas, f'Fixed chapter {chapter_index}/5 | {item["title"]} | H40 dt=0.05s | t={tick * .05:.2f}s', 18, 57, size=.50)
            canvas[100:340, 20:260] = cv2.resize(cv2.cvtColor(body, cv2.COLOR_RGB2BGR), (240, 240), interpolation=cv2.INTER_NEAREST)
            canvas[100:340, 275:515] = cv2.resize(cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR), (240, 240), interpolation=cv2.INTER_NEAREST)
            line(canvas, 'Actual saved BODY 96x96', 20, 365, size=.43)
            line(canvas, 'Actual saved WRIST 96x96', 275, 365, size=.43)
            line(canvas, item['instruction'][:70], 20, 400, size=.43)
            line(canvas, item['sequence'], 20, 425, size=.40)
            plot(canvas, (595, 125), 'XY', (0, 1), item['paths'], item['origin'], extent, tick)
            plot(canvas, (930, 125), 'XZ', (0, 2), item['paths'], item['origin'], extent, tick)
            for index, name in enumerate(item['paths']):
                y = 445 + index * 33
                cv2.line(canvas, (600, y - 5), (630, y - 5), COLORS[name], 3)
                detail = item['details'].get(name)
                label = name if detail is None else f'{name}: XYZ {detail[0]:.4f}m, SO3 {detail[1]:.3f}rad, gate {detail[2]}'
                line(canvas, label, 640, y, size=.42)
            line(canvas, 'Original row labels; shared physical axes. Synthetic quality is not deployment success.', 18, 690, size=.44)
            writer.write(canvas)
    writer.release()
    capture = cv2.VideoCapture(str(video))
    assert capture.isOpened()
    frames = 0
    while capture.read()[0]:
        frames += 1
    capture.release()
    assert frames == 300
    receipt = dict(schema='PI0260_R3_SAVED_ARRAY_FIXED_VIDEO_V1', status='PASS_FIXED_VIDEO_DECODE',
                   video=video.name, video_sha256=sha(video), frames=frames, fps=20, duration_s=15,
                   axes_extent_m=extent, selection_sha256=sha(R3 / 'S4_FIXED_DISPLAY_SELECTION.json'),
                   chapters=[item['source_case'] for item in items], model_queries=0, physics_steps=0)
    (output / 'video_receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    render()
