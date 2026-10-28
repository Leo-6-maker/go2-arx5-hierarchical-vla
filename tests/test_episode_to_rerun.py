"""Actual pinned Rerun SDK serialization/reader tests; no mock Rerun objects.

Requires the separate PC viewer environment. Skips explicitly when the optional
viewer SDK is absent; this is not a Core/Vanda runtime dependency.
"""
import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import pytest

rr = pytest.importorskip('rerun', reason='PC-only rerun-sdk is not installed in this environment')
from rerun.chunk import RrdReader

ROOT = Path(__file__).resolve().parents[1]
# Reuse synthetic source-layout fixtures, not a second mock of Rerun itself.
spec = importlib.util.spec_from_file_location('replay_v1_source_fixture', ROOT/'tests/test_episode_replay_viewer.py')
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
from replay_v1_test_package.episode_to_rerun import export_episode, SDK_VERSION
from replay_v1_test_package.records import Episode, sha, add_offline_predictions


def component_rows(chunks, entity, component):
    rows = []
    for chunk in chunks:
        if chunk.entity_path.lstrip('/') != entity.lstrip('/'):
            continue
        batch = chunk.to_record_batch()
        if component not in batch.schema.names:
            continue
        values = batch.column(batch.schema.get_field_index(component)).to_pylist()
        def index(name):
            return batch.column(batch.schema.get_field_index(name)).cast('int64').to_pylist() if name in batch.schema.names else [None]*batch.num_rows
        for t, source, value in zip(index('recorded_time'), index('source_frame'), values):
            rows.append({'time_ns': t, 'source_frame': source, 'value': value})
    return rows


def test_real_sdk_file_and_blueprint_reader_roundtrip(tmp_path):
    source=tmp_path/'source';source.mkdir();episode=Episode(fixtures.make(source))
    before={str(p):sha(p) for p in source.rglob('*') if p.is_file()}
    out=tmp_path/'export';receipt=export_episode(episode,out)
    rrd=out/(episode.id+'.rrd');rbl=out/(episode.id+'.rbl')
    reader=RrdReader(rrd)
    assert len(reader.recordings())==1 and reader.blueprints()
    chunks=list(reader.stream())
    paths={c.entity_path.lstrip('/') for c in chunks}
    assert {'episode/rgb/body_rgb','episode/rgb/wrist_rgb','episode/world/actual_tcp','episode/original_audit','episode/current_frame'} <= paths
    for cam in ['body_rgb','wrist_rgb']:
        image_chunks=[c for c in chunks if c.entity_path.lstrip('/')=='episode/rgb/'+cam]
        assert sum(c.num_rows for c in image_chunks)==3
        assert all({'recorded_time','source_frame'} <= set(c.timeline_names) for c in image_chunks)
    for camera in ('body_rgb', 'wrist_rgb'):
        observed=component_rows(chunks,'episode/rgb/'+camera,'EncodedImage:blob')
        assert sorted((r['source_frame'],r['time_ns']) for r in observed)==[(0,0),(1,50000000),(2,100000000)]
        for r in observed:
            assert hashlib.sha256(bytes(r['value'][0])).hexdigest()==episode.frames[r['source_frame']]['cameras'][camera]['sha256']
    frames=component_rows(chunks,'episode/current_frame','TextDocument:text')
    assert sorted((r['source_frame'],json.loads(r['value'][0])) for r in frames)==list(enumerate(episode.frames))
    assert RrdReader(rbl).blueprints()
    assert out.stat().st_mode & 0o777 == 0o700
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in out.iterdir())
    assert receipt['sdk_version']==SDK_VERSION and receipt['RGB_records']==6 and receipt['new_integrations']==0
    assert {str(p):sha(p) for p in source.rglob('*') if p.is_file()}==before
    with pytest.raises(FileExistsError):export_episode(episode,out)


def test_real_sdk_keeps_prediction_separate_and_input_immutable(tmp_path):
    source=tmp_path/'source';source.mkdir();episode=Episode(fixtures.make(source))
    actual_before=json.loads(json.dumps(episode.frames[0]['actual_tcp']))
    overlay=source/'predictions.json';overlay.write_text(json.dumps([{'frame_index':0,'observation_time_s':0,
        'executed':False,'label':'SYNTHETIC_MODEL','xyz':[[1.,2.,3.]],'model_identity':'SYNTHETIC_ONLY','coordinate_frame':'world'}]))
    add_offline_predictions(episode,{'path':overlay.name,'sha256':sha(overlay)})
    out=tmp_path/'export';export_episode(episode,out)
    chunks=list(RrdReader(out/(episode.id+'.rrd')).stream())
    assert any(c.entity_path.lstrip('/').startswith('episode/world/offline_prediction_NOT_EXECUTED/0') for c in chunks)
    frames=component_rows(chunks,'episode/current_frame','TextDocument:text')
    zero=next(json.loads(r['value'][0]) for r in frames if r['source_frame']==0)
    assert zero['offline_predictions'][0]['executed'] is False and zero['actual_tcp']==actual_before
    assert episode.frames[0]['actual_tcp']==actual_before
    assert episode.frames[0]['offline_predictions'][0]['executed'] is False
