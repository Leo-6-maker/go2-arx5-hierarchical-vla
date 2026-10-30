"""Complete pinned asset/task closure before a cold source root is created.

Pure bytes and manifest validation. No filesystem, network, simulator, clock,
authority or resource mutation. Historical BOOK-only payloads retain their plan.
"""
import base64
import hashlib
import json
from pathlib import PurePosixPath

CARD_SHA = {
    'BOOK': '5d7226641adccd4a6a1eb935f3c29824893a0a22d16ee6447b90554f0a824c8b',
    'ALL_PINNED': '6ac4ec4e979b28dc2a2c24a697caed54d8abde4a47df62fa1d00f8de96f60057',
    'ROBOCASA_CABINET': '6ac4ec4e979b28dc2a2c24a697caed54d8abde4a47df62fa1d00f8de96f60057',
}
BOOK_XML = {
    'black_book/black_book.xml': '73481d4f87d7480cc75add0fe7cce147e5ea2921',
    'desk_caddy/desk_caddy.xml': 'bb59530d8629878f8974efaed4abf79d31743ee3',
}
CABINET_XML = {
    'cabinet_single': '4359060211bb42e341c23479249eb3fe25fb4918',
    'slab': '696e08782235be3d726985b3f77a58a834c5951f',
    'bar_handle': '6120c2eb0187e48f44ae2560aa5c2829888e187c',
}
BOOK_BINARY = {
    'black_book/visual/black_book_vis.msh': (773296, 'ca4a7f95bad5dee37a99c0374df132eadf5f8304770e5830f3286d730c844049'),
    'black_book/black_book_texture.png': (186624, 'b743b0e878a10e579371707f3d135c6e7a80a8c307fffabfe1fea6a784830ec2'),
    'desk_caddy/visual/desk_caddy_vis.msh': (2696776, 'c2a31df6411aa382d91553a594faa2cb9577908c3881b9022616f9d6604333a4'),
    'desk_caddy/desk_caddy_texture.png': (11230178, 'ea07fb85ab5ef7167040e459ae4442827d114d778aaf4f6a8229bc3898406238'),
}
CABINET_TEXTURE = {
    'wood': (1460776, '18d87069529422d832559ce969858576dd46f83efc3dfd9e5ffbf26d740be123'),
    'metal': (1998226, 'e67c23a272f98890b485c5e2ae2ef1088080067954c676cb70170252ab931bf4'),
}


def raw(value):
    return (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()


def plan(kind):
    if kind not in CARD_SHA:
        raise ValueError('EXPLICIT_BOOK_OR_ALL_FOUR_SOURCE_STAGE')
    tasks = [('BOOK', 'left'), ('BOOK', 'right')]
    if kind == 'ALL_PINNED':
        tasks += [('ROBOCASA_CABINET', 'right'), ('ROBOCASA_CABINET', 'left')]
    elif kind == 'ROBOCASA_CABINET':
        tasks = [('ROBOCASA_CABINET', 'right'), ('ROBOCASA_CABINET', 'left')]
    return [dict(task_kind=k, side=s) for k, s in tasks]


def checked_manifest(manifest):
    kind = manifest.get('task_kind', 'BOOK')
    tasks = plan(kind)
    declared = ['left', 'right'] if kind == 'BOOK' else tasks
    if (manifest['predeclared_tasks'] != declared
            or manifest['attempts_this_chain_max'] != len(tasks)):
        raise PermissionError('EXACT_COMPLETE_SOURCE_STAGE_PLAN')
    return kind, tasks


def _decode(encoded, limit):
    if not isinstance(encoded, str) or len(encoded) > 4*((limit+2)//3):
        raise ValueError('BOUNDED_SOURCE_ASSET_ENCODING')
    body = base64.b64decode(encoded, validate=True)
    if not 0 < len(body) <= limit:
        raise ValueError('BOUNDED_SOURCE_ASSET_BYTES')
    return body


def _definitions(values, expected):
    if set(values) != set(expected):
        raise ValueError('COMPLETE_PINNED_SOURCE_DEFINITIONS')
    bodies = {}
    for name, pin in expected.items():
        body = _decode(values[name], 32768)
        blob = hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()
        if blob != pin:
            raise ValueError('EXACT_PINNED_SOURCE_DEFINITION_BLOB')
        bodies[name] = body
    return bodies


def _binary(values, expected):
    if set(values) != set(expected):
        raise ValueError('COMPLETE_PINNED_SOURCE_BINARY_CLOSURE')
    bodies = {}
    for name, (count, pin) in expected.items():
        row = values[name]
        body = _decode(row['base64'], count)
        if (len(body) != count or hashlib.sha256(body).hexdigest() != pin
                or row['bytes'] != count or row['sha256'] != pin):
            raise ValueError('EXACT_PINNED_SOURCE_BINARY_BYTES')
        bodies[name] = body
    return bodies


def unpack(payload):
    """Validate actual transferred bytes, including both cabinet orientations."""
    kind = payload.get('task_kind', 'BOOK')
    tasks = plan(kind)
    card_sha = hashlib.sha256(raw(payload['cards'])).hexdigest()
    qualification = payload['qualification']
    if (card_sha != CARD_SHA[kind]
            or qualification['fixed_task_cards_sha256'] != card_sha
            or qualification['source_sha'] != payload['source_sha']):
        raise PermissionError('FIXED_COMPLETE_SOURCE_CARDS_AND_QUALIFICATION')
    definitions = _definitions(payload['definition_files'], BOOK_XML)
    binaries = _binary(payload['libero_binary_files'], BOOK_BINARY)
    cabinet_definitions, cabinet_textures = {}, {}
    if kind in ('ALL_PINNED','ROBOCASA_CABINET'):
        ds = _definitions(payload['robocasa_definition_files'], CABINET_XML)
        ts = _binary(payload['robocasa_texture_files'], CABINET_TEXTURE)
        for name, body in ds.items():
            path = 'robocasa/'+name+'.xml'
            definitions[path] = body
            cabinet_definitions[name] = 'definition/'+path
        for name, body in ts.items():
            path = 'robocasa/'+name+'.png'
            binaries[path] = body
            cabinet_textures[name] = path
        if qualification.get('independent_raw_auditors') != [
                'scripts/pi0294/multitask/night_book_audit.py',
                'scripts/pi0294/multitask/night_cabinet_audit.py']:
            raise PermissionError('BOTH_SEPARATE_SOURCE_RAW_AUDITORS')
    elif payload.get('robocasa_definition_files') or payload.get('robocasa_texture_files'):
        raise PermissionError('CABINET_ASSETS_REQUIRE_EXPLICIT_FOUR_TASK_PLAN')
    for name in [*definitions, *binaries]:
        path = PurePosixPath(name)
        if path.is_absolute() or str(path) != name or '..' in path.parts:
            raise ValueError('SOURCE_BUNDLE_RELATIVE_MEMBER')
    return dict(task_kind=kind, plan=tasks, definitions=definitions, binaries=binaries,
                robocasa_definition_members=cabinet_definitions,
                robocasa_texture_members=cabinet_textures,
                total_binary_bytes=sum(map(len, binaries.values())))
