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
REPAIR02_VERSION='ROBOCASA_ARX5_PORT02_GEOMETRY_REPAIR'
REPAIR02_CARD_SHA='4ac838e931633fedb3a89a2c6a7c58df4ba044176d1eb2808426944484323df8'
PORT03_VERSION='ROBOCASA_ARX5_PORT03_FIXED_CAMERA_HOME'
PORT03_CARD_SHA='ccaa4188e6d497d172b66f04fc5ad8e15cc74fba1c03133cfcc4beb1912e23ee'
PORT04_VERSION='ROBOCASA_ARX5_PORT04_MEASURED_HINGE_ANCHOR'
PORT04_CARD_SHA='5b00e311f33acdcb804267cb0ee234497b16d851a3b02d866c1e75729dd94090'
PORT05_VERSION='ROBOCASA_ARX5_PORT05_SLOWER_TURN_REFERENCE'
PORT05_CARD_SHA='07e49210e58ad42cfb794a7a678ba8f6f4c2f4c6f9d649f4ddf01411665417bd'
PORT06_VERSION='ROBOCASA_ARX5_PORT06_LAYOUT40_RETRACT8'
PORT06_CARD_SHA='8d51b23e78f925fb66de2611f0d32fe0e5d5701f7e129dd7860a8d69215b1908'
PORT07_VERSION='ROBOCASA_ARX5_PORT07_EDGE_HINGE_LAYOUT60'
PORT07_CARD_SHA='ebdd82b5b12bf8f6949b03271c7f1235284229e6a491cbbdbb243cbe013640f9'
PORT08_VERSION='ROBOCASA_ARX5_PORT08_EDGE_HINGE_BRAKED_TURN'
PORT08_CARD_SHA='821420e0840aac479599d0dd85729f14b0802f701f7ad35f07768532fde2550e'
BOOK_ROUTE_VERSION='LIBERO_ARX5_PORT03_TOP_EDGE_FRONT_CROSS_LOWER'
BOOK_ROUTE_CARD_SHA='080f49c2cb45cea7c13237cf62254b0c65a3da44f2f1af91abd34bf5cd2399fb'
BOOK_MARGIN_VERSION='LIBERO_ARX5_PORT04_FRONT_TRACKING_MARGIN'
BOOK_MARGIN_CARD_SHA='d2ad227cd041b456fbe052aea2a716c4a10b1297e0a497fae6d2f30bdd93c09b'
BOOK_DESCENT_VERSION='LIBERO_ARX5_PORT05_SLOW_SUPPORTED_DESCENT'
BOOK_DESCENT_CARD_SHA='7a1b93f86b36fca9215477bd9f3c1032c9f90e3d0b876b1c0a8bab1a7fb6bbf4'
BOOK_RELEASE_VERSION='LIBERO_ARX5_PORT06_SUPPORTED_IN_PLACE_RELEASE'
BOOK_RELEASE_CARD_SHA='dccf632eef8342c5fe5fd23a6b57448736034e7b34ffd7fca87f323caed76b67'
BOOK_SEED_CARD_SHA={
    1:'b1fa2e4140573cb8273446ef1b240bf83c8916f022faf86341ac447ba2906db8',
    2:'be10e8263b696a8e8207b2fd9630da54dc3e3ee359b7f04ef6f9048c4a4948cc',
    3:'ff29fd3872406b645546f7a94f580e1ecdf25a0b0ef592a68487d4d6676ef187',
    4:'e2693ca4b90bda7312be495ec3fdd6d3464667e32a7de60f5522737bf86d0211'}
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


def plan(kind, *, side=None):
    if kind not in CARD_SHA:
        raise ValueError('EXPLICIT_BOOK_OR_ALL_FOUR_SOURCE_STAGE')
    if side is not None:
        if(kind,side)not in(('BOOK','left'),('ROBOCASA_CABINET','right')):
            raise PermissionError('ONE_STATIC_QUALIFIED_SOURCE_CONDITION')
        return [dict(task_kind=kind,side=side)]
    tasks = [('BOOK', 'left'), ('BOOK', 'right')]
    if kind == 'ALL_PINNED':
        tasks += [('ROBOCASA_CABINET', 'right'), ('ROBOCASA_CABINET', 'left')]
    elif kind == 'ROBOCASA_CABINET':
        tasks = [('ROBOCASA_CABINET', 'right'), ('ROBOCASA_CABINET', 'left')]
    return [dict(task_kind=k, side=s) for k, s in tasks]


def checked_manifest(manifest):
    kind = manifest.get('task_kind', 'BOOK')
    side=manifest.get('condition_side')
    tasks = plan(kind,side=side)
    if manifest.get('cabinet_port_revision')in(PORT06_VERSION,PORT07_VERSION,PORT08_VERSION)and(kind,side)!=('ROBOCASA_CABINET','right'):
        raise PermissionError('ONLY_STATIC_CHECKED_RIGHT_LAYOUT40_MANIFEST')
    if side is not None:
        if kind=='BOOK'and manifest.get('book_route_revision')not in(BOOK_ROUTE_VERSION,BOOK_MARGIN_VERSION,BOOK_DESCENT_VERSION,BOOK_RELEASE_VERSION):
            raise PermissionError('EXPLICIT_STATIC_BOOK_ROUTE_MANIFEST')
        if kind=='ROBOCASA_CABINET'and manifest.get('cabinet_port_revision')not in(PORT06_VERSION,PORT07_VERSION,PORT08_VERSION):
            raise PermissionError('EXPLICIT_STATIC_RIGHT_CABINET_LAYOUT_MANIFEST')
    seed=manifest.get('book_seed_index')
    if seed is not None and (type(seed)is not int or seed not in BOOK_SEED_CARD_SHA
            or manifest.get('book_route_revision')!=BOOK_RELEASE_VERSION or kind!='BOOK' or side!='left'):
        raise PermissionError('DECLARED_SOURCE_BOOK_SEED_MANIFEST')
    declared = [x['side']for x in tasks] if kind == 'BOOK' else tasks
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
    side=payload.get('condition_side')
    tasks = plan(kind,side=side)
    card_sha = hashlib.sha256(raw(payload['cards'])).hexdigest()
    qualification = payload['qualification']
    revision=payload.get('cabinet_port_revision')
    revisions={REPAIR02_VERSION:REPAIR02_CARD_SHA,PORT03_VERSION:PORT03_CARD_SHA,PORT04_VERSION:PORT04_CARD_SHA,PORT05_VERSION:PORT05_CARD_SHA,PORT06_VERSION:PORT06_CARD_SHA,PORT07_VERSION:PORT07_CARD_SHA,PORT08_VERSION:PORT08_CARD_SHA}
    if revision is not None and (revision not in revisions or kind!='ROBOCASA_CABINET'
            or qualification.get('cabinet_port_revision')!=revision):
        raise PermissionError('EXPLICIT_SEPARATE_CABINET_REPAIR02_STAGE')
    if revision in(PORT06_VERSION,PORT07_VERSION,PORT08_VERSION)and(side!='right'or qualification.get('condition_side')!='right'):
        raise PermissionError('ONLY_STATIC_CHECKED_RIGHT_LAYOUT40_CONDITION')
    book_revision=payload.get('book_route_revision')
    book_revisions={BOOK_ROUTE_VERSION:BOOK_ROUTE_CARD_SHA,BOOK_MARGIN_VERSION:BOOK_MARGIN_CARD_SHA,
                    BOOK_DESCENT_VERSION:BOOK_DESCENT_CARD_SHA,BOOK_RELEASE_VERSION:BOOK_RELEASE_CARD_SHA}
    if book_revision is not None:
        if (book_revision not in book_revisions or kind!='BOOK' or side!='left'
                or revision is not None or qualification.get('book_route_revision')!=book_revision
                or qualification.get('condition_side')!=side):
            raise PermissionError('EXACT_STATIC_BOOK_ROUTE_QUALIFICATION')
    elif side is not None and not(kind=='ROBOCASA_CABINET'and revision in(PORT06_VERSION,PORT07_VERSION,PORT08_VERSION)):
        raise PermissionError('SINGLE_BOOK_CONDITION_NEEDS_STATIC_ROUTE')
    expected_sha=book_revisions[book_revision]if book_revision else revisions[revision]if revision else CARD_SHA[kind]
    seed=payload.get('book_seed_index')
    if seed is not None:
        if (type(seed)is not int or seed not in BOOK_SEED_CARD_SHA
                or book_revision!=BOOK_RELEASE_VERSION or qualification.get('book_seed_index')!=seed):
            raise PermissionError('EXACT_DECLARED_SOURCE_BOOK_SEED')
        expected_sha=BOOK_SEED_CARD_SHA[seed]
    if (card_sha != expected_sha
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
