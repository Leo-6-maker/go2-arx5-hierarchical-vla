"""Bind each repaired TRAIN capture to the producer and condition actually registered on Core."""

ORIGINAL = '89f058d5cd3a69cae8f68eb6a63f63a040f20151d07daf0d88e6a0ec5c3f8ffe'
ORIGINAL_SOURCE = '7090e2ad238e76cd6278cf65405f7785604abfdb'
RED_PLACE_SPEC = '73a7e59b8c0061f42013f45730f35d573518f659ed1e4bc29ff202296f0ab947'
SERIES_SOURCE = 'b353b99d7149becf29b85d42820e73cc7985a78e'
SERIES = {
    '2baf989a1af041dfb9287d8b53002ef4d83a99a57c6863d566e353c922e58d15':
        (tuple(f'CHV1-TRAIN-F2-PLACE-blue-right-{i:02d}' for i in range(3,20)), SERIES_SOURCE),
    '8633df7283e1081aa96d5264a9eff2fdd0d40db1c09e849ac5fe917f3dc7c622':
        (tuple(f'CHV1-TRAIN-F2-HOLD-red-{i:02d}' for i in range(3,20)), SERIES_SOURCE),
}
REPAIRS = {
    'b29a56c7c083208408fddbcf34023d8af9374ef1b4cda20e0f90dc57df3c0aa3':
        (('CHV1-TRAIN-F2-PLACE-blue-right-02',), 'e4e95b4f8f16f0a45091de9bac789e5e6ce68540'),
    'deaa9a88dc1426a8a74a46a026a7c9f7ee190bca3256ba301ff98c0b1f9920dc':
        (('CHV1-TRAIN-F2-HOLD-red-02',), '26d10c56362c7fa6879f10a394ff5b9fcec505e3'),
    RED_PLACE_SPEC:
        (('CHV1-TRAIN-F2-PLACE-red-left-02','CHV1-TRAIN-F2-PLACE-red-right-02'),
         'd3933e5d93baf9ef2617bf7d14f3d8c6b2124db5'),
    **SERIES,
}
SCHEMA = 'SLA_CORE_HOPPER_PER_EPISODE_PRODUCERS_V1'


def check_case(condition, spec, source):
    if spec == ORIGINAL:
        if source != ORIGINAL_SOURCE or any(condition in cells for cells,_ in REPAIRS.values()):
            raise ValueError('Capture belongs to a different registered producer')
        return
    cells, revision = REPAIRS.get(spec,((),None))
    if condition not in cells or source != revision:
        raise ValueError('Unregistered repaired capture identity')


def check_admission_provenance(proof):
    if 'schema' not in proof:
        if proof.get('producer_spec') != ORIGINAL:
            raise ValueError('Unexpected original TRAIN authority')
        for case in proof['cases']:
            check_case(case['id'], ORIGINAL, ORIGINAL_SOURCE)
        return
    if (proof['schema'] != SCHEMA or proof.get('original_campaign_spec') != ORIGINAL
            or 'producer_spec' in proof):
        raise ValueError('Unknown or ambiguous producer contract')
    for case in proof['cases']:
        check_case(case['id'], case['producer_spec'], case['producer_source'])
    if not any(c['producer_spec'] in REPAIRS for c in proof['cases']):
        raise ValueError('An aggregate requires at least one admitted repair')
