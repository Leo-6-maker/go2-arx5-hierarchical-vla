"""Bind each repaired TRAIN capture to the producer and condition actually registered on Core."""

ORIGINAL = '89f058d5cd3a69cae8f68eb6a63f63a040f20151d07daf0d88e6a0ec5c3f8ffe'
ORIGINAL_SOURCE = '7090e2ad238e76cd6278cf65405f7785604abfdb'
REPAIRS = {
    'b29a56c7c083208408fddbcf34023d8af9374ef1b4cda20e0f90dc57df3c0aa3':
        ('CHV1-TRAIN-F2-PLACE-blue-right-02', 'e4e95b4f8f16f0a45091de9bac789e5e6ce68540'),
    'deaa9a88dc1426a8a74a46a026a7c9f7ee190bca3256ba301ff98c0b1f9920dc':
        ('CHV1-TRAIN-F2-HOLD-red-02', '26d10c56362c7fa6879f10a394ff5b9fcec505e3'),
}
SCHEMA = 'SLA_CORE_HOPPER_PER_EPISODE_PRODUCERS_V1'


def check_case(condition, spec, source):
    if spec == ORIGINAL:
        if source != ORIGINAL_SOURCE or any(condition == row[0] for row in REPAIRS.values()):
            raise ValueError('Capture belongs to a different registered producer')
    elif REPAIRS.get(spec) != (condition, source):
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
