"""Inspect pinned LIBERO task text without importing or running its simulator.

This is a deliberately small BDDL reader for the selected single-book tasks.
It retains the original predicate and distributions; it cannot create an
environment, translate an action, admit a demonstration or grant execution.
"""
import hashlib
import re


def _fail(why):
    raise ValueError('OPEN_SOURCE_TASK_' + why)


def _tree(text):
    # BDDL task names and numbers need no string evaluation or executable code.
    text = re.sub(r';[^\n]*', '', text)
    tokens = re.findall(r'\(|\)|[^\s()]+', text)
    if not tokens or len(tokens) > 16384:
        _fail('BOUNDED_TOKEN_COUNT')
    stack, root = [], None
    for token in tokens:
        if token == '(':
            if len(stack) >= 32:
                _fail('BOUNDED_NESTING')
            node = []
            if stack:
                stack[-1].append(node)
            elif root is not None:
                _fail('ONE_COMPLETE_DOCUMENT')
            else:
                root = node
            stack.append(node)
        elif token == ')':
            if not stack:
                _fail('BALANCED_PARENTHESES')
            stack.pop()
        elif not stack:
            _fail('ONE_COMPLETE_DOCUMENT')
        else:
            stack[-1].append(token)
    if stack or not root or root[0] != 'define':
        _fail('ONE_BALANCED_TASK_DEFINITION')
    return root


def inspect_libero_book_task(task_bytes, *, expected_sha256):
    """Return source task semantics only, after exact upstream-byte checking."""
    if (type(task_bytes) is not bytes or not 0 < len(task_bytes) <= 256 * 1024
            or not isinstance(expected_sha256, str)
            or re.fullmatch('[0-9a-f]{64}', expected_sha256) is None
            or hashlib.sha256(task_bytes).hexdigest() != expected_sha256):
        _fail('EXACT_PINNED_BDDL_BYTES')
    root = _tree(task_bytes.decode('utf-8'))
    fields = {}
    for node in root[1:]:
        if not isinstance(node, list) or not node or not isinstance(node[0], str):
            _fail('EXPLICIT_TASK_FIELDS')
        if node[0] in fields:
            _fail('DUPLICATE_TASK_FIELD')
        fields[node[0]] = node[1:]
    required = {'problem', ':domain', ':language', ':regions', ':fixtures', ':objects', ':init', ':goal'}
    if not required <= fields.keys() or fields[':domain'] != ['robosuite']:
        _fail('SUPPORTED_ROBOSUITE_TASK')
    language = fields[':language']
    if not language or not all(isinstance(x, str) for x in language):
        _fail('EXPLICIT_LANGUAGE')

    def declarations(name):
        values = fields[name]
        if not values or not all(isinstance(x, str) for x in values):
            _fail('TYPED_OBJECT_DECLARATIONS')
        pairs, i = {}, 0
        while i < len(values):
            names = []
            while i < len(values) and values[i] != '-':
                names.append(values[i])
                i += 1
            if not names or i+1 >= len(values) or values[i+1] == '-':
                _fail('TYPED_OBJECT_DECLARATIONS')
            kind = values[i+1]
            for key in names:
                if key in pairs:
                    _fail('UNIQUE_TYPED_OBJECT_DECLARATIONS')
                pairs[key] = kind
            i += 2
        if not pairs:
            _fail('UNIQUE_TYPED_OBJECT_DECLARATIONS')
        return pairs

    objects, fixtures = declarations(':objects'), declarations(':fixtures')
    if set(objects) & set(fixtures):
        _fail('DISTINCT_OBJECT_AND_FIXTURE_IDS')
    goals = fields[':goal']
    if (len(goals) != 1 or not isinstance(goals[0], list)
            or len(goals[0]) != 2 or goals[0][0] != 'And'
            or not isinstance(goals[0][1], list) or len(goals[0][1]) != 3):
        _fail('SELECTED_SINGLE_CONTAINMENT_GOAL_ONLY')
    predicate, object_id, target_region = goals[0][1]
    if (predicate != 'In' or not isinstance(object_id, str)
            or objects.get(object_id) not in {'black_book', 'yellow_book'}
            or not isinstance(target_region, str)):
        _fail('DECLARED_SINGLE_BOOK_CONTAINMENT')
    regions = {}
    for region in fields[':regions']:
        if not isinstance(region, list) or not region or not isinstance(region[0], str):
            _fail('NAMED_REGIONS')
        name = region[0]
        targets = [x for x in region[1:] if isinstance(x, list) and x and x[0] == ':target']
        if name in regions or len(targets) != 1 or len(targets[0]) != 2:
            _fail('UNIQUE_REGION_WITH_EXPLICIT_TARGET')
        regions[name] = targets[0][1]
    matching = [(name, target) for name, target in regions.items()
                if isinstance(target, str) and target in fixtures
                and target + '_' + name == target_region]
    if len(matching) != 1:
        _fail('GOAL_REGION_BOUND_TO_DECLARED_FIXTURE')
    return dict(schema='PINNED_LIBERO_BOOK_TASK_STATIC_INSPECTION_V1',
        source_sha256=expected_sha256, language=' '.join(language),
        upstream_goal=dict(predicate=predicate, object=object_id, region=target_region),
        objects=objects, fixtures=fixtures,
        original_regions=fields[':regions'], original_initial_state=fields[':init'],
        task_executed=False, action_conversion_qualified=False,
        demonstration_admitted=False, execution_authority=False)
