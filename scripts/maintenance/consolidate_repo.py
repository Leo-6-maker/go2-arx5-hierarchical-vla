#!/usr/bin/env python3
"""One-shot, reversible GitHub repository administration; no project imports.

Inventory is the default. Apply requires the exact inventory commit and digest.
Historical heads become same-name immutable compatibility tags before retirement.
Git deletion uses explicit expected-SHA leases and an atomic push.
"""
from __future__ import annotations
import argparse, base64, hashlib, json, os, re, subprocess, tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import quote
from urllib.error import HTTPError

REPO = 'Leo-6-maker/go2-arx5-hierarchical-vla'
BRANCH = 'repo-maintenance/consolidation-20261001'
ROOT = 'docs/maintenance/2026-10-01'
MARKER = 'REPO_CONSOLIDATION_20261001'
AUTH = 5929691215
KEEP = {
    'main': 'accepted code baseline; no experimental bulk merge',
    'pi0266/state-conditioned-feedback': 'current execution; do not move or retire',
    'pi0265/new-scene-feedback': 'recent controller source/dependency',
    'pi0264/language-diversity-transfer': 'current donor-producing source',
    'pi0263/train-template-counterbalancing': 'recent completed source retained during transition',
    'pi0263/storage-subagent-model-audit-go': 'separate storage-agent authority',
    'pi0257/corelab-video-evidence-workflow-v1': 'maintained video tooling during transition',
    BRANCH: 'one-shot administrative work and immutable inventory',
}
OLD_PRS = {46, 49, *range(51, 70), *range(71, 79)}
STALE_ISSUES = {48, 50}


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n'


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def historical(name):
    m = re.match(r'^pi(\d{4})(?:[-/])', name)
    return bool((m and int(m[1]) <= 265) or name.startswith('stage') or name in {
        'chore/github-connection-smoke-test-20260913',
        'handoff/gpt-pi-stage45b-20260908', 'repair/pi0096-nested-cosmos-hf-cache',
        'repo-consolidation/stage44r-baseline'})


def plan_branches(branches, prs):
    """Unknown names, protected refs and retained PR dependencies always survive."""
    dependency = set()
    for pr in prs:
        if pr['number'] not in OLD_PRS or pr['head_repository'] != REPO:
            dependency.update([pr['head_ref'], pr['base_ref']])
    out = []
    for b in branches:
        n = b['name']
        why = KEEP.get(n)
        if not why and b['protected']:
            why = 'GitHub-protected branch'
        if not why and n in dependency:
            why = 'unretired/open PR head or base'
        if not why and not historical(n):
            why = 'unknown or newer namespace; outside audited retirement scope'
        out.append({**b, 'decision': 'KEEP' if why else 'ARCHIVE_COMPATIBILITY_TAG',
                    'reason': why or 'historical branch name; retain exact source via verified tag',
                    'archive_tag': None if why else n})
    return out


def delete_candidates(plan, current_heads, current_tags, remaining_prs):
    blocked = set(KEEP)
    for p in remaining_prs:
        blocked.update([p['head_ref'], p['base_ref']])
    return [x for x in plan if x['decision'] == 'ARCHIVE_COMPATIBILITY_TAG'
            and x['name'] not in blocked
            and current_heads.get(x['name']) == x['sha']
            and current_tags.get(x['name']) == x['sha']]


class API:
    def __init__(self, token):
        if not token:
            raise RuntimeError('Missing explicitly supplied GITHUB_TOKEN; no credential discovery')
        self.token = token

    def request(self, path, method='GET', data=None):
        if not path.startswith('/') or path.startswith('//'):
            raise ValueError('repository-relative API path required')
        req = Request('https://api.github.com/repos/' + REPO + path,
            data=None if data is None else json.dumps(data).encode(), method=method,
            headers={'Authorization': 'Bearer ' + self.token,
                'Accept': 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28',
                'Content-Type': 'application/json'})
        try:
            with urlopen(req, timeout=60) as r:
                raw = r.read()
                return json.loads(raw) if raw else None
        except HTTPError as e:
            # Never print response headers, token, authenticated download URLs or request objects.
            raise RuntimeError(f'GitHub HTTP {e.code} for {method} {path.split("?")[0]}') from None

    def pages(self, path):
        ans = []
        for page in range(1, 51):
            sep = '&' if '?' in path else '?'
            batch = self.request(f'{path}{sep}per_page=100&page={page}')
            if not isinstance(batch, list):
                raise ValueError('expected list response')
            ans.extend(batch)
            if len(batch) < 100:
                return ans
        raise RuntimeError('Pagination ceiling reached; incomplete inventory is not admissible')

    def file(self, path, ref):
        d = self.request('/contents/' + quote(path, safe='/') + '?ref=' + quote(ref, safe=''))
        if d.get('encoding') != 'base64':
            raise RuntimeError('Expected small base64 text repository file')
        return base64.b64decode(d['content']).decode('utf-8')

    def publish(self, files, message):
        """Forward-only administrative commit on the sole maintenance branch."""
        ref = self.request('/git/ref/heads/' + BRANCH)
        old = ref['object']['sha']
        tree = self.request('/git/commits/' + old)['tree']['sha']
        for path in files:
            if not (path.startswith(ROOT + '/') or path in {
                    'docs/REPOSITORY_WORKFLOW.md', 'docs/PROJECT_STATUS.md',
                    'docs/RESEARCH_INDEX.md', 'README.md'}):
                raise ValueError('Outside administrative path allowlist: ' + path)
        entries = [{'path': p, 'mode': '100644', 'type': 'blob', 'content': t}
                   for p, t in files.items()]
        tree_new = self.request('/git/trees', 'POST', {'base_tree': tree, 'tree': entries})['sha']
        commit = self.request('/git/commits', 'POST', {
            'message': message, 'tree': tree_new, 'parents': [old]})['sha']
        self.request('/git/refs/heads/' + BRANCH, 'PATCH', {'sha': commit, 'force': False})
        for path, text in files.items():
            if self.file(path, commit) != text:
                raise RuntimeError('Publication readback mismatch: ' + path)
        return commit


def normalized_pr(p):
    return {k: p.get(k) for k in ['number','title','body','state','draft','created_at','updated_at','html_url']} | {
        'head_ref': p['head']['ref'], 'head_sha': p['head']['sha'],
        'head_repository': (p['head'].get('repo') or {}).get('full_name'),
        'base_ref': p['base']['ref'], 'base_sha': p['base']['sha']}


def normalized_issue(i):
    return {k: i.get(k) for k in ['number','title','body','state','state_reason',
                                'comments','created_at','updated_at','html_url']}


def read_state(api):
    branches = [{'name': x['name'], 'sha': x['commit']['sha'], 'protected': x['protected']}
                for x in api.pages('/branches')]
    prs = [normalized_pr(p) for p in api.pages('/pulls?state=open')]
    issues = [normalized_issue(i) for i in api.pages('/issues?state=open') if 'pull_request' not in i]
    return {'observed_at_utc': datetime.now(timezone.utc).isoformat(),
            'branches': branches, 'pull_requests': prs, 'issues': issues}


def table_plan(snap):
    rows = ['# Repository inventory and retirement plan', '',
        f"Observed {snap['observed_at_utc']}; scope: `{REPO}`.", '',
        f"Branches: **{len(snap['branches'])}** (includes this maintenance branch); "
        f"open PRs: **{len(snap['pull_requests'])}**; open issues: **{len(snap['issues'])}**.", '',
        'This is an inventory, not a statement that any deletion or closure has occurred.', '',
        '| Branch | Exact HEAD | Administrative decision |', '|---|---|---|']
    for x in snap['branch_plan']:
        rows.append(f"| `{x['name']}` | `{x['sha']}` | {x['decision']}: {x['reason']} |")
    rows += ['', '## Historical PRs', '', '| PR | Title | HEAD | Decision |', '|---|---|---|---|']
    for p in snap['pull_requests']:
        d = 'archive unmerged; not scientific acceptance' if p['number'] in OLD_PRS else 'keep'
        rows.append(f"| #{p['number']} | {p['title'].replace('|','/')} | `{p['head_sha']}` | {d} |")
    rows += ['', '## Issues', '',
        '- #11 remains the one scientific authorization/return ledger; its outdated landing body gets an indexed summary.',
        '- #47 remains the explicit unmerged source-integration backlog; it is not marked technically completed.',
        '- #48 and #50 are administratively superseded, closed as not-planned with original bodies and evidence intact.',
        '', '## Recovery', '',
        'Each retired historical branch is first preserved as a lightweight tag with its **original branch name** and identical commit SHA. This is a compatibility archive, not a release or quality PASS.',
        'No scientific branch history is merged into main. Unknown, moved, protected or remaining-PR-dependent branches survive.',
        'Restore from an authorized local checkout: `git fetch origin tag <old-branch-name>` then `git switch -c <new-working-branch> refs/tags/<old-branch-name>`.',
        'Recreate the original remote branch only when explicitly resuming that historical line, at its recorded SHA; do not move the archive tag.', '']
    return '\n'.join(rows)


def inventory(api):
    s = read_state(api)
    if not any(b['name'] == 'main' for b in s['branches']):
        raise RuntimeError('main missing')
    if len(s['branches']) > 200:
        raise RuntimeError('Unexpected branch growth; stop before mutation')
    s.update({'schema': MARKER + '_V1', 'repository': REPO, 'authority_comment': AUTH,
              'administrative_only': True})
    s['branch_plan'] = plan_branches(s['branches'], s['pull_requests'])
    txt = json_text(s)
    files = {ROOT + '/BEFORE.json': txt, ROOT + '/INVENTORY.md': table_plan(s)}
    for i in s['issues']:
        files[ROOT + f"/issue-{i['number']}-original.md"] = (i['body'] or '')
    before_sha = api.publish(files, 'docs(repo): freeze complete before-state and reversible retirement inventory')
    summary = {'status': 'INVENTORY_COMPLETE_NO_RETIREMENT', 'snapshot_commit': before_sha,
        'snapshot_sha256': digest(txt), 'branches': len(s['branches']), 'open_prs': len(s['pull_requests']),
        'open_issues': len(s['issues']),
        'archive_candidate_count': sum(x['decision'] != 'KEEP' for x in s['branch_plan']),
        'keep_names': [x['name'] for x in s['branch_plan'] if x['decision'] == 'KEEP']}
    api.publish({ROOT + '/PLAN_RESULT.json': json_text(summary)}, 'docs(repo): seal inventory digest and counts')
    print(json_text(summary))


class GitRemote:
    def __init__(self, token, cwd):
        self.cwd = cwd
        self.env = dict(os.environ)
        auth = base64.b64encode(('x-access-token:' + token).encode()).decode()
        self.env.update({'GIT_TERMINAL_PROMPT': '0', 'GIT_CONFIG_COUNT': '1',
            'GIT_CONFIG_KEY_0': 'http.https://github.com/.extraheader',
            'GIT_CONFIG_VALUE_0': 'AUTHORIZATION: basic ' + auth,
            'GIT_LFS_SKIP_SMUDGE': '1'})
        self.run(['init', '--bare', '.'])
        self.run(['remote', 'add', 'origin', 'https://github.com/' + REPO + '.git'])
        self.run(['fetch', '--no-tags', 'origin', '+refs/heads/*:refs/remotes/origin/*', '+refs/tags/*:refs/tags/*'])

    def run(self, args):
        p = subprocess.run(['git', *args], cwd=self.cwd, env=self.env, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
        if p.returncode:
            # Never include environment or credential-bearing invocation in error messages.
            raise RuntimeError('Git operation failed (' + args[0] + '): ' + p.stderr[-3000:])
        return p.stdout

    def refs(self):
        heads, tags = {}, {}
        for line in self.run(['ls-remote', '--heads', '--tags', 'origin']).splitlines():
            sha, ref = line.split('\t')
            if ref.startswith('refs/heads/'):
                heads[ref[11:]] = sha
            elif ref.startswith('refs/tags/') and not ref.endswith('^{}'):
                tags[ref[10:]] = sha
        return heads, tags

    def archive(self, plan):
        heads, tags = self.refs()
        refs, notes = [], []
        for x in plan:
            if x['decision'] == 'KEEP':
                continue
            n, sha = x['name'], x['sha']
            if heads.get(n) != sha:
                notes.append({'branch': n, 'decision': 'SKIP_HEAD_MOVED_OR_ABSENT'})
            elif n in tags and tags[n] != sha:
                notes.append({'branch': n, 'decision': 'SKIP_EXISTING_TAG_CONFLICT'})
            elif n not in tags:
                if self.run(['cat-file', '-t', sha]).strip() != 'commit':
                    raise RuntimeError('Archive target is not a commit')
                refs.append(f'{sha}:refs/tags/{n}')
        if refs:
            self.run(['push', '--atomic', 'origin', *refs])
        heads, tags = self.refs()
        verified = [x for x in plan if x['decision'] != 'KEEP' and tags.get(x['name']) == x['sha']]
        return verified, notes

    def delete(self, rows):
        if rows:
            self.run(['push', '--atomic',
                *[f"--force-with-lease=refs/heads/{x['name']}:{x['sha']}" for x in rows],
                'origin', *[f":refs/heads/{x['name']}" for x in rows]])


def post_once(api, number, text):
    comments = api.pages(f'/issues/{number}/comments')
    if any(MARKER in c.get('body', '') and text.splitlines()[0] in c.get('body', '') for c in comments):
        return
    api.request(f'/issues/{number}/comments', 'POST', {'body': text})


def landing_body():
    return f'''# Scientific ledger and current repository entry points

**Repository organization updated 2026-10-01. This landing page is navigation, not a new experiment authorization.**

- [Current project status](https://github.com/{REPO}/blob/main/docs/PROJECT_STATUS.md)
- [Repository / branch / PR workflow](https://github.com/{REPO}/blob/main/docs/REPOSITORY_WORKFLOW.md)
- [Research and evidence index](https://github.com/{REPO}/blob/main/docs/RESEARCH_INDEX.md)
- [Pre-consolidation issue body](https://github.com/{REPO}/blob/{BRANCH}/{ROOT}/issue-11-original.md)
- Source integration backlog: #47. Historical PR #77 remains a closed, recoverable provenance record; it is not the current implementation branch.

## Current sources and claim boundary

Latest scientific GO read during this maintenance: PI0266, comment 5929179412, immutable protocol release `5f728de34c900523fa933a75c82be1b310ad814b`. Current execution branch: `pi0266/state-conditioned-feedback`; observed source `59356c33637cea5d19fa9e1a9223eeeb940792d1`. A source commit or CPU preparation is not proof of model/physical completion. Read newer experiment returns below before any execution.

PI0264 completed with quality FAIL; PI0265 is PARTIAL / controller admission blocked; selected/deployment remain NONE until a later explicit scientific decision. Repository administration does not change these results.

The long-term objective is a reusable vision-language task-space EEF policy with explicit, replaceable embodiment-specific execution. Same-controller adaptation and cross-embodiment generalization are distinct claims. This objective is not itself an authorization to change the frozen PI0266 recipe.

## Operating rules

1. A direct current user instruction and the latest applicable explicit scientific GO define scope. Administrative updates do not reopen exhausted experimental budgets.
2. Use one main execution owner; optional helpers, including storage maintenance, are not mandatory approval gates.
3. Protocol, execution code SHA, model/processor/statistics/decode identity, counts, metrics, and evidence remain distinct. Preserve historical FAIL/PARTIAL/NONE and raw evidence.
4. Mechanical fixes within the frozen scientific definition may proceed under its existing authorization. New science, protected data, hardware, budgets, architecture or post-outcome threshold changes need explicit authorization.
5. Publish one concise stage return here, linking immutable source/evidence. Do not post every new experiment to unrelated historical PRs. New PRs represent a bounded code change, not another general-purpose ledger.
6. Historical branches can be archived as same-name tags with exact SHA preservation. An archived/closed PR is neither merged code nor a scientific PASS. No rebase/force rewrite of scientific history.
7. Keep model weights, datasets, raw large videos and private WBC/vendor assets outside Git. Private assets must not be published to public HF.

All historical comments remain unchanged and searchable. Old Stage25 instructions in the saved original body are historical, not the current status panel.
'''


def apply(api, snapshot_commit, snapshot_sha256):
    raw = api.file(ROOT + '/BEFORE.json', snapshot_commit)
    if digest(raw) != snapshot_sha256:
        raise RuntimeError('Approved before-state digest mismatch')
    s = json.loads(raw)
    if s['schema'] != MARKER + '_V1' or s['repository'] != REPO:
        raise RuntimeError('Wrong approved snapshot')
    plan = s['branch_plan']
    if any(x['name'] in KEEP and x['decision'] != 'KEEP' for x in plan):
        raise RuntimeError('Attempted protected-branch retirement')
    audit = {'schema': MARKER + '_V1', 'snapshot_commit': snapshot_commit,
             'snapshot_sha256': snapshot_sha256, 'status': 'IN_PROGRESS',
             'archived_prs': [], 'closed_issues': [], 'skipped': [], 'deleted_branches': [],
             'scientific_merges': 0, 'model_or_simulator_calls': 0,
             'main_runtime_sources_changed': False}
    with tempfile.TemporaryDirectory(prefix='repo-consolidation-') as tmp:
        git = GitRemote(api.token, tmp)
        verified, notes = git.archive(plan)
        audit['skipped'].extend(notes)
        audit['verified_archive_tags'] = [{'tag': x['name'], 'sha': x['sha']} for x in verified]
        archive_commit = api.publish({ROOT + '/ARCHIVE_PREPARED.json': json_text(audit)},
            'docs(repo): verify immutable source tags before closing or deleting refs')
        archive_link = f'https://github.com/{REPO}/blob/{archive_commit}/{ROOT}/ARCHIVE_PREPARED.json'
        validtags = {x['name']: x['sha'] for x in verified}
        for p in s['pull_requests']:
            n = p['number']
            if n not in OLD_PRS or p['head_repository'] != REPO:
                continue
            cur = normalized_pr(api.request(f'/pulls/{n}'))
            if cur['state'] != 'open' or cur['head_sha'] != p['head_sha']:
                audit['skipped'].append({'pr': n, 'decision': 'STATE_OR_HEAD_CHANGED'})
                continue
            if validtags.get(p['head_ref']) != p['head_sha']:
                audit['skipped'].append({'pr': n, 'decision': 'HEAD_NOT_ARCHIVED'})
                continue
            text = f'''## Administrative archive — {MARKER}

Closed **unmerged**, not scientifically accepted, under the user-requested repository consolidation / Issue #11 comment {AUTH}.

Original title, description, review discussion and experimental results remain unchanged. This is a historical implementation/provenance surface, not an active job queue. Its unmerged contents are not silently imported into main.

- Exact source: `{p['head_ref']}@{p['head_sha']}`.
- Same-name compatibility archive tag: `{p['head_ref']}` at the identical commit.
- [Verified source archive]({archive_link}).
- Continuing navigation and scientific returns: #11. Unintegrated reusable code work is tracked in #47.

Restore a new working branch from the recorded tag/SHA if a later authorized task needs this code. No model, simulator, historical result, experiment budget or HF artifact was changed by closing this PR.
'''
            post_once(api, n, text)
            api.request(f'/pulls/{n}', 'PATCH', {'state': 'closed'})
            if api.request(f'/pulls/{n}')['state'] != 'closed':
                raise RuntimeError('PR closure readback failed')
            audit['archived_prs'].append(n)
        for n in sorted(STALE_ISSUES):
            cur = api.request(f'/issues/{n}')
            if cur['state'] == 'open':
                post_once(api, n, f'''## Administrative supersession — {MARKER}

This historical stage-specific work queue is retired into #11 (immutable scientific ledger) and #47 (unmerged reusable-code integration backlog). Original instructions, comments and results remain intact; [the pre-change body](https://github.com/{REPO}/blob/{snapshot_commit}/{ROOT}/issue-{n}-original.md) is archived. Closed as **not planned / superseded**, not as a new scientific PASS or a claim that every engineering item was implemented. No historical evidence or source is deleted.''')
                api.request(f'/issues/{n}', 'PATCH', {'state': 'closed', 'state_reason': 'not_planned'})
                audit['closed_issues'].append(n)
        old47 = next((i for i in s['issues'] if i['number'] == 47), None)
        if old47:
            api.request('/issues/47', 'PATCH', {'title': 'Reusable-code integration backlog — current EEF research to maintained main',
                'body': f'''# One code-integration backlog, not another scientific gate

The original Stage45C infrastructure request is preserved [verbatim](https://github.com/{REPO}/blob/{snapshot_commit}/{ROOT}/issue-47-original.md). PR #49 and the other historical PRs are archived unmerged; their implementation is recoverable, not automatically accepted.

Current research source: `pi0266/state-conditioned-feedback`. Maintained `main` retains its accepted code baseline and gets current navigation. Integrating runtime changes is distinct from this administrative cleanup.

Remaining work, for a bounded future engineering change:
- [ ] Inventory reusable training/evaluation/video/controller helpers against the current source; distinguish implementations from plan-only stubs.
- [ ] Extract only a coherent, tested dependency set; do not octopus-merge incompatible experiment branches.
- [ ] Prove API/shape/coordinate/serialization compatibility with CPU tests without rerunning historical experiments.
- [ ] Land narrow PRs into main, retaining old source/report paths and immutable execution identities.

No new GPU job, model query, simulator/WBC, data access or scientific recipe is authorized by this backlog. Scientific scope and returns remain in #11. Repository consolidation status: [audit](https://github.com/{REPO}/blob/{BRANCH}/{ROOT}/RESULT.md).
'''})
        api.request('/issues/11', 'PATCH', {'title': 'Research control plane — current EEF mainline, scientific gates and evidence index',
                                          'body': landing_body()})
        # Refresh dependencies immediately before a single lease-checked atomic deletion.
        now = read_state(api)
        heads, tags = git.refs()
        currently_protected = {x['name'] for x in now['branches'] if x['protected']}
        rows = [x for x in delete_candidates(plan, heads, tags, now['pull_requests'])
                if x['name'] not in currently_protected]
        if rows:
            try:
                git.delete(rows)
            except RuntimeError as exc:
                audit['skipped'].append({'operation': 'atomic_branch_delete', 'reason': str(exc)})
        heads, tags = git.refs()
        for x in plan:
            n, sha = x['name'], x['sha']
            if n not in heads:
                if tags.get(n) != sha:
                    raise RuntimeError('Lost source retention invariant for ' + n)
                audit['deleted_branches'].append(n)
            elif heads[n] != sha and n != BRANCH:
                audit['skipped'].append({'branch': n, 'decision': 'CONCURRENT_HEAD_PRESERVED', 'new_sha': heads[n]})
        audit['source_heads_retained'] = all(x['name'] in heads or tags.get(x['name']) == x['sha'] for x in plan)
        audit['remaining_branch_heads'] = heads
        audit['archive_tag_readback_verified'] = all(tags.get(x['name']) == x['sha'] for x in verified)
    after = read_state(api)
    audit.update({'status': 'ADMINISTRATIVE_CONSOLIDATION_COMPLETE' if not audit['skipped'] else 'COMPLETED_WITH_EXPLICIT_SKIPS',
        'after': after, 'before_counts': {'branches': len(s['branches']), 'open_prs': len(s['pull_requests']), 'open_issues': len(s['issues'])},
        'after_counts': {'branches': len(after['branches']), 'open_prs': len(after['pull_requests']), 'open_issues': len(after['issues'])}})
    lines = ['# Repository consolidation — executed result', '',
        f"Status: **{audit['status']}**", '',
        f"- Branch names retired after identical tag preservation: **{len(audit['deleted_branches'])}**.",
        f"- Historical PRs closed unmerged: **{len(audit['archived_prs'])}**.",
        f"- Superseded issues closed: **{audit['closed_issues']}**; #11 and #47 remain.",
        f"- After counts: `{json.dumps(audit['after_counts'])}` (maintenance PR may be added later).",
        f"- Every pre-change source head retained: **{audit['source_heads_retained']}**.",
        '- No experiment, runtime source, model, HF evidence, private input or scientific result was changed.',
        '- Main code integration is not claimed; #47 records remaining tested extraction work.',
        '', '## Retained branch names', '', *[f'- `{x["name"]}`' for x in after['branches']], '',
        '## Closed historical PRs', '', ', '.join('#' + str(n) for n in sorted(audit['archived_prs'])), '',
        '## Explicit skips', '', '```json', json_text(audit['skipped']), '```', '',
        '## Restore historical source', '',
        'The original branch name is now a compatibility tag at the exact old SHA. It is not a release or a result promotion.',
        '```bash', 'git fetch origin tag <historical-name>',
        'git switch -c <new-working-name> refs/tags/<historical-name>', '```', '',
        'Original branch heads, PR descriptions and issue bodies: [BEFORE.json](BEFORE.json).',
        'Verified archive before retirement: [ARCHIVE_PREPARED.json](ARCHIVE_PREPARED.json).',
        'Machine-readable after-state: [AFTER.json](AFTER.json).', '']
    final = api.publish({ROOT + '/AFTER.json': json_text(audit), ROOT + '/RESULT.md': '\n'.join(lines)},
                        'docs(repo): seal executed archive, closure and branch-retirement receipt')
    print(json_text({'status': audit['status'], 'receipt_commit': final,
        'archived_prs': audit['archived_prs'], 'deleted_branch_count': len(audit['deleted_branches']),
        'after_counts': audit['after_counts'], 'source_heads_retained': audit['source_heads_retained']}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--request', required=True)
    args = p.parse_args()
    r = json.loads(Path(args.request).read_text())
    if os.environ.get('GITHUB_REPOSITORY') != REPO or os.environ.get('GITHUB_REF') != 'refs/heads/' + BRANCH:
        raise RuntimeError('This one-shot administration only runs in the authorized repository/branch')
    if r.get('authority_comment') != AUTH or r.get('repository') != REPO:
        raise RuntimeError('Authorization binding mismatch')
    api = API(os.environ.get('GITHUB_TOKEN', ''))
    if r.get('mode') == 'inventory':
        inventory(api)
    elif r.get('mode') == 'apply':
        apply(api, r['snapshot_commit'], r['snapshot_sha256'])
    else:
        raise RuntimeError('Unknown mode; nothing executed')

if __name__ == '__main__':
    main()
