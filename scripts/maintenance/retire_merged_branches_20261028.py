#!/usr/bin/env python3
"""Retire six exact, merged repository refs; dry-run unless --apply is explicit.

Requires already authenticated git/gh. Uses a temporary bare repository only.
Archives same-name lightweight tags before atomic, expected-SHA deletions.
Never modifies source worktrees, main, the former active branch, or servers.
This helper was published but NOT executed during the connector consolidation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

REPO = "Leo-6-maker/go2-arx5-hierarchical-vla"
BASELINE = "2080c725cc810f6fe1423cda90076206bf089641"
KEEP = {"main", "pi0267/executor-decoupled-recovery"}
EXPECTED = {
    "maintenance/hopper-september-retirement-20261013": "4f5fe8ea9b4309b2aa8792d60475a1f1daa7d0a8",
    "ops/main-cpu-ci-20261013": "1279ecca379514a2a8040e2a378a67a4161ab4f3",
    "pi0257/corelab-video-evidence-workflow-v1": "1e760ba055106cb85b508711f1466d1cb78a9656",
    "pi0263/storage-subagent-model-audit-go": "b809ee4315b585bd9e77400115ead784e33465ae",
    "pi0264/language-diversity-transfer": "f2eafce181499e6854f879533f3394b5b379524e",
    "maintenance/main-consolidation-20261028": "c1d340d23ed4b65f9a1533ba86d31318abbc3a32",
}
ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "docs/maintenance/2026-10-28/BRANCH_RETIREMENT.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--apply", action="store_true", help="Archive tags, then delete only the six exact verified merged refs")
    return p


def validate_plan(plan: dict) -> dict[str, str]:
    require(plan.get("schema") == "EXACT_MERGED_BRANCH_RETIREMENT_20261028_V1", "Wrong plan schema")
    require(plan.get("repository") == REPO and plan.get("integrated_main_minimum") == BASELINE, "Wrong repository or integration baseline")
    rows = plan.get("candidates", [])
    require(isinstance(rows, list) and len(rows) == len(EXPECTED), "Candidate set changed")
    actual = {row["name"]: row["sha"] for row in rows}
    require(actual == EXPECTED and not (set(actual) & KEEP), "Exact candidate names/SHAs changed; refusing broader deletion")
    require(set(plan.get("keep", [])) == KEEP, "Protected transition refs changed")
    return actual


class Client:
    def __init__(self, cwd: str):
        self.cwd = cwd
        self.env = dict(os.environ)
        self.env.update({"GH_HOST": "github.com", "GIT_TERMINAL_PROMPT": "0", "GIT_LFS_SKIP_SMUDGE": "1"})

    def run(self, argv: list[str], *, allow_failure: bool = False, timeout: int = 180) -> subprocess.CompletedProcess:
        try:
            result = subprocess.run(argv, cwd=self.cwd, env=self.env, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise RuntimeError("Command timed out; do not automatically retry a push. Read remote refs first.") from None
        if result.returncode and not allow_failure:
            # Do not print credentials, environment, authenticated URLs or raw API errors.
            raise RuntimeError(f"{argv[0]} command failed with exit {result.returncode}; check auth/permissions and actual remote state before any retry")
        return result

    def git(self, *args: str, allow_failure: bool = False) -> subprocess.CompletedProcess:
        return self.run(["git", *args], allow_failure=allow_failure)

    def api(self, path: str):
        require(path.startswith("repos/" + REPO), "API outside the exact repository")
        raw = self.run(["gh", "api", "--hostname", "github.com", "--method", "GET", path], timeout=90).stdout
        return json.loads(raw)

    def pages(self, path: str) -> list:
        result = []
        for page in range(1, 21):
            separator = "&" if "?" in path else "?"
            batch = self.api(f"repos/{REPO}/{path}{separator}per_page=100&page={page}")
            require(isinstance(batch, list), "Incomplete/unexpected GitHub list")
            result.extend(batch)
            if len(batch) < 100:
                return result
        raise RuntimeError("Pagination ceiling reached; refusing incomplete dependency inventory")

    def refs(self) -> dict[str, str]:
        names = ["refs/heads/main"]
        for name in EXPECTED:
            names.extend(["refs/heads/" + name, "refs/tags/" + name])
        output = self.git("ls-remote", "--refs", "origin", *names).stdout
        result = {}
        for line in output.splitlines():
            sha, ref = line.split("\t", 1)
            require(ref not in result, "Duplicate remote ref")
            result[ref] = sha
        return result

    def ancestor(self, older: str, newer: str) -> bool:
        result = self.git("merge-base", "--is-ancestor", older, newer, allow_failure=True)
        require(result.returncode in (0, 1), "Cannot prove complete Git ancestry")
        return result.returncode == 0

    def inspect(self) -> dict:
        metadata = self.api(f"repos/{REPO}")
        require(metadata.get("full_name") == REPO and metadata.get("default_branch") == "main", "Repository/default branch changed")
        require(not metadata.get("archived") and not metadata.get("disabled"), "Repository is not writable")
        branches = {b["name"]: b for b in self.pages("branches")}
        pulls = self.pages("pulls?state=open")
        blocked_refs = set()
        for pr in pulls:
            if (pr.get("head", {}).get("repo") or {}).get("full_name") == REPO:
                blocked_refs.add(pr["head"]["ref"])
            if (pr.get("base", {}).get("repo") or {}).get("full_name") == REPO:
                blocked_refs.add(pr["base"]["ref"])
        remote = self.refs()
        main = remote.get("refs/heads/main")
        require(main is not None and "main" in branches and branches["main"]["commit"]["sha"] == main, "Main inventory changed during read")
        self.git("fetch", "--no-tags", "origin", "refs/heads/main:refs/remotes/origin/main")
        fetched = self.git("rev-parse", "refs/remotes/origin/main").stdout.strip()
        require(fetched == main and self.ancestor(BASELINE, main), "Main moved or no longer preserves the integration baseline")
        decisions = []
        for name, sha in EXPECTED.items():
            head = remote.get("refs/heads/" + name)
            tag = remote.get("refs/tags/" + name)
            row = {"name": name, "expected_sha": sha, "live_sha": head, "archive_tag_sha": tag}
            branch = branches.get(name)
            if head is None and branch is None:
                row.update(decision="ALREADY_ABSENT", reason="No deletion will be attempted")
            elif head != sha or branch is None or branch["commit"]["sha"] != sha:
                row.update(decision="BLOCKED", reason="Head moved or inventory disagrees")
            elif branch.get("protected", True):
                row.update(decision="BLOCKED", reason="Protected or protection state unknown")
            elif name in blocked_refs:
                row.update(decision="BLOCKED", reason="Open PR head/base dependency")
            elif not self.ancestor(sha, main):
                row.update(decision="BLOCKED", reason="Not fully reachable from main")
            elif tag not in (None, sha):
                row.update(decision="BLOCKED", reason="Existing same-name tag differs; never replace it")
            else:
                row.update(decision="READY", reason="Exact merged unprotected head without open PR dependency")
            decisions.append(row)
        return {"main": main, "branches": decisions, "open_pr_count": len(pulls)}


def ready(snapshot: dict) -> list[dict]:
    require(not any(r["decision"] == "BLOCKED" for r in snapshot["branches"]), "At least one candidate changed/protected/dependent; no deletion permitted")
    return [r for r in snapshot["branches"] if r["decision"] == "READY"]


def main() -> int:
    args = parser().parse_args()
    validate_plan(json.loads(PLAN.read_text(encoding="utf-8")))
    require(shutil.which("git") is not None and shutil.which("gh") is not None, "Install git and gh and authenticate to the intended GitHub account first")
    with tempfile.TemporaryDirectory(prefix="groot-repo-ref-retirement-") as tmp:
        client = Client(tmp)
        client.git("init", "--bare", ".")
        client.git("remote", "add", "origin", f"https://github.com/{REPO}.git")
        # Only the temporary repository config changes; no global auth configuration.
        client.git("config", "credential.helper", "")
        client.git("config", "credential.https://github.com.helper", "!gh auth git-credential")
        before = client.inspect()
        output = {"schema": "EXACT_MERGED_BRANCH_RETIREMENT_RESULT_V1", "repository": REPO,
                  "observed_at_utc": datetime.now(timezone.utc).isoformat(),
                  "apply_requested": args.apply, "before": before,
                  "kept_by_design": sorted(KEEP), "new_scientific_effects": 0}
        if not args.apply:
            output.update(status="DRY_RUN_NO_REMOTE_WRITES", deleted=[], created_tags=[])
            print(json.dumps(output, ensure_ascii=False, indent=2))
            return 0
        planned = ready(before)
        if not planned:
            output.update(status="NOTHING_TO_DELETE", deleted=[], created_tags=[])
            print(json.dumps(output, ensure_ascii=False, indent=2))
            return 0
        # Refresh before the first mutation. Any candidate change aborts the entire set.
        refresh = client.inspect()
        live = ready(refresh)
        require({r["name"] for r in live} == {r["name"] for r in planned}, "Candidate inventory changed; stop and inspect")
        missing_tags = [r for r in live if r["archive_tag_sha"] is None]
        if missing_tags:
            client.git("push", "--atomic", "origin", *[r["expected_sha"] + ":refs/tags/" + r["name"] for r in missing_tags])
        # Readback recovery tags, ancestry, protection and PR dependencies before deletion.
        final_check = client.inspect()
        final_rows = ready(final_check)
        require({r["name"] for r in final_rows} == {r["name"] for r in planned}, "Head set changed after archive; tags retained, no deletion")
        require(all(r["archive_tag_sha"] == r["expected_sha"] for r in final_rows), "Exact recovery tag not verified")
        leases = ["--force-with-lease=refs/heads/" + r["name"] + ":" + r["expected_sha"] for r in final_rows]
        deletes = [":refs/heads/" + r["name"] for r in final_rows]
        # Leases protect deletion only. No surviving branch/tag is force-updated.
        client.git("push", "--atomic", *leases, "origin", *deletes)
        after = client.refs()
        require(all("refs/heads/" + r["name"] not in after for r in final_rows), "Deletion readback incomplete; do not blindly repeat")
        require(all(after.get("refs/tags/" + r["name"]) == r["expected_sha"] for r in final_rows), "Recovery tag readback mismatch")
        output.update(status="DELETED_EXACT_MERGED_REFS_WITH_VERIFIED_RECOVERY_TAGS",
                      deleted=[r["name"] for r in final_rows],
                      created_tags=[r["name"] for r in missing_tags], after_refs=after)
        print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError, KeyError) as exc:
        print("STOP: " + str(exc), file=sys.stderr)
        raise SystemExit(1)
