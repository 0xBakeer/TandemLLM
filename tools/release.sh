#!/usr/bin/env bash
# One command for a release (OPS-223): preflight, the box gate, tag, push, the GitHub release, verify.
#
#   bash tools/release.sh X.Y.Z --notes FILE --box HOST [options]        every section, in order
#   bash tools/release.sh X.Y.Z --notes FILE --box HOST --dry-run        every section but tag, push and
#                                                                        release, whose commands are printed
#   bash tools/release.sh X.Y.Z --only verify                            one section (read-only for verify)
#   bash tools/release.sh X.Y.Z --only bump                              write X.Y.Z into the version lines
#
# Sections (--only takes a comma-separated list; the default is
# preflight,gate,tag,push,release,milestone,verify):
#
#   bump       VERSION, install.sh's default tag, the manual's \version, and the lines that point at the
#              previous tag (the install commands and --ref defaults of README.md, docs/setup.md and the
#              manual, the generated install flags); prose (CHANGELOG.md, README's release paragraph,
#              install.sh's release comment, the manual's changelog) stays the author's. Writes files, no commit.
#   preflight  read-only: a clean tree on main; origin's main an ancestor of HEAD (fast-forward only); no
#              vX.Y.Z on origin; VERSION, install.sh, the manual's \version and the pointer lines say
#              X.Y.Z, none still says the previous tag; CHANGELOG.md's first section and the manual's
#              changelog have X.Y.Z; no commit since the previous tag carries a co-author or session trailer
#              (subjects over 72 characters are listed); the notes file is there (RELEASE_NOTES_CHECK, a
#              command, runs on it; RELEASE_NOTES_DENY, a regex, must not match it); gh is logged in; each
#              weight pack a profile added or changed since the previous tag (QSE_WEIGHTS_HF at
#              QSE_WEIGHTS_REVISION), and each --hf, answers an anonymous HEAD with 200 for README.md and
#              config.json
#   gate       on the box (ssh --box): a git archive of HEAD into ~/release-gate-X.Y.Z-<sha>, then
#              tools/release_box_gate.sh there under nohup (a dropped ssh session does not stop it), polled
#              until it ends: the CPU suite through tests/script_suite.py, the Setup Guide gate
#              (tools/release_gate_setup.sh), client_smoke on the conductor path and the raw path against
#              --smoke-base, whose /health must already name X.Y.Z (deploy the candidate first). --skip
#              suite,setup,smoke leaves parts out.
#   tag        the annotated tag vX.Y.Z "TandemLLM X.Y.Z" on HEAD (git's configured identity)
#   push       HEAD as origin's main and the tag, no force
#   release    one GitHub release, "TandemLLM X.Y.Z", the notes file, latest. CI's manual workflow
#              attaches the PDF on the tag push; when it created a bare release first, this edits that one.
#   milestone  the GitHub milestone vX.Y.Z is closed when it is open, and the next patch's, vX.Y.(Z+1),
#              is created when it is missing (CONTRIBUTING.md: a milestone is a release)
#   verify     read-only, against what is published: the tag on origin (and the local tag's commit), main
#              containing it, the release's title, draft state and latest flag, CI's manual PDF (waits up
#              to --asset-wait minutes), README.md, install.sh and VERSION at the tag (each body saved to a
#              file and then searched: a reader that stops early under a pipe makes curl fail with
#              "(56) Failure writing output"), anonymous HTTP 200 for install.sh, the manual asset and the
#              weight packs' README.md and config.json
#
# --dry-run runs preflight, gate and verify (verify only lists its checks while the tag is not on origin)
# and prints the tag, push, release and milestone commands without running them. Exit 0 when every
# section run passed. Bash 3.2 (macOS) compatible: no associative arrays, no empty-array expansion under set -u.
set -euo pipefail

usage() { sed -n '2,/^set -euo pipefail$/p' "$0" | sed '$d'; }

VERSION=""; NOTES=""; BOX="${RELEASE_BOX:-}"; BOX_PY="${RELEASE_BOX_PY:-python3}"
BOX_PYTHONPATH="${RELEASE_BOX_PYTHONPATH:-}"; SMOKE_BASE="http://127.0.0.1:8000/v1"; SKIP=""; ONLY=""
DRY=0; ASSET_WAIT=30; GATE_WAIT=180; REMOTE=origin; GH_REPO=""; HF_EXTRA=""; HF_AUTO=1; JOBS=6
while [ $# -gt 0 ]; do
    case "$1" in
        --notes) NOTES="$2"; shift 2 ;;
        --box) BOX="$2"; shift 2 ;;
        --box-py) BOX_PY="$2"; shift 2 ;;
        --box-pythonpath) BOX_PYTHONPATH="$2"; shift 2 ;;
        --smoke-base) SMOKE_BASE="$2"; shift 2 ;;
        --skip) SKIP="$2"; shift 2 ;;
        --only) ONLY="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        --asset-wait) ASSET_WAIT="$2"; shift 2 ;;
        --gate-wait) GATE_WAIT="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        --remote) REMOTE="$2"; shift 2 ;;
        --repo) GH_REPO="$2"; shift 2 ;;
        --hf) HF_EXTRA="$HF_EXTRA $2"; shift 2 ;;
        --no-hf-auto) HF_AUTO=0; shift ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "unknown option: $1" >&2; exit 2 ;;
        *) [ -z "$VERSION" ] || { echo "one version only: $1" >&2; exit 2; }; VERSION="$1"; shift ;;
    esac
done
case "$VERSION" in
    [0-9]*.[0-9]*.[0-9]*) ;;
    *) echo "usage: tools/release.sh X.Y.Z [--notes FILE] [--box HOST] [--dry-run] [--only SECTIONS] (--help)" >&2; exit 2 ;;
esac
printf '%s' "$VERSION" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$' || { echo "not a version: $VERSION" >&2; exit 2; }
TAG="v$VERSION"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET="tandemllm-manual-$VERSION.pdf"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/tandem-release.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
FAILS=0

g() { git -C "$ROOT" "$@"; }
say() { echo "[release] $*"; }
pass() { echo "PASS  $*"; }
fail() { echo "FAIL  $*"; FAILS=$((FAILS + 1)); }
die() { echo "STOP: $*" >&2; exit 1; }
want() {                         # want SECTION: in --only, or in the default list
    local list="${ONLY:-preflight,gate,tag,push,release,milestone,verify}"
    case ",$list," in *",$1,"*) return 0 ;; *) return 1 ;; esac
}
skipped() { case ",$SKIP," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
run() {                          # a command that publishes or changes refs: printed, run unless --dry-run
    echo "+ $*"
    if [ "$DRY" = 1 ]; then return 0; fi
    "$@"
}
has() {                          # has FILE FIXED-STRING: the file holds the string (no pipe)
    grep -qF -- "$2" "$1"
}
ver_gt() {                       # ver_gt A B: version A is above version B
    local a1 a2 a3 b1 b2 b3
    IFS=. read -r a1 a2 a3 <<EOF
$1
EOF
    IFS=. read -r b1 b2 b3 <<EOF
$2
EOF
    [ "$a1" -gt "$b1" ] && return 0; [ "$a1" -lt "$b1" ] && return 1
    [ "$a2" -gt "$b2" ] && return 0; [ "$a2" -lt "$b2" ] && return 1
    [ "$a3" -gt "$b3" ]
}
fetch() {                        # fetch URL FILE: the body into a file (curl's exit status), anonymous
    curl -fsSL --retry 2 -o "$2" "$1"
}
code_of() {                      # code_of URL [HEAD]: the HTTP status after redirects, anonymous
    if [ "${2:-}" = HEAD ]; then
        curl -sIL -o /dev/null -w '%{http_code}' "$1" || true
    else
        curl -sL -o /dev/null -w '%{http_code}' "$1" || true
    fi
}

# the repository on GitHub, from the remote's URL unless --repo names it
if [ -z "$GH_REPO" ]; then
    url="$(g remote get-url "$REMOTE" 2>/dev/null || true)"
    GH_REPO="$(printf '%s\n' "$url" | sed -E 's#^(git@github\.com:|https://github\.com/|ssh://git@github\.com/)##; s#\.git$##')"
    case "$GH_REPO" in */*) ;; *) die "cannot read the GitHub repository from $REMOTE's URL ($url): pass --repo OWNER/NAME" ;; esac
fi

# the previous release: the highest vA.B.C below this one that HEAD (or the tag) contains
prev_tag() {
    local ref="$1" t best=""
    g tag --list 'v*' --merged "$ref" > "$WORK/tags" 2>/dev/null || true
    while read -r t; do
        printf '%s' "$t" | grep -Eq '^v[0-9]+\.[0-9]+\.[0-9]+$' || continue
        [ "$t" = "$TAG" ] && continue
        ver_gt "$VERSION" "${t#v}" || continue
        if [ -z "$best" ] || ver_gt "${t#v}" "${best#v}"; then best="$t"; fi
    done < "$WORK/tags"
    echo "$best"
}

# the weight packs to check: profiles added or changed since the previous tag, and --hf REPO@REV
hf_packs() {                     # hf_packs FROM TO -> lines "repo rev"
    local from="$1" to="$2" f repo rev
    if [ "$HF_AUTO" = 1 ] && [ -n "$from" ]; then
        g diff --name-only --diff-filter=AM "$from" "$to" -- 'ops/profiles/*.env' > "$WORK/profiles" || true
        while read -r f; do
            [ -n "$f" ] || continue
            g show "$to:$f" > "$WORK/profile.env" 2>/dev/null || continue
            repo="$(sed -n 's/^QSE_WEIGHTS_HF=\([^$ ]*\)$/\1/p' "$WORK/profile.env" | tail -1)"
            rev="$(sed -n 's/^QSE_WEIGHTS_REVISION=\([0-9a-f]\{7,40\}\)$/\1/p' "$WORK/profile.env" | tail -1)"
            [ -n "$repo" ] && [ -n "$rev" ] && echo "$repo $rev"
        done < "$WORK/profiles"
    fi
    for f in $HF_EXTRA; do
        echo "${f%@*} ${f#*@}"
    done
}
check_hf() {                     # check_hf FROM TO
    local repo rev name code n=0
    hf_packs "$1" "$2" | sort -u > "$WORK/packs"
    while read -r repo rev; do
        [ -n "$repo" ] || continue
        for name in README.md config.json; do
            code="$(code_of "https://huggingface.co/$repo/resolve/$rev/$name" HEAD)"
            n=$((n + 1))
            if [ "$code" = 200 ]; then pass "anonymous $repo@${rev:0:8}/$name: HTTP 200"
            else fail "anonymous $repo@${rev:0:8}/$name: HTTP $code (a download needs a token, or the file is missing)"; fi
        done
    done < "$WORK/packs"
    [ "$n" -gt 0 ] || say "no weight pack added or changed since ${1:-the start} (--hf REPO@REV checks one)"
}

# ------------------------------------------------------------------------------------------------ bump
if want bump; then
    say "== bump to $VERSION"
    PREV="$(prev_tag HEAD)"
    [ -n "$PREV" ] || die "no earlier vA.B.C tag in HEAD's history"
    OLD="${PREV#v}"
    say "previous release $PREV"
    printf '%s\n' "$VERSION" > "$ROOT/VERSION"
    # the pointer lines, by exact pattern (python3: sed's alternation and in-place flags differ between
    # BSD and GNU sed)
    python3 - "$ROOT" "$OLD" "$VERSION" <<'PY'
import re, sys
root, old, new = sys.argv[1:]
ov = re.escape("v" + old)
tag = "v" + new
END = r"(?=[^0-9.]|\.[^0-9]|$)"          # v0.13.0, not v0.13.01 or v0.13.0.1


def edit(path, fn):
    p = f"{root}/{path}"
    try:
        text = open(p, encoding="utf-8").read()
    except FileNotFoundError:
        return
    out = "".join(fn(line) for line in text.splitlines(keepends=True))
    if out != text:
        open(p, "w", encoding="utf-8").write(out)
        print("  " + path)


every = lambda line: re.sub(ov + END, tag, line)
edit("install.sh", lambda l: l.replace('TANDEM_REF="${TANDEM_REF:-v%s}"' % old, 'TANDEM_REF="${TANDEM_REF:-%s}"' % tag))
edit("manual/preamble.tex", lambda l: l.replace("\\newcommand{\\version}{%s}" % old, "\\newcommand{\\version}{%s}" % new))
# files whose every mention of the previous tag points at the current release
for f in ("docs/setup.md", "manual/chapters/install-engine.tex", "manual/chapters/troubleshooting.tex",
          "manual/generated/install-flags.tex"):
    edit(f, every)
# README: the install command, the clone and missing-tag sentences, the --ref row; not the release paragraph
KEYS = re.compile(r"install\.sh|--ref REF|It clones the engine at the tag|Where the tag")
edit("README.md", lambda l: every(l) if KEYS.search(l) and not l.startswith("The current release is") else l)
PY
    say "written; still the author's: CHANGELOG.md, README.md's release paragraph, install.sh's release"
    say "comment, manual/chapters/changelog.tex; then preflight checks every line"
    g diff --stat
fi

# ------------------------------------------------------------------------------------------- preflight
if want preflight; then
    say "== preflight $TAG ($GH_REPO, remote $REMOTE)"
    [ -z "$(g status --porcelain --untracked-files=no)" ] && pass "no uncommitted change to a tracked file" \
        || fail "uncommitted changes: $(g status --porcelain --untracked-files=no | wc -l | tr -d ' ') files"
    BR="$(g symbolic-ref --short -q HEAD || echo detached)"
    [ "$BR" = main ] && pass "on main" || fail "on $BR, not main"
    HEAD_SHA="$(g rev-parse HEAD)"
    g fetch -q "$REMOTE" main || fail "git fetch $REMOTE main"
    OM="$(g rev-parse "$REMOTE/main" 2>/dev/null || echo none)"
    if [ "$OM" != none ] && g merge-base --is-ancestor "$OM" "$HEAD_SHA"; then
        pass "fast-forward: $REMOTE/main ${OM:0:8} -> HEAD ${HEAD_SHA:0:8} ($(g rev-list --count "$OM..$HEAD_SHA") commits)"
    else
        fail "$REMOTE/main ${OM:0:8} is not an ancestor of HEAD: rebase the release on it"
    fi
    g ls-remote --tags "$REMOTE" "refs/tags/$TAG" > "$WORK/remote-tag" || fail "git ls-remote $REMOTE"
    [ -s "$WORK/remote-tag" ] && fail "$REMOTE already has $TAG" || pass "$REMOTE has no $TAG yet"
    if g rev-parse -q --verify "refs/tags/$TAG" > /dev/null; then
        if [ "$(g cat-file -t "$TAG")" = tag ] && [ "$(g rev-parse "$TAG^{commit}")" = "$HEAD_SHA" ]; then
            pass "the local tag $TAG is annotated and on HEAD"
        else
            fail "a local tag $TAG exists and is not an annotated tag on HEAD"
        fi
    fi
    PREV="$(prev_tag HEAD)"
    if [ -n "$PREV" ]; then pass "previous release $PREV"; else fail "no earlier vA.B.C tag in HEAD's history"; fi
    OLD="${PREV#v}"

    g show HEAD:VERSION > "$WORK/VERSION" 2>/dev/null || : > "$WORK/VERSION"
    [ "$(cat "$WORK/VERSION")" = "$VERSION" ] && pass "VERSION $VERSION" || fail "VERSION says $(cat "$WORK/VERSION")"
    pin() {                      # pin FILE STRING [WHAT]: HEAD's FILE holds STRING
        g show "HEAD:$1" > "$WORK/pin" 2>/dev/null || : > "$WORK/pin"
        if has "$WORK/pin" "$2"; then pass "$1: ${3:-$2}"; else fail "$1 lacks: ${3:-$2}"; fi
    }
    pin install.sh "TANDEM_REF=\"\${TANDEM_REF:-$TAG}\"" "default tag $TAG"
    pin manual/preamble.tex "\\newcommand{\\version}{$VERSION}" "\\version $VERSION"
    pin README.md "raw.githubusercontent.com/$GH_REPO/$TAG/install.sh" "install command at $TAG"
    pin README.md "default \`$TAG\`" "--ref default $TAG"
    pin README.md "The current release is \`$TAG\`" "release paragraph names $TAG"
    pin docs/setup.md "raw.githubusercontent.com/$GH_REPO/$TAG/install.sh" "install command at $TAG"
    pin manual/chapters/install-engine.tex "raw.githubusercontent.com/$GH_REPO/$TAG/install.sh" "install command at $TAG"
    pin manual/generated/install-flags.tex "(default $TAG)" "--ref default $TAG"
    if [ -n "$OLD" ]; then
        for f in docs/setup.md manual/chapters/install-engine.tex manual/chapters/troubleshooting.tex \
                 manual/generated/install-flags.tex; do
            g show "HEAD:$f" > "$WORK/pin" 2>/dev/null || continue
            if grep -nE "v${OLD//./\\.}([^0-9.]|\\.[^0-9]|\$)" "$WORK/pin" > "$WORK/stale"; then
                fail "$f still points at $PREV: line $(cut -d: -f1 "$WORK/stale" | paste -sd, -)"
            fi
        done
        g show "HEAD:README.md" > "$WORK/pin"
        grep -nE "install\\.sh|--ref REF|It clones the engine at the tag|Where the tag" "$WORK/pin" \
            | grep -v '^[0-9]*:The current release is' > "$WORK/lines" || true
        if grep -E "v${OLD//./\\.}([^0-9.]|\\.[^0-9]|\$)" "$WORK/lines" > "$WORK/stale"; then
            fail "README.md's install lines still point at $PREV: line $(cut -d: -f1 "$WORK/stale" | paste -sd, -)"
        fi
    fi
    g show HEAD:CHANGELOG.md > "$WORK/changelog"
    first="$(grep -m1 '^## ' "$WORK/changelog" || true)"
    if printf '%s\n' "$first" | grep -Eq "^## $(printf '%s' "$VERSION" | sed 's/\./\\./g'), [0-9]{4}-[0-9]{2}-[0-9]{2}\$"; then
        pass "CHANGELOG.md: $first"
    else
        fail "CHANGELOG.md's first section is \"$first\", not \"## $VERSION, YYYY-MM-DD\""
    fi
    pin manual/chapters/changelog.tex "$VERSION & " "changelog table row"
    pin manual/chapters/changelog.tex "\\section{$VERSION" "changelog section"

    if [ -n "$PREV" ]; then
        n=0; long=0
        for c in $(g rev-list "$PREV..HEAD"); do
            n=$((n + 1))
            g log -1 --format=%B "$c" > "$WORK/msg"
            if grep -qiE '^co-authored-by:|generated with|signed-off-by:.*bot|session[_:]' "$WORK/msg"; then
                fail "${c:0:8} carries a co-author or session line: $(g log -1 --format=%s "$c")"
            fi
            s="$(g log -1 --format=%s "$c")"
            if [ "${#s}" -gt 72 ]; then long=$((long + 1)); echo "      note: ${c:0:8} subject ${#s} characters: $s"; fi
        done
        pass "$n commits since $PREV checked for trailers ($long subjects over 72 characters)"
        say "authors since $PREV: $(g log --format='%an' "$PREV..HEAD" | sort | uniq -c | sed 's/^ *//' | paste -sd, -)"
    fi

    if [ -n "$NOTES" ]; then
        if [ -s "$NOTES" ]; then
            pass "notes $NOTES ($(wc -l < "$NOTES" | tr -d ' ') lines)"
            if [ -n "${RELEASE_NOTES_DENY:-}" ] && grep -qiE -- "$RELEASE_NOTES_DENY" "$NOTES"; then
                fail "the notes match RELEASE_NOTES_DENY: $(grep -oiE -- "$RELEASE_NOTES_DENY" "$NOTES" | sort -u | paste -sd, -)"
            fi
            if [ -n "${RELEASE_NOTES_CHECK:-}" ]; then
                if $RELEASE_NOTES_CHECK "$NOTES" > "$WORK/notes-check" 2>&1; then pass "notes: $RELEASE_NOTES_CHECK"
                else fail "notes: $RELEASE_NOTES_CHECK failed: $(tail -3 "$WORK/notes-check" | paste -sd' ' -)"; fi
            fi
        else
            fail "notes missing or empty: $NOTES"
        fi
    elif want release; then
        fail "--notes FILE is needed for the release section"
    fi
    if want release || want milestone || want verify; then
        gh auth status -h github.com > /dev/null 2>&1 && pass "gh is logged in" || fail "gh is not logged in (gh auth login)"
    fi
    check_hf "$PREV" HEAD
    [ "$FAILS" = 0 ] || die "preflight: $FAILS failed"
    say "PREFLIGHT: all checks pass"
fi

# ------------------------------------------------------------------------------------------------ gate
if want gate; then
    say "== gate on the box"
    [ -n "$BOX" ] || die "--box HOST (or RELEASE_BOX) names the box the gate runs on"
    SHA="$(g rev-parse --short=8 HEAD)"
    DIR="release-gate-$VERSION-$SHA"
    STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
    say "$BOX:~/$DIR (git archive of $SHA; uncommitted changes are not in it)"
    g archive --format=tar HEAD | ssh "$BOX" "rm -rf ~/$DIR && mkdir -p ~/$DIR && tar -xf - -C ~/$DIR" \
        || die "copying the tree to $BOX"
    # the smoke's API key (RELEASE_SMOKE_KEY) goes to the box on stdin, into a file of mode 600 that the
    # box gate reads and removes; it is on no command line
    if [ -n "${RELEASE_SMOKE_KEY:-}" ]; then
        printf '%s' "$RELEASE_SMOKE_KEY" | ssh "$BOX" "umask 077 && cat > ~/$DIR/.smoke-key" || die "copying the smoke key"
    fi
    # only the gate is in the background, its output in a file: the session returns at once
    ssh "$BOX" "cd ~/$DIR || exit 1; nohup bash tools/release_box_gate.sh \
        --py '$BOX_PY' --pythonpath '$BOX_PYTHONPATH' --jobs '$JOBS' --smoke-base '$SMOKE_BASE' --skip '$SKIP' \
        --version '$VERSION' --out ~/$DIR/results/release-gate/$STAMP > ~/$DIR/release-gate-$STAMP.log 2>&1 < /dev/null &" \
        || die "starting the gate on $BOX"
    say "started; log $BOX:~/$DIR/release-gate-$STAMP.log (waits up to $GATE_WAIT min)"
    deadline=$(( $(date +%s) + GATE_WAIT * 60 ))
    status=""
    while [ "$(date +%s)" -lt "$deadline" ]; do
        status="$(ssh -o ConnectTimeout=20 "$BOX" "cat ~/$DIR/results/release-gate/$STAMP/EXIT 2>/dev/null" || true)"
        [ -n "$status" ] && break
        sleep 30
    done
    ssh "$BOX" "cat ~/$DIR/results/release-gate/$STAMP/SUMMARY 2>/dev/null" || true
    case "$status" in
        0) say "GATE: passed ($BOX:~/$DIR/results/release-gate/$STAMP)" ;;
        "") die "the gate did not end within $GATE_WAIT min; it still runs: $BOX:~/$DIR/release-gate-$STAMP.log" ;;
        *) die "the gate failed (exit $status): $BOX:~/$DIR/results/release-gate/$STAMP" ;;
    esac
fi

# ------------------------------------------------------------------------------------------- tag, push
if want tag; then
    say "== tag"
    if g rev-parse -q --verify "refs/tags/$TAG" > /dev/null; then
        [ "$(g rev-parse "$TAG^{commit}")" = "$(g rev-parse HEAD)" ] || die "a tag $TAG exists on another commit"
        say "$TAG is already on HEAD"
    else
        run git -C "$ROOT" tag -a "$TAG" -m "TandemLLM $VERSION" HEAD
    fi
fi
if want push; then
    say "== push"
    run git -C "$ROOT" push "$REMOTE" "HEAD:refs/heads/main" "refs/tags/$TAG:refs/tags/$TAG"
fi
if want release; then
    say "== release"
    [ -n "$NOTES" ] && [ -s "$NOTES" ] || die "--notes FILE is needed for the release"
    if [ "$DRY" = 0 ] && gh release view "$TAG" -R "$GH_REPO" > /dev/null 2>&1; then
        # CI's manual job created a bare release first: give it its title and notes
        run gh release edit "$TAG" -R "$GH_REPO" --title "TandemLLM $VERSION" --notes-file "$NOTES" --latest
    else
        run gh release create "$TAG" -R "$GH_REPO" --verify-tag --latest --title "TandemLLM $VERSION" --notes-file "$NOTES"
    fi
    say "CI's manual workflow attaches $ASSET when its build on the tag ends"
fi

# ------------------------------------------------------------------------------------------- milestone
if want milestone; then
    say "== milestone"
    IFS=. read -r M1 M2 M3 <<EOF
$VERSION
EOF
    NEXT="v$M1.$M2.$((M3 + 1))"
    gh api "repos/$GH_REPO/milestones?state=all&per_page=100" \
        -q '.[] | "\(.number) \(.state) \(.title)"' > "$WORK/milestones" || die "reading $GH_REPO's milestones"
    line="$(awk -v t="$TAG" '$3 == t' "$WORK/milestones" | head -1)"
    if [ -z "$line" ]; then
        say "no milestone $TAG on $GH_REPO: nothing to close"
    elif [ "$(echo "$line" | cut -d' ' -f2)" = open ]; then
        run gh api -X PATCH "repos/$GH_REPO/milestones/$(echo "$line" | cut -d' ' -f1)" -f state=closed
    else
        say "milestone $TAG is already closed"
    fi
    if awk -v t="$NEXT" '$3 == t { f = 1 } END { exit !f }' "$WORK/milestones"; then
        say "milestone $NEXT exists"
    else
        run gh api -X POST "repos/$GH_REPO/milestones" -f title="$NEXT" -f description="Next patch release."
    fi
fi

# ---------------------------------------------------------------------------------------------- verify
if want verify; then
    say "== verify $TAG on $GH_REPO"
    git ls-remote "https://github.com/$GH_REPO.git" "refs/tags/$TAG^{}" "refs/tags/$TAG" "refs/heads/main" > "$WORK/refs" \
        || : > "$WORK/refs"
    TAG_SHA="$(grep -F "refs/tags/$TAG^{}" "$WORK/refs" | cut -c1-40 || true)"
    if [ -z "$TAG_SHA" ]; then
        if [ "$DRY" = 1 ]; then
            say "(dry run: $TAG is not on $GH_REPO yet; after the push these run)"
            echo "  the tag on origin = the local tag's commit; main contains it; release \"TandemLLM $VERSION\","
            echo "  not a draft, latest; CI's $ASSET (up to $ASSET_WAIT min); README.md, install.sh and VERSION"
            echo "  at $TAG; anonymous HTTP 200: install.sh, the manual asset, each weight pack's README.md, config.json"
        else
            fail "no annotated tag $TAG on $GH_REPO"
        fi
    else
        pass "tag $TAG on $GH_REPO: ${TAG_SHA:0:8}"
        if g rev-parse -q --verify "refs/tags/$TAG" > /dev/null; then
            [ "$(g rev-parse "$TAG^{commit}")" = "$TAG_SHA" ] && pass "the local tag is the same commit" \
                || fail "the local tag $TAG is $(g rev-parse --short=8 "$TAG^{commit}"), origin's ${TAG_SHA:0:8}"
        fi
        MAIN_SHA="$(grep -F "refs/heads/main" "$WORK/refs" | cut -c1-40 || true)"
        if [ "$MAIN_SHA" = "$TAG_SHA" ]; then
            pass "main = $TAG"
        else
            # the objects only (FETCH_HEAD): verify changes no ref of this checkout
            g fetch -q "https://github.com/$GH_REPO.git" main "refs/tags/$TAG" 2>/dev/null || true
            if g merge-base --is-ancestor "$TAG_SHA" "$MAIN_SHA" 2>/dev/null; then
                pass "main ${MAIN_SHA:0:8} contains $TAG"
            else
                fail "main ${MAIN_SHA:0:8} does not contain $TAG (or this checkout lacks the commits)"
            fi
        fi
        if gh release view "$TAG" -R "$GH_REPO" --json name,isDraft,assets > "$WORK/release.json" 2>/dev/null; then
            name="$(gh release view "$TAG" -R "$GH_REPO" --json name -q .name)"
            draft="$(gh release view "$TAG" -R "$GH_REPO" --json isDraft -q .isDraft)"
            [ "$name" = "TandemLLM $VERSION" ] && pass "release title \"$name\"" || fail "release title \"$name\""
            [ "$draft" = false ] && pass "release published (not a draft)" || fail "the release is a draft"
            gh release list -R "$GH_REPO" -L 20 --json tagName,isLatest -q '.[] | select(.isLatest) | .tagName' > "$WORK/latest" || : > "$WORK/latest"
            [ "$(cat "$WORK/latest")" = "$TAG" ] && pass "$TAG is the latest release" \
                || fail "the latest release is $(cat "$WORK/latest"), not $TAG"
            deadline=$(( $(date +%s) + ASSET_WAIT * 60 ))
            while :; do
                gh release view "$TAG" -R "$GH_REPO" --json assets -q '.assets[].name' > "$WORK/assets" 2>/dev/null || : > "$WORK/assets"
                grep -qxF -- "$ASSET" "$WORK/assets" && break
                [ "$(date +%s)" -lt "$deadline" ] || break
                sleep 30
            done
            if grep -qxF -- "$ASSET" "$WORK/assets"; then pass "asset $ASSET"
            else fail "no $ASSET after $ASSET_WAIT min: see the manual workflow's run on $TAG"; fi
        else
            fail "no GitHub release $TAG"
        fi
        RAW="https://raw.githubusercontent.com/$GH_REPO/$TAG"
        if fetch "$RAW/README.md" "$WORK/README.md"; then
            has "$WORK/README.md" "curl -fsSLO $RAW/install.sh" && pass "README.md at $TAG: install command" \
                || fail "README.md at $TAG lacks \"curl -fsSLO $RAW/install.sh\""
        else
            fail "README.md at $TAG: download failed"
        fi
        if fetch "$RAW/install.sh" "$WORK/install.sh"; then
            has "$WORK/install.sh" "TANDEM_REF=\"\${TANDEM_REF:-$TAG}\"" && pass "install.sh default tag $TAG" \
                || fail "install.sh default tag: not $TAG"
        else
            fail "install.sh at $TAG: download failed"
        fi
        if fetch "$RAW/VERSION" "$WORK/VERSION.tag"; then
            [ "$(cat "$WORK/VERSION.tag")" = "$VERSION" ] && pass "VERSION at $TAG: $VERSION" \
                || fail "VERSION at $TAG: $(cat "$WORK/VERSION.tag")"
        else
            fail "VERSION at $TAG: download failed"
        fi
        for u in "$RAW/install.sh" "https://github.com/$GH_REPO/releases/download/$TAG/$ASSET"; do
            code="$(code_of "$u")"
            [ "$code" = 200 ] && pass "anonymous $u: HTTP 200" || fail "anonymous $u: HTTP $code"
        done
        PREV="$(prev_tag "$TAG_SHA" 2>/dev/null || true)"
        check_hf "$PREV" "$TAG_SHA"
    fi
    if [ "$FAILS" != 0 ]; then die "verify: $FAILS failed"; fi
    if [ -z "$TAG_SHA" ]; then say "VERIFY: listed (dry run)"; else say "VERIFY: all checks pass"; fi
fi
exit 0
