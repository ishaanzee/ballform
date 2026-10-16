#!/usr/bin/env python3
"""Write, check and commit label rows without hand-quoting CSV.

    uv run --inexact python scripts/labeling/rows.py add eval/staging/dev/<event>.csv --job 1a2b3c4d5e6f \\
        --release 3.45 --outcome missed --shot-type "jump shot" --distance 26 --zone above_break_three \\
        --shooter "#15 LAL white (Austin Reaves)" --source espn_pbp --notes "WAS@LAL 2026-03-30 ESPN event ..."
    uv run --inexact python scripts/labeling/rows.py check eval/staging/dev/<event>.csv
    uv run --inexact python scripts/labeling/rows.py append eval/staging/dev/<event>.csv eval/labels.csv
    uv run --inexact python scripts/labeling/rows.py commit eval/staging/dev/<event>.csv eval/labels.csv \\
        -m "label six WAS@LAL 2026-03-30 clips"

check validates the header, the column count of every row, the values (with
app.evaluation.read_labels), that each job exists under data/jobs with input.mp4 and
settings.json, and that the notes start with the game ('AAA@BBB YYYY-MM-DD ...' or
'FIXED youtube:<id> ...'). append adds the rows to a target CSV (creating it with the
header), refusing job_ids already in any eval/*.csv. commit does check + append + a git
commit of only the target file, holding an exclusive lock (eval/staging/.commit.lock)
so parallel labelers never interleave; it prints the commit hash.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import io
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evaluation import OUTCOMES, SHOT_TYPES, ZONES, game_of, read_labels  # noqa: E402

HEADER = ["job_id", "release_s", "outcome", "shot_type", "distance_ft", "zone", "shooter", "source", "notes"]
EVAL_DIR = ROOT / "eval"
JOBS_DIR = ROOT / "data" / "jobs"
LOCK = EVAL_DIR / "staging" / ".commit.lock"
TRAILER = "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
# NBA abbreviations as the eval CSVs write them.
TEAMS = {"ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW", "HOU", "IND", "LAC", "LAL", "MEM",
         "MIA", "MIL", "MIN", "NOP", "NYK", "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"}
GAME_START = re.compile(r"^([A-Z]{2,4})@([A-Z]{2,4}) (\d{4})-(\d{2})-(\d{2}) ")
FIXED_START = re.compile(r"^FIXED youtube:[A-Za-z0-9_-]{11}\b")
JOB_ID = re.compile(r"^[0-9a-f]{12}$")
SOURCES = {"espn_pbp", "nba_pbp", "tape", ""}


def notes_problem(notes: str) -> str | None:
    """Why the notes do not start with the game, or None if they do."""
    notes = notes or ""
    if FIXED_START.match(notes):
        return None if game_of(notes) else "FIXED notes must name youtube:<11-char id>"
    found = GAME_START.match(notes)
    if not found:
        return "notes must start with 'AAA@BBB YYYY-MM-DD ' or 'FIXED youtube:<id> '"
    away, home = found.group(1), found.group(2)
    bad = [t for t in (away, home) if t not in TEAMS]
    if bad:
        return f"unknown team abbreviation(s) {', '.join(bad)}: use the NBA ones ({' '.join(sorted(TEAMS))})"
    if away == home:
        return "away and home teams are the same"
    month, day = int(found.group(4)), int(found.group(5))
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return "bad date"
    return None if game_of(notes) else "app.evaluation.game_of cannot parse the game"


def row_problems(row: dict, check_job: bool = True) -> list[str]:
    """Problems with one row as a dict of the nine columns (values already stripped)."""
    problems = []
    job = row.get("job_id", "")
    if not JOB_ID.match(job):
        problems.append(f"job_id {job!r} is not 12 hex characters")
    elif check_job:
        job_dir = JOBS_DIR / job
        for name in ("input.mp4", "settings.json"):
            if not (job_dir / name).exists():
                problems.append(f"data/jobs/{job}/{name} is missing")
    release = row.get("release_s", "")
    if release:
        try:
            value = float(release)
            if value < 0 or value > 60:
                problems.append(f"release_s {release} is outside the clip")
        except ValueError:
            problems.append(f"release_s {release!r} is not a number")
        if not row.get("outcome"):
            problems.append("a shot row needs an outcome (made, missed or unclear)")
    else:
        extra = [k for k in ("outcome", "shot_type", "distance_ft", "zone", "shooter", "source") if row.get(k)]
        if extra:
            problems.append(f"a no-shot row (empty release_s) has only job_id and notes, not {', '.join(extra)}")
    if row.get("outcome", "").lower() not in OUTCOMES:
        problems.append(f"outcome must be one of {sorted(OUTCOMES - {''})}")
    if row.get("shot_type", "").lower() not in SHOT_TYPES:
        problems.append(f"shot_type must be one of {sorted(SHOT_TYPES - {''})} or empty")
    if row.get("zone", "").lower() not in ZONES:
        problems.append(f"zone must be one of {sorted(ZONES - {''})} or empty")
    if row.get("distance_ft"):
        try:
            float(row["distance_ft"])
        except ValueError:
            problems.append(f"distance_ft {row['distance_ft']!r} is not a number")
        if row.get("source") not in ("espn_pbp", "nba_pbp", "tape"):
            problems.append("a row with distance_ft needs source espn_pbp (or tape)")
    if row.get("source", "") not in SOURCES:
        problems.append(f"source must be espn_pbp, nba_pbp, tape or empty, not {row.get('source')!r}")
    why = notes_problem(row.get("notes", ""))
    if why:
        problems.append(why)
    return problems


def check(path: Path, check_jobs: bool = True, quiet: bool = False) -> list[dict]:
    """Validate a rows CSV; raises SystemExit listing every problem. Returns the rows."""
    if not path.exists():
        raise SystemExit(f"no file {path}")
    raw = path.read_bytes()
    problems = []
    if b"\r" in raw:
        problems.append("file has CR line endings; rows must end in \\n only")
    text = raw.decode("utf-8")
    records = list(csv.reader(io.StringIO(text, newline="")))
    if not records:
        raise SystemExit(f"{path}: empty file")
    if records[0] != HEADER:
        problems.append(f"header must be exactly {','.join(HEADER)}")
    rows = []
    seen = set()
    for line, record in enumerate(records[1:], start=2):
        if not record:
            problems.append(f"line {line}: blank line")
            continue
        if len(record) != len(HEADER):
            problems.append(f"line {line}: {len(record)} columns, not {len(HEADER)} (an unquoted comma?)")
            continue
        row = {k: v.strip() for k, v in zip(HEADER, record)}
        for p in row_problems(row, check_jobs):
            problems.append(f"line {line} ({row['job_id']}): {p}")
        key = (row["job_id"], row["release_s"])
        if key in seen:
            problems.append(f"line {line}: duplicate row for {row['job_id']} at {row['release_s'] or 'no shot'}")
        seen.add(key)
        rows.append(row)
    try:
        read_labels(path)
    except ValueError as exc:
        problems.append(f"app.evaluation.read_labels: {exc}")
    if problems:
        raise SystemExit(f"{path}: {len(problems)} problem(s)\n  " + "\n  ".join(problems))
    if not quiet:
        clips = {r["job_id"] for r in rows}
        no_shot = sum(not r["release_s"] for r in rows)
        games = sorted({game_of(r["notes"]) for r in rows})
        print(f"{path}: OK, {len(rows)} rows ({len(rows) - no_shot} attempts, {no_shot} no-shot) in "
              f"{len(clips)} clips; games: {', '.join(games)}")
    return rows


def structure_check(path: Path) -> None:
    """A target CSV's header, column counts and values (older rows predate the notes rules)."""
    with path.open(newline="") as handle:
        records = list(csv.reader(handle))
    if not records or records[0] != HEADER:
        raise SystemExit(f"{path}: header must be exactly {','.join(HEADER)}")
    bad = [str(n) for n, r in enumerate(records[1:], start=2) if r and len(r) != len(HEADER)]
    if bad:
        raise SystemExit(f"{path}: lines {', '.join(bad)} do not have {len(HEADER)} columns")
    try:
        read_labels(path)
    except ValueError as exc:
        raise SystemExit(f"{path}: {exc}") from None


def write_rows(path: Path, rows: list[dict]) -> None:
    """Append rows with the csv module (QUOTE_MINIMAL, \\n endings), adding the header to a new file."""
    new = not path.exists() or path.stat().st_size == 0
    if not new:
        with path.open("rb") as handle:
            handle.seek(-1, 2)
            needs_newline = handle.read(1) != b"\n"
    with path.open("a", newline="") as handle:
        if not new and needs_newline:
            handle.write("\n")
        writer = csv.writer(handle, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        if new:
            writer.writerow(HEADER)
        for row in rows:
            writer.writerow([row.get(k, "") for k in HEADER])


def existing_jobs(exclude: Path | None = None) -> dict[str, str]:
    found = {}
    for csv_path in sorted(EVAL_DIR.glob("*.csv")):
        if exclude is not None and csv_path.resolve() == exclude.resolve():
            continue
        with csv_path.open(newline="") as handle:
            for record in csv.reader(handle):
                if record and JOB_ID.match(record[0].strip()):
                    found.setdefault(record[0].strip(), csv_path.name)
    return found


def add(args) -> int:
    row = {"job_id": args.job, "release_s": args.release or "", "outcome": args.outcome or "",
           "shot_type": args.shot_type or "", "distance_ft": args.distance or "", "zone": args.zone or "",
           "shooter": args.shooter or "", "source": args.source or "", "notes": args.notes}
    row = {k: " ".join(str(v).split()) for k, v in row.items()}
    problems = row_problems(row)
    if problems:
        raise SystemExit("row not added:\n  " + "\n  ".join(problems))
    if args.rows.exists():
        check(args.rows, quiet=True)
        with args.rows.open(newline="") as handle:
            if any(r.get("job_id") == row["job_id"] and r.get("release_s") == row["release_s"]
                   for r in csv.DictReader(handle)):
                raise SystemExit(f"{args.rows} already has a row for {row['job_id']} at {row['release_s'] or 'no shot'}")
    args.rows.parent.mkdir(parents=True, exist_ok=True)
    write_rows(args.rows, [row])
    rows = check(args.rows, quiet=True)
    print(f"added {row['job_id']} {row['release_s'] or '(no shot)'} {row['outcome']} to {args.rows} "
          f"({len(rows)} rows)")
    return 0


def append_rows(staging: Path, target: Path) -> list[dict]:
    rows = check(staging)
    taken = existing_jobs()
    clash = sorted({f"{r['job_id']} (in {taken[r['job_id']]})" for r in rows if r["job_id"] in taken})
    if clash:
        raise SystemExit(f"refusing: job_ids already in eval CSVs: {', '.join(clash)}")
    if target.exists():
        structure_check(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_rows(target, rows)
    structure_check(target)
    print(f"appended {len(rows)} rows to {target}")
    return rows


def append(args) -> int:
    append_rows(args.rows, args.target)
    return 0


def git(*cmd: str, retries: int = 8) -> subprocess.CompletedProcess:
    for attempt in range(retries):
        result = subprocess.run(["git", "-C", str(ROOT), *cmd], capture_output=True, text=True)
        if result.returncode == 0 or "index.lock" not in (result.stderr + result.stdout):
            return result
        time.sleep(0.5 + attempt)
    return result


def commit(args) -> int:
    target = args.target.resolve()
    if target.parent != EVAL_DIR.resolve() or target.suffix != ".csv":
        raise SystemExit("commit targets an eval/*.csv file (labels.csv, holdout3.csv, fixed.csv, ...)")
    message = args.message.strip()
    if not message:
        raise SystemExit("give a commit message with -m")
    if TRAILER not in message:
        message = f"{message}\n\n{TRAILER}"
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w") as lock:
        print(f"waiting for {LOCK} ...", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)  # blocks; released automatically if this process dies
        rel = str(target.relative_to(ROOT))
        status = git("status", "--porcelain", "--", rel)
        if status.stdout.strip():
            raise SystemExit(f"{rel} has uncommitted changes from someone else; not committing:\n{status.stdout}")
        append_rows(args.rows, target)
        added = git("add", "--", rel)
        if added.returncode:
            raise SystemExit(f"git add failed: {added.stderr}")
        done = git("commit", "-m", message, "--", rel)
        if done.returncode:
            raise SystemExit(f"git commit failed (rows were appended to {rel} but not committed):\n"
                             f"{done.stdout}{done.stderr}")
        head = git("rev-parse", "HEAD").stdout.strip()
    print(f"committed {rel}: {head}")
    print(head)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check", help="validate a rows CSV")
    c.add_argument("rows", type=Path)
    c.add_argument("--no-jobs", action="store_true", help="skip the data/jobs existence check")
    a = sub.add_parser("add", help="append one row to a staging CSV")
    a.add_argument("rows", type=Path)
    a.add_argument("--job", required=True)
    a.add_argument("--release", help="release_s; leave out for a no-shot clip")
    a.add_argument("--outcome", choices=["made", "missed", "unclear"])
    a.add_argument("--shot-type", choices=["jump shot", "floater", "layup", "dunk", "tip"])
    a.add_argument("--distance", help="distance_ft from the play-by-play (only the pbp attempt)")
    a.add_argument("--zone", choices=["paint", "midrange", "corner_three", "above_break_three"])
    a.add_argument("--shooter", help="e.g. '#15 LAL white (Austin Reaves)'")
    a.add_argument("--source", choices=["espn_pbp", "nba_pbp", "tape"])
    a.add_argument("--notes", required=True)
    p = sub.add_parser("append", help="append a staging CSV to a target CSV")
    p.add_argument("rows", type=Path)
    p.add_argument("target", type=Path)
    m = sub.add_parser("commit", help="check, append and git-commit a staging CSV under a lock")
    m.add_argument("rows", type=Path)
    m.add_argument("target", type=Path)
    m.add_argument("-m", "--message", required=True)
    args = parser.parse_args(argv)
    if args.command == "check":
        check(args.rows, check_jobs=not args.no_jobs)
        return 0
    return {"add": add, "append": append, "commit": commit}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
