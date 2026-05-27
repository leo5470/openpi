#!/usr/bin/env python3
"""Aggregate LIBERO-Plus rollouts into the four artifacts the paper reports.

Dependency-free (stdlib only) so it runs on any client box and in CI without
pandas/numpy.

Inputs
------
--rollouts        JSONL, one object per rollout, written by the eval client:
                    {"suite", "task_idx", "task_name", "success", ...}
                  `success` may be bool or 0/1; extra fields are ignored.
--classification  task_classification.json
                    {suite: [{id, name, category, difficulty_level}, ...]}

The join key is (suite, task_name) against classification[suite][i]["name"].

Outputs
-------
Pretty tables on stdout plus tidy CSVs under --out-dir:
  rollouts_tidy.csv   leaderboard_row.csv   table10.csv
  figure5.csv         subperturbations.csv

Modes
-----
--strict     enforce the full-run sanity assertions; use only when all 10,030
             tasks were rolled out.
--selftest   synthesize a full 10,030-row rollout set from the classification
             file and run the whole pipeline + strict assertions, with no real
             rollout data.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# --- canonical orderings / labels --------------------------------------------
CAT_ORDER = [
    "Camera Viewpoints", "Robot Initial States", "Language Instructions",
    "Light Conditions", "Background Textures", "Sensor Noise", "Objects Layout",
]
CAT_LABEL = {
    "Camera Viewpoints": "Camera", "Robot Initial States": "Robot",
    "Language Instructions": "Language", "Light Conditions": "Light",
    "Background Textures": "Background", "Sensor Noise": "Noise",
    "Objects Layout": "Layout",
}
SUITE_LABEL = {
    "libero_spatial": "Spatial", "libero_object": "Object",
    "libero_goal": "Goal", "libero_10": "Long",  # libero_10 -> "Long" in tables
}
SUITE_ORDER = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

# Expected full-run counts (verified against the shipped JSON).
EXPECT_TOTAL = 10030
EXPECT_BY_CATEGORY = {
    "Camera Viewpoints": 1599, "Robot Initial States": 1550,
    "Language Instructions": 1537, "Light Conditions": 1142,
    "Background Textures": 1076, "Sensor Noise": 1601, "Objects Layout": 1525,
}
EXPECT_BY_SUITE = {
    "libero_spatial": 2402, "libero_object": 2518,
    "libero_goal": 2591, "libero_10": 2519,
}
EXPECT_NULL_DIFFICULTY = 121  # all libero_goal x Light Conditions

VIEW_RE = re.compile(r"_view_(-?\d+)_(-?\d+)_(-?\d+)_(-?\d+)_(-?\d+)_initstate_(-?\d+)$")


# --- I/O ----------------------------------------------------------------------
def load_classification(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return [{**e, "suite": s} for s, lst in data.items() for e in lst]


def load_rollouts(path: Path) -> list[dict]:
    rows = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            sys.exit(f"{path}:{n}: malformed JSON: {exc}")
    return rows


def join(rollouts: list[dict], cls_rows: list[dict]) -> list[dict]:
    """Left-join rollouts -> classification on (suite, task_name)."""
    by_key = {(r["suite"], r["name"]): r for r in cls_rows}
    if len(by_key) != len(cls_rows):
        sys.exit("classification has duplicate (suite, name) keys")
    merged, unmatched = [], []
    seen = set()
    for r in rollouts:
        key = (r["suite"], r["task_name"])
        if key in seen:
            sys.exit(f"duplicate rollout for {key}")
        seen.add(key)
        c = by_key.get(key)
        if c is None:
            unmatched.append(key)
            continue
        merged.append({
            "suite": r["suite"],
            "task_idx": r.get("task_idx"),
            "task_name": r["task_name"],
            "success": 1.0 if bool(r["success"]) else 0.0,
            "steps": r.get("steps"),
            "wall_seconds": r.get("wall_seconds"),
            "category": c["category"],
            "difficulty_level": c.get("difficulty_level"),
        })
    if unmatched:
        sys.exit(f"{len(unmatched)} rollouts have no classification entry, "
                 f"e.g. {unmatched[:3]}")
    return merged


# --- small group-by helpers ---------------------------------------------------
def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def group_mean(rows, keyfn, valfn=lambda r: r["success"]):
    buckets = defaultdict(list)
    for r in rows:
        buckets[keyfn(r)].append(valfn(r))
    return {k: mean(v) for k, v in buckets.items()}, {k: len(v) for k, v in buckets.items()}


def fmt(x):
    return "  -  " if x != x else f"{100 * x:5.1f}"  # NaN check


# --- artifacts ----------------------------------------------------------------
def artifact1_leaderboard(rows):
    """7 categories + a micro-average Total."""
    cat_acc, _ = group_mean(rows, lambda r: r["category"])
    micro = mean([r["success"] for r in rows])
    macro = mean([cat_acc[c] for c in CAT_ORDER if c in cat_acc])
    print("=== Artifact 1: leaderboard row (paper Table 1/2, README) ===")
    print("  " + "  ".join(f"{CAT_LABEL[c]:>10}" for c in CAT_ORDER) +
          f"  {'Total':>10}")
    print("  " + "  ".join(f"{fmt(cat_acc.get(c, float('nan'))):>10}" for c in CAT_ORDER) +
          f"  {fmt(micro):>10}")
    print(f"  (Total above = micro-avg over {len(rows)} rollouts; "
          f"macro-avg of 7 categories = {fmt(macro).strip()})\n")
    out = {CAT_LABEL[c]: cat_acc.get(c, float("nan")) for c in CAT_ORDER}
    out["Total_micro"] = micro
    out["Total_macro"] = macro
    return out


def artifact2_table10(rows):
    """4 suites x 7 categories, Total column + Avg row."""
    cell, _ = group_mean(rows, lambda r: (r["suite"], r["category"]))
    suite_total, _ = group_mean(rows, lambda r: r["suite"])
    print("=== Artifact 2: per-suite x per-category (paper Table 10) ===")
    hdr = f"  {'':>8}" + "".join(f"{CAT_LABEL[c]:>11}" for c in CAT_ORDER) + f"{'Total':>11}"
    print(hdr)
    grid = {}
    for s in SUITE_ORDER:
        line = f"  {SUITE_LABEL[s]:>8}"
        for c in CAT_ORDER:
            v = cell.get((s, c), float("nan"))
            grid[(s, c)] = v
            line += f"{fmt(v):>11}"
        line += f"{fmt(suite_total.get(s, float('nan'))):>11}"
        print(line)
    avg_line = f"  {'Avg':>8}"
    avg_row = {}
    for c in CAT_ORDER:
        col = [grid[(s, c)] for s in SUITE_ORDER if grid.get((s, c), float("nan")) == grid.get((s, c))]
        avg_row[c] = mean(col)
        avg_line += f"{fmt(avg_row[c]):>11}"
    tcol = [suite_total[s] for s in SUITE_ORDER if s in suite_total]
    avg_line += f"{fmt(mean(tcol)):>11}"
    print(avg_line + "\n")
    return grid, suite_total, avg_row


def artifact3_figure5(rows):
    """Difficulty L1-L5 x category, nulls dropped."""
    rated = [r for r in rows if r["difficulty_level"] is not None]
    cell, _ = group_mean(rated, lambda r: (int(r["difficulty_level"]), r["category"]))
    levels = sorted({int(r["difficulty_level"]) for r in rated})
    print("=== Artifact 3: difficulty L1-L5 by category (paper Fig 5/8) ===")
    print(f"  ({len(rows) - len(rated)} null-difficulty rows dropped)")
    print(f"  {'level':>6}" + "".join(f"{CAT_LABEL[c]:>11}" for c in CAT_ORDER))
    grid = {}
    for lv in levels:
        line = f"  {'L'+str(lv):>6}"
        for c in CAT_ORDER:
            v = cell.get((lv, c), float("nan"))
            grid[(lv, c)] = v
            line += f"{fmt(v):>11}"
        print(line)
    print()
    return grid, levels


def _camera_sub(name):
    m = VIEW_RE.search(name)
    if not m:
        return "C?-unparsed"
    hr, vr, dis, chr_, cvr, _ = map(int, m.groups())
    flags = []
    if dis != 100:
        flags.append("C1 distance")
    if (hr, vr) != (0, 0):
        flags.append("C2 spherical")
    if (chr_, cvr) != (0, 0):
        flags.append("C3 orientation")
    if len(flags) == 1:
        return flags[0]
    return "C?-multi" if flags else "C?-default"


def _noise_sub(name):
    m = re.search(r"_noise_(\d+)$", name)
    if not m:
        return "N?-unparsed"
    n = int(m.group(1))
    return ["N1 motion", "N2 gaussian", "N3 zoom", "N4 fog", "N5 glass"][(n - 1) // 10]


def _layout_sub(name):
    if re.search(r"_add_\d+$", name):
        return "O1 confounding"
    if re.search(r"_level\d_sample\d+$", name):
        return "O2 target-pose"
    return "O?-other"


def _bg_sub(name):
    if re.search(r"_table_\d+$", name):
        return "B1 scene-theme"
    if re.search(r"_tb_\d+$", name):
        return "B2 surface"
    return "B?-other"


def artifact4_subperturbations(rows):
    """Recoverable sub-dimensions from the BDDL filename.

    Camera (C1/C2/C3), Sensor Noise (N1-N5), Objects Layout (O1/O2) and
    Background (B1/B2) decode cleanly from the name. Language (R1-R3) and
    Light (L1-L4) need the BDDL `language_instruction` / scene XML, so only
    their parent category is reported here.
    """
    specs = [
        ("Camera Viewpoints", _camera_sub),
        ("Sensor Noise", _noise_sub),
        ("Objects Layout", _layout_sub),
        ("Background Textures", _bg_sub),
    ]
    print("=== Artifact 4: sub-perturbation breakdowns (paper Appendix A) ===")
    out_rows = []
    for cat, fn in specs:
        sub = [r for r in rows if r["category"] == cat]
        if not sub:
            continue
        acc, cnt = group_mean(sub, lambda r: fn(r["task_name"]))
        print(f"  {CAT_LABEL[cat]}:")
        for k in sorted(acc):
            print(f"    {k:<16} acc={fmt(acc[k]).strip():>5}  n={cnt[k]}")
            out_rows.append({"category": CAT_LABEL[cat], "sub": k,
                             "accuracy_pct": round(100 * acc[k], 1), "n": cnt[k]})
    print("  (Language R1-R3 and Light L1-L4 need a BDDL-side table; parent "
          "category only.)\n")
    return out_rows


# --- sanity assertions --------------------------------------------------------
def run_strict_assertions(rows):
    names = [r["task_name"] for r in rows]
    assert len(rows) == EXPECT_TOTAL, f"expected {EXPECT_TOTAL} rollouts, got {len(rows)}"
    assert len(set(names)) == EXPECT_TOTAL, "task_name not unique"
    by_cat = Counter(r["category"] for r in rows)
    assert dict(by_cat) == EXPECT_BY_CATEGORY, f"category counts off: {dict(by_cat)}"
    by_suite = Counter(r["suite"] for r in rows)
    assert dict(by_suite) == EXPECT_BY_SUITE, f"suite counts off: {dict(by_suite)}"
    n_null = sum(1 for r in rows if r["difficulty_level"] is None)
    assert n_null == EXPECT_NULL_DIFFICULTY, f"null difficulty count {n_null}"
    n_null_lg = sum(1 for r in rows if r["difficulty_level"] is None
                    and r["category"] == "Light Conditions" and r["suite"] == "libero_goal")
    assert n_null_lg == EXPECT_NULL_DIFFICULTY, "nulls not all libero_goal x Light"
    n_add = sum(1 for n in names if re.search(r"_add_\d+$", n))
    n_lev = sum(1 for n in names if re.search(r"_level\d_sample\d+$", n))
    assert n_add == 829 and n_lev == 696 and n_add + n_lev == 1525, \
        f"layout subcount off: add={n_add} lev={n_lev}"
    print(">>> strict sanity assertions PASSED\n")


# --- CSV writers --------------------------------------------------------------
def write_csvs(out_dir: Path, merged, art1, t10, fig5, art4):
    out_dir.mkdir(parents=True, exist_ok=True)
    grid, suite_total, avg_row = t10
    fig5_grid, levels = fig5

    with (out_dir / "rollouts_tidy.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["suite", "task_idx", "task_name", "success",
                                          "steps", "wall_seconds", "category",
                                          "difficulty_level"])
        w.writeheader()
        w.writerows(merged)

    with (out_dir / "leaderboard_row.csv").open("w", newline="") as f:
        w = csv.writer(f)
        cols = [CAT_LABEL[c] for c in CAT_ORDER] + ["Total_micro", "Total_macro"]
        w.writerow(cols)
        w.writerow([round(100 * art1[c], 1) for c in cols])

    with (out_dir / "table10.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["suite"] + [CAT_LABEL[c] for c in CAT_ORDER] + ["Total"])
        for s in SUITE_ORDER:
            w.writerow([SUITE_LABEL[s]] + [round(100 * grid[(s, c)], 1) for c in CAT_ORDER]
                       + [round(100 * suite_total[s], 1)])
        tcol = mean([suite_total[s] for s in SUITE_ORDER if s in suite_total])
        w.writerow(["Avg"] + [round(100 * avg_row[c], 1) for c in CAT_ORDER]
                   + [round(100 * tcol, 1)])

    with (out_dir / "figure5.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["difficulty_level"] + [CAT_LABEL[c] for c in CAT_ORDER])
        for lv in levels:
            w.writerow([lv] + [round(100 * fig5_grid[(lv, c)], 1) for c in CAT_ORDER])

    with (out_dir / "subperturbations.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["category", "sub", "accuracy_pct", "n"])
        w.writeheader()
        w.writerows(art4)
    print(f">>> wrote 5 CSVs to {out_dir}/\n")


# --- self-test ----------------------------------------------------------------
def synthesize_rollouts(cls_rows):
    """Deterministic pseudo-random success per task -> full 10,030-row set."""
    out = []
    for r in cls_rows:
        h = abs(hash((r["suite"], r["name"]))) % 100
        out.append({"suite": r["suite"], "task_idx": r["id"] - 1,
                    "task_name": r["name"], "success": h < 70,  # ~70% base
                    "steps": 100, "wall_seconds": 20.0})
    return out


# --- main ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollouts", type=Path, default=Path("data/libero_plus/rollouts.jsonl"))
    ap.add_argument("--classification", type=Path,
                    default=Path("third_party/libero/libero/libero/benchmark/task_classification.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/libero_plus"))
    ap.add_argument("--strict", action="store_true",
                    help="enforce full-run sanity assertions (all 10,030 tasks)")
    ap.add_argument("--selftest", action="store_true",
                    help="synthesize a full rollout set and validate end-to-end")
    args = ap.parse_args()

    cls_rows = load_classification(args.classification)

    if args.selftest:
        print(f"[selftest] synthesizing {len(cls_rows)} rollouts from "
              f"{args.classification.name}\n")
        rollouts = synthesize_rollouts(cls_rows)
        args.strict = True
    else:
        if not args.rollouts.exists():
            sys.exit(f"no rollouts at {args.rollouts}\n"
                     f"  run the eval client first, or pass --selftest to validate "
                     f"the pipeline against synthetic data.")
        rollouts = load_rollouts(args.rollouts)

    merged = join(rollouts, cls_rows)
    print(f"joined {len(merged)} rollouts to classification "
          f"({100 * mean([r['success'] for r in merged]):.1f}% overall success)\n")

    if args.strict:
        run_strict_assertions(merged)

    art1 = artifact1_leaderboard(merged)
    t10 = artifact2_table10(merged)
    fig5 = artifact3_figure5(merged)
    art4 = artifact4_subperturbations(merged)

    if not args.selftest:
        write_csvs(args.out_dir, merged, art1, t10, fig5, art4)
    else:
        print(">>> selftest complete (CSVs not written)\n")


if __name__ == "__main__":
    main()
