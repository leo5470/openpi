# π₀.₅ LIBERO-Plus eval — fork-of-openpi deployment plan

> Execute this **inside a fork of openpi** (the "new workspace"). LIBERO-plus is
> wired in as openpi's `third_party/libero` submodule. Supersedes the in-place
> approach in `PI05_DEPLOYMENT_PLAN.md`.
>
> This doc is self-contained: Appendices A–C carry the full, verified source for
> the rollout client, the results aggregator, and the env-setup script, so the
> new repo can be built from scratch without referencing any prior workspace.

## Context

Goal: stress-test the `pi05_libero` checkpoint against LIBERO-Plus's 10,030
perturbed tasks. The earlier plan edited a throwaway openpi clone in place, which
puts openpi-shaped code (a rollout client + a `third_party/libero` swap) inside
the *benchmark* repo — neither reproducible nor clean.

New architecture: **all openpi-side changes live in a fork of openpi**, and
**LIBERO-plus is wired in as openpi's `third_party/libero` submodule**, repointed
from upstream LIBERO to `github.com/leo5470/LIBERO-plus`. One-directional and
cycle-free: the fork nests LIBERO-plus; LIBERO-plus knows nothing about openpi.
Reproducible via `git submodule update --init --recursive`.

Assumed workspace: the openpi fork (`origin = leo5470/openpi`,
`upstream = Physical-Intelligence/openpi`, pinned near upstream `c23745b`), with
LIBERO-plus reachable as `third_party/libero`.

### Verified findings that shape the plan
- **`task_classification.json` is exact**: 10,030 rows (spatial 2402 / object 2518
  / goal 2591 / 10 2519); 7 categories (Camera 1599, Robot 1550, Language 1537,
  Light 1142, Background 1076, Noise 1601, Layout 1525); 121 null difficulty (all
  `libero_goal` × Light). Aggregator joins by `(suite, task_name)`. Sub-counts
  verified: Camera C1/C2/C3 = 313/992/294 (one axis varies at a time);
  Noise N maps 1–50 → N1–N5; Layout O1 `_add_`=829 / O2 `_level…_sample`=696;
  Background B1 `_table_`=613 / B2 `_tb_`=463.
- **Path→TypeError bug (keep the fix)**: LIBERO-Plus's `ControlEnv` does
  `"_view_" in bddl_file_name` at `libero/libero/envs/env_wrapper.py:207`. openpi's
  upstream `main.py` passes a `pathlib.Path`, which raises `TypeError`. The client
  passes `str(task_bddl_file)` in `_get_libero_env` (Appendix A). Vanilla LIBERO
  never did string ops on the name, so this only bites with Plus.
- **Perturbed BDDLs are not all materialized as files**: the env strips
  `_view_…_initstate_` to load the base XML and applies the perturbation from the
  parsed codes; `task.bddl_file` carries the perturbed name. So env *construction*
  carries over from upstream `main.py` unchanged (besides the `str()` fix).
- **Config gotcha (non-destructive override required)**: `~/.libero/config.yaml`
  may already exist pointing at a *different* install (e.g. vanilla
  `~/LIBERO`). Because the file exists, installing the Plus package will NOT
  rewrite it, so the client would silently load vanilla BDDLs/assets. Fix:
  generate a project-local `config.yaml` and select it via `LIBERO_CONFIG_PATH`
  (Appendix C); do **not** clobber `~/.libero`. Confirmed: with the override,
  `get_libero_path("bddl_files")` resolves into the LIBERO-plus tree.
- **Assets must be downloaded**: `libero/libero/assets/` ships empty (gitignored).
  Perturbation BDDLs reference textures/objects/scenes in `assets.zip`.
- **Server**: `scripts/serve_policy.py --env LIBERO` → config `pi05_libero`,
  checkpoint `gs://openpi-assets/checkpoints/pi05_libero` (auto-downloaded at
  runtime; **never committed**).
- **Aggregator is dependency-free (stdlib only)** — pandas need not be installed.
  Its `--selftest` (synthesizes a full 10,030-row set) passes all strict
  assertions; it was validated before this doc was written.

## Steps (inside the openpi fork)

### 1. Repoint the libero submodule to LIBERO-plus
- Edit `.gitmodules`: change the `third_party/libero` url from
  `…/Lifelong-Robot-Learning/LIBERO.git` to
  `https://github.com/leo5470/LIBERO-plus.git`.
- `git submodule sync third_party/libero` →
  `git submodule update --init --recursive third_party/libero`, check out the
  intended LIBERO-plus commit, then commit the `.gitmodules` + gitlink.
- Sanity: `third_party/libero/libero/libero/envs/env_wrapper.py` is the **525-line
  Plus** version (upstream LIBERO is 278).
- (Quick local-dev alternative, uncommitted: relative symlink
  `third_party/libero → ../../LIBERO-plus` when both are sibling clones.)

### 2. Add the rollout client
Create `examples/libero/main_libero_plus.py` from **Appendix A** and commit.
Deltas from upstream `examples/libero/main.py`: `num_trials_per_task=1`; per-rollout
JSONL log (`{suite, task_idx, task_name, success, steps, wall_seconds,
video_path}`, line-buffered/resumable); video named by `task.name` (perturbations
share a language string, so the upstream language-based filename collides);
**`str(bddl_file)`** in `_get_libero_env`; `save_video` toggle; `done=False`
guard. Unchanged: 180° rotation, `resize_with_pad` to 224, 8-dim
eef+axisangle+gripper state, replan-5-of-≥5, per-suite `max_steps`
(220/280/300/520).

### 3. Generate the non-destructive LIBERO config
Create `examples/libero/setup_libero_plus_env.sh` from **Appendix C** and run it.
It writes a project-local `examples/libero/libero_config/config.yaml` pointing
into `third_party/libero/libero/libero/…` and prints the
`export LIBERO_CONFIG_PATH=…` to use. Do not touch `~/.libero`. Then
`pip install -e third_party/libero` in the client venv.

### 4. Download assets
Fetch `assets.zip` from `huggingface.co/datasets/Sylvest/LIBERO-plus` and unzip
into `third_party/libero/libero/libero/assets/`. The setup script warns when empty.

### 5. Serve the policy (GPU box, server venv)
```bash
uv run scripts/serve_policy.py --env LIBERO   # config pi05_libero, gs checkpoint
```

### 6. Drive the clients (client venv)
```bash
export LIBERO_CONFIG_PATH=$PWD/examples/libero/libero_config
export MUJOCO_GL=egl
OUT=data/libero_plus
for s in libero_spatial libero_object libero_goal libero_10; do
  python examples/libero/main_libero_plus.py \
    --task-suite-name "$s" --num-trials-per-task 1 \
    --results-jsonl "$OUT/rollouts.jsonl" --video-out-path "$OUT/videos"
done
```

### 7. Aggregate
Create `examples/libero/aggregate_libero_plus.py` from **Appendix B**, then:
```bash
python examples/libero/aggregate_libero_plus.py \
  --rollouts data/libero_plus/rollouts.jsonl \
  --classification third_party/libero/libero/libero/benchmark/task_classification.json \
  --out-dir data/libero_plus --strict
```
Produces the four artifacts (leaderboard row; per-suite×category Table 10;
difficulty L1–L5; sub-perturbation breakdowns — Camera C1/C2/C3, Noise N1–N5,
Layout O1/O2, Background B1/B2) + tidy CSVs. `--strict` enforces full-run counts;
`--selftest` validates the pipeline with no rollout data.

## Files (in the fork)
- `.gitmodules` — repoint `third_party/libero` url (committed).
- `examples/libero/main_libero_plus.py` — Appendix A (committed).
- `examples/libero/aggregate_libero_plus.py` — Appendix B (committed).
- `examples/libero/setup_libero_plus_env.sh` — Appendix C (committed).
- `.gitignore` — ignore `examples/libero/libero_config/config.yaml` (machine paths)
  and `data/`. Checkpoints are never local.

## Verification
1. **Aggregator self-test (no GPU):** `python examples/libero/aggregate_libero_plus.py
   --selftest --classification third_party/libero/libero/libero/benchmark/task_classification.json`
   → must print "strict sanity assertions PASSED".
2. **Config resolution (no GPU):** with `LIBERO_CONFIG_PATH` set,
   `python -c "from libero.libero import get_libero_path; print(get_libero_path('bddl_files'))"`
   resolves into `third_party/libero/…`, not `~/LIBERO`.
3. **Smoke (1 task):** one `libero_spatial` task, `num_trials_per_task=1` — video
   lands under `data/.../videos/libero_spatial/`, one JSONL line written, no Path
   `TypeError` from `env_wrapper.py:207`, no missing-asset crash.
4. **Unperturbed ceiling:** run a handful of *original* (unperturbed) LIBERO tasks
   and confirm π₀.₅ lands near openpi's published ~96.85% original-LIBERO average.
   This isolates "is the integration correct?" (rotation/state/assets/checkpoint).
   Anchor the sanity check to that published unperturbed number — do **not** treat
   π₀.₅'s relative ranking vs π₀ on any Plus axis as a pass/fail bar; strong
   original-LIBERO performance does not predict robustness, so under/over-
   performing on a perturbation axis is a finding, not a bug.
5. **Full run:** all 10,030 tasks → `aggregate … --strict` passes (len 10,030,
   per-category and per-suite counts, 121 null difficulty, layout sub-counts
   829/696).

---

## Appendix A — `examples/libero/main_libero_plus.py`

```python
"""LIBERO-Plus rollout client for openpi-served policies (e.g. pi05_libero).

Fork of openpi/examples/libero/main.py adapted per the deployment plan §2.
Deltas from the upstream file (kept intentionally small):

  1. num_trials_per_task defaults to 1 -- LIBERO-Plus encodes the variation in
     the BDDL itself, so each of the thousands of tasks per suite gets exactly
     one rollout, not 50.
  2. Per-rollout JSONL logging ({suite, task_idx, task_name, success, steps,
     wall_seconds, video_path}) flushed after every episode, so a run is
     resumable/aggregatable and feeds aggregate_libero_plus.py directly.
  3. Videos are named by task.name (the perturbed BDDL stem), NOT the language
     instruction -- many Plus perturbations of one task share a language
     string, so the upstream language-based filename would clobber videos.
  4. bddl_file_name is passed to OffScreenRenderEnv as a *str*, not a Path:
     LIBERO-Plus's ControlEnv does `"_view_" in bddl_file_name` (env_wrapper.py
     L207) which raises TypeError on a pathlib.Path. Vanilla LIBERO never did
     string ops on the name, so this only matters for the Plus env.

Everything else (180-deg image rotation, resize_with_pad to 224, the 8-dim
eef-pos + axisangle + gripper-qpos state, replan-every-5 of a >=5 chunk,
per-suite max_steps) is unchanged -- it matches how pi05_libero was trained.
"""

import collections
import dataclasses
import json
import logging
import math
import pathlib
import time

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    # Model server
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    # LIBERO-Plus environment
    task_suite_name: str = "libero_spatial"  # libero_spatial | libero_object | libero_goal | libero_10
    num_steps_wait: int = 10  # steps to let objects settle in sim before acting
    num_trials_per_task: int = 1  # Plus: one rollout per (already-perturbed) task

    # Utils
    video_out_path: str = "data/libero_plus/videos"  # per-suite subdir is appended
    results_jsonl: str = "data/libero_plus/rollouts.jsonl"  # consumed by aggregate_libero_plus.py
    save_video: bool = True  # set False to skip mp4 writing on a full 10k run

    seed: int = 7  # affects object positions even with a fixed init state


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name} ({num_tasks_in_suite} tasks)")

    video_dir = pathlib.Path(args.video_out_path) / args.task_suite_name
    video_dir.mkdir(parents=True, exist_ok=True)
    results_path = pathlib.Path(args.results_jsonl)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    results_file = results_path.open("a", buffering=1)  # line-buffered: resumable mid-run
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        for episode_idx in range(args.num_trials_per_task):
            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            done = False  # guard: referenced below even if the first step raises
            episode_steps = 0
            replay_images = []
            wall_start = time.time()

            while t < max_steps + args.num_steps_wait:
                try:
                    # Do nothing for the first few steps: the sim drops objects
                    # and we wait for them to settle.
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # rotate 180 deg to match training preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )
                    if args.save_video:
                        replay_images.append(img)

                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }
                        action_chunk = client.infer(element)["actions"]
                        assert len(action_chunk) >= args.replan_steps, (
                            f"want to replan every {args.replan_steps} steps, but policy "
                            f"only predicts {len(action_chunk)} steps."
                        )
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(action.tolist())
                    episode_steps += 1
                    if done:
                        total_successes += 1
                        break
                    t += 1
                except Exception as e:  # noqa: BLE001 - one bad task must not kill the suite
                    logging.error(f"task {task_id} ({task.name}) raised: {e}")
                    break

            total_episodes += 1
            wall_seconds = round(time.time() - wall_start, 2)

            video_path = ""
            if args.save_video and replay_images:
                suffix = "success" if done else "failure"
                video_path = str(video_dir / f"{task.name}_{suffix}.mp4")
                imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=10)

            # One JSONL line per rollout -> aggregate_libero_plus.py join key is task_name.
            results_file.write(json.dumps({
                "suite": args.task_suite_name,
                "task_idx": task_id,
                "task_name": task.name,
                "success": bool(done),
                "steps": episode_steps,
                "wall_seconds": wall_seconds,
                "video_path": video_path,
            }) + "\n")

        if (task_id + 1) % 50 == 0 or task_id + 1 == num_tasks_in_suite:
            logging.info(
                f"[{args.task_suite_name}] {task_id + 1}/{num_tasks_in_suite} tasks, "
                f"running success rate {total_successes / max(total_episodes, 1) * 100:.1f}%"
            )

    results_file.close()
    logging.info(
        f"Done {args.task_suite_name}: {total_successes}/{total_episodes} = "
        f"{total_successes / max(total_episodes, 1) * 100:.1f}% -> {results_path}"
    )


def _get_libero_env(task, resolution, seed):
    """Initialize the LIBERO-Plus env and return (env, language)."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    # str(): Plus's ControlEnv parses perturbation codes via `"_view_" in name`,
    # which raises TypeError on a Path (env_wrapper.py L207).
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # seed affects object positions even with a fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """Copied from robosuite transform_utils."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
```

## Appendix B — `examples/libero/aggregate_libero_plus.py`

```python
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
```

## Appendix C — `examples/libero/setup_libero_plus_env.sh`

```bash
#!/usr/bin/env bash
# Run from anywhere inside the openpi fork. Writes a project-local LIBERO config
# pointing at the third_party/libero (= LIBERO-plus) submodule, so rollouts load
# the Plus BDDLs/assets. Non-destructive: does NOT touch ~/.libero/config.yaml
# (which may point at a different LIBERO install). Idempotent.
set -euo pipefail

FORK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # examples/libero/../.. = fork root
PLUS="$FORK_ROOT/third_party/libero/libero/libero"
CONFIG_DIR="$FORK_ROOT/examples/libero/libero_config"

# the submodule must be initialized AND be the Plus repo (525-line env_wrapper)
if [ ! -f "$PLUS/envs/env_wrapper.py" ]; then
  echo "ERROR: $PLUS/envs/env_wrapper.py missing." >&2
  echo "  Repoint third_party/libero to leo5470/LIBERO-plus and run:" >&2
  echo "    git submodule sync third_party/libero" >&2
  echo "    git submodule update --init --recursive third_party/libero" >&2
  exit 1
fi
LINES=$(wc -l < "$PLUS/envs/env_wrapper.py")
[ "$LINES" -lt 400 ] && echo "WARNING: env_wrapper.py is $LINES lines (Plus is ~525; vanilla is 278) -- submodule may still be upstream LIBERO." >&2

mkdir -p "$CONFIG_DIR"
cat > "$CONFIG_DIR/config.yaml" <<YAML
assets: $PLUS/assets
bddl_files: $PLUS/bddl_files
benchmark_root: $PLUS
datasets: $FORK_ROOT/third_party/libero/libero/datasets
init_states: $PLUS/init_files
YAML
echo "wrote $CONFIG_DIR/config.yaml"

if [ -z "$(ls -A "$PLUS/assets" 2>/dev/null)" ]; then
  echo "WARNING: $PLUS/assets is EMPTY. Download assets.zip from" >&2
  echo "         https://huggingface.co/datasets/Sylvest/LIBERO-plus and unzip here" >&2
  echo "         before running rollouts, or perturbed scenes will fail to load." >&2
fi

echo
echo "Setup done. Before running the client, export:"
echo "    export LIBERO_CONFIG_PATH=$CONFIG_DIR"
```
```
.gitignore additions (in the fork):
  examples/libero/libero_config/config.yaml
  data/
```
