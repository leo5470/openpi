# π₀.₅ × LIBERO-Plus robustness eval

Stress-tests the `pi05_libero` checkpoint against **LIBERO-Plus**'s 10,030
perturbed tasks (7 perturbation categories across 4 task suites), and aggregates
the results into the artifacts the LIBERO-Plus paper reports.

This is a thin eval harness layered on top of openpi. All openpi-side code lives
in this fork; **LIBERO-Plus is wired in as the `third_party/libero` submodule**
(repointed from upstream LIBERO to `leo5470/LIBERO-plus`). One-directional and
reproducible via `git submodule update --init --recursive`.

For the full rationale, verified dataset findings, and the source-of-truth spec,
see [`../../PI05_FORK_DEPLOYMENT_PLAN.md`](../../PI05_FORK_DEPLOYMENT_PLAN.md).

## Files

| File | Role |
|------|------|
| `main_libero_plus.py` | Rollout client. Fork of `main.py`: 1 trial/task, resumable JSONL log, `task.name`-based video names, `str(bddl)` Path fix for the Plus env. |
| `aggregate_libero_plus.py` | Stdlib-only aggregator → 4 paper artifacts + tidy CSVs. Has `--selftest` (no GPU) and `--strict` (full-run assertions). |
| `setup_libero_plus_env.sh` | Writes the non-destructive project-local LIBERO config. |
| `INSTALL.md` | Machine setup (server env, client env, assets, config). |

## Status

- ✅ Submodule repointed to LIBERO-Plus (`4976dc3`, 525-line Plus `env_wrapper`).
- ✅ Client, aggregator, and setup script committed.
- ✅ **Verification 1** (aggregator `--selftest`) passes against the real
  `task_classification.json`: 10,030 rows; per-category/-suite counts; 121 null
  difficulties; sub-counts Camera 313/992/294, Layout 829/696, Background 613/463.
- ⏳ **Next (needs a free GPU):** install per `INSTALL.md`, then run the eval below.

## Running the eval

> Two terminals. Install both environments first — see [`INSTALL.md`](INSTALL.md).

**Terminal 1 — serve the policy** (server env):

```bash
uv run scripts/serve_policy.py --env LIBERO   # config pi05_libero; checkpoint auto-downloads
```

**Terminal 2 — drive the clients** (client env):

```bash
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
export LIBERO_CONFIG_PATH=$PWD/examples/libero/libero_config
export MUJOCO_GL=egl          # use glx if you hit EGL errors

OUT=data/libero_plus
for s in libero_spatial libero_object libero_goal libero_10; do
  python examples/libero/main_libero_plus.py \
    --task-suite-name "$s" --num-trials-per-task 1 \
    --results-jsonl "$OUT/rollouts.jsonl" --video-out-path "$OUT/videos"
done
```

The JSONL log is line-buffered and appended, so a run is **resumable** — re-running
continues into the same file. Pass `--no-save-video` to skip mp4 writing on the
full 10k run.

**Aggregate:**

```bash
python examples/libero/aggregate_libero_plus.py \
  --rollouts data/libero_plus/rollouts.jsonl \
  --classification third_party/libero/libero/libero/benchmark/task_classification.json \
  --out-dir data/libero_plus --strict
```

Produces 4 artifacts (leaderboard row; per-suite×category Table 10; difficulty
L1–L5; sub-perturbation breakdowns) + tidy CSVs under `data/libero_plus/`.
`--strict` enforces the full-run counts; drop it for partial runs.

## Verification ladder

1. **Aggregator self-test (no GPU)** — already green; re-run anytime:
   ```bash
   python examples/libero/aggregate_libero_plus.py --selftest \
     --classification third_party/libero/libero/libero/benchmark/task_classification.json
   ```
2. **Config resolution (no GPU)** — `get_libero_path("bddl_files")` resolves into
   `third_party/libero/…`, not `~/LIBERO`. (See INSTALL.md step 5.)
3. **Smoke (1 task)** — one `libero_spatial` task: a video lands under
   `data/.../videos/libero_spatial/`, one JSONL line is written, **no** Path
   `TypeError` from `env_wrapper.py:207`, **no** missing-asset crash.
4. **Unperturbed ceiling** — run a handful of *original* (unperturbed) LIBERO
   tasks and confirm π₀.₅ lands near openpi's published **96.85%** original-LIBERO
   average. This isolates integration correctness (rotation / state / assets /
   checkpoint). Strong original-LIBERO performance does **not** predict
   robustness, so under/over-performing on a perturbation axis is a *finding*,
   not a bug.
5. **Full run** — all 10,030 tasks → `aggregate … --strict` passes.

## Notes / gotchas

- **Path→TypeError fix:** the Plus `ControlEnv` does `"_view_" in bddl_file_name`
  (`env_wrapper.py:207`); the client passes `str(bddl_file)` so a `pathlib.Path`
  never reaches it. Keep this.
- **Don't clobber `~/.libero`:** an existing `~/.libero/config.yaml` (e.g. a
  vanilla install) would otherwise silently win; we override via
  `LIBERO_CONFIG_PATH`.
- **Assets are mandatory:** perturbed scenes load textures/objects from
  `assets.zip`; an empty `assets/` fails at env construction.
- **Checkpoints are never committed** — fetched from `gs://openpi-assets`.
