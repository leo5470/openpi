# Installing the LIBERO-Plus eval (π₀.₅ robustness)

Setup for a **fresh GPU machine**. Produces two Python environments, following
openpi's standard LIBERO pattern:

| Env | Python | Purpose | Built with |
|-----|--------|---------|-----------|
| **server** | repo default (3.11) | runs `scripts/serve_policy.py` (JAX, the policy) | `uv sync` |
| **client** | 3.8 + cu113 | runs `examples/libero/main_libero_plus.py` (LIBERO-Plus sim) | `examples/libero/.venv` |

They are kept separate on purpose: LIBERO pins old deps (numpy 1.22, torch
cu113, gym 0.25) that conflict with the server's JAX stack. The two talk over a
websocket, so they can even live on different hosts.

> All paths below are relative to the **fork root** (the openpi checkout).

---

## 0. Prerequisites

- NVIDIA GPU with **fully free memory** (the policy + MuJoCo EGL rendering want
  headroom; a 24 GB card is plenty when not shared).
- Recent NVIDIA driver + CUDA userspace.
- [`uv`](https://docs.astral.sh/uv/) and `git-lfs`.
- **System ImageMagick** — required by the LIBERO-Plus *Sensor Noise* axis (see
  step 2, the `Wand` dependency):

  ```bash
  sudo apt-get update && sudo apt-get install -y libmagickwand-dev imagemagick
  ```

## 1. Clone the fork with the LIBERO-Plus submodule

```bash
git clone --recurse-submodules https://github.com/leo5470/openpi.git
cd openpi
# If already cloned without --recurse-submodules:
git submodule update --init --recursive
```

`third_party/libero` is repointed to `leo5470/LIBERO-plus` (pinned at `4976dc3`).
Sanity-check that you got **Plus**, not vanilla LIBERO:

```bash
wc -l third_party/libero/libero/libero/envs/env_wrapper.py   # expect ~525 (vanilla is 278)
```

## 2. Server environment (the policy)

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync          # GIT_LFS_SKIP_SMUDGE pulls LeRobot as a dep
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

The `pi05_libero` checkpoint is **not** downloaded here — `serve_policy.py`
fetches `gs://openpi-assets/checkpoints/pi05_libero` on first launch (cached
under `~/.cache/openpi`). Never commit it.

## 3. Client environment (LIBERO-Plus sim)

```bash
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate

uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero

# LIBERO-Plus extras NOT in third_party/libero/requirements.txt: env_wrapper.py
# imports these at module top level for the Sensor Noise perturbations
# (motion/gaussian/zoom/fog/glass). Without them, importing the env fails.
uv pip install Wand scikit-image scipy

export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
```

Verify the client env can import the Plus env (this is what catches a missing
`Wand`/ImageMagick/`skimage`):

```bash
python -c "from libero.libero.envs import OffScreenRenderEnv; print('libero env import OK')"
```

## 4. Download the LIBERO-Plus assets (~6.4 GB)

The submodule ships `assets/` empty (gitignored). Perturbed BDDLs reference
textures/objects/scenes packed in `assets.zip`:

```bash
# either the HF CLI:
uv pip install -U "huggingface_hub[cli]"
huggingface-cli download Sylvest/LIBERO-plus assets.zip --repo-type dataset --local-dir /tmp/libero_plus_dl
# or a direct download:
#   wget -O /tmp/libero_plus_dl/assets.zip \
#     "https://huggingface.co/datasets/Sylvest/LIBERO-plus/resolve/main/assets.zip"

unzip -q /tmp/libero_plus_dl/assets.zip -d third_party/libero/libero/libero/assets/
# Verify it's populated (and not nested one level too deep):
ls third_party/libero/libero/libero/assets/ | head
```

If the zip unpacked into a nested `assets/assets/…`, move its contents up one
level so files sit directly under `third_party/libero/libero/libero/assets/`.

## 5. Project-local LIBERO config (non-destructive)

```bash
bash examples/libero/setup_libero_plus_env.sh
```

This writes `examples/libero/libero_config/config.yaml` pointing into the
submodule and prints the `export LIBERO_CONFIG_PATH=…` line. It **does not**
touch `~/.libero/config.yaml` (which may point at a different LIBERO install).
The generated `config.yaml` holds absolute machine paths and is gitignored.

Confirm resolution points into the submodule, not `~/LIBERO` (**Verification 2**):

```bash
export LIBERO_CONFIG_PATH=$PWD/examples/libero/libero_config
python -c "from libero.libero import get_libero_path; print(get_libero_path('bddl_files'))"
# -> .../third_party/libero/libero/libero/bddl_files
```

---

## Done

You now have: a server env, a client env that imports the Plus sim, the assets,
and a config that resolves into the submodule. Continue with **running the eval**
in `README_libero_plus.md`.
