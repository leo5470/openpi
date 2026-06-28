# Installing the LIBERO-Plus eval (π₀.₅ robustness)

Setup for a **fresh GPU machine**. Produces two Python environments, following
openpi's standard LIBERO pattern:

| Env | Python | Purpose | Built with |
|-----|--------|---------|-----------|
| **server** | repo default (3.11) | runs `scripts/serve_policy.py` (JAX, the policy) | `uv sync` |
| **client** | 3.8 + cu113 | runs `examples/libero/main_libero_plus.py` (LIBERO-Plus sim) | `uv venv` **or conda** (see §3) |

They are kept separate on purpose: LIBERO pins old deps (numpy 1.22, torch
cu113, gym 0.25) that conflict with the server's JAX stack. The two talk over a
websocket, so they can even live on different hosts.

> **No sudo on the box?** Build the *client* env with **conda**, not `uv` — see
> §3's alternative. LIBERO-Plus needs native C libraries (ImageMagick,
> fontconfig, expat) that `uv`/`pip` cannot install but conda-forge can, into a
> user-writable prefix with no root. The server env stays on `uv` either way.

> All paths below are relative to the **fork root** (the openpi checkout).

---

## 0. Prerequisites

- NVIDIA GPU with **fully free memory** (the policy + MuJoCo EGL rendering want
  headroom; a 24 GB card is plenty when not shared).
- Recent NVIDIA driver + CUDA userspace.
- [`uv`](https://docs.astral.sh/uv/) and `git-lfs`. (`conda` too if you take the
  no-sudo client path in §3.)
- **Native C libraries for the client** — LIBERO-Plus's `env_wrapper.py` binds
  ImageMagick (via `Wand`) at import time, and the MuJoCo/robosuite stack needs
  fontconfig + expat. `uv`/`pip` cannot provide these; get them one of two ways:

  ```bash
  # (a) with sudo — system packages:
  sudo apt-get update && sudo apt-get install -y \
    libmagickwand-dev imagemagick libfontconfig1-dev libexpat1
  ```

  ```bash
  # (b) no sudo — conda-forge ships them; they're declared in the client env's
  #     environment.yml, so just create it in §3 Option B (nothing to do here).
  ```

  `libpython3-stdlib` is only relevant to the *system* python; conda/uv pythons
  ship their own stdlib, so you can ignore it.

## 1. Clone the fork with the LIBERO-Plus submodule

```bash
git clone https://github.com/leo5470/openpi.git
cd openpi
git checkout pi05-libero-plus-eval   # do this BEFORE touching submodules:
                                     # main's .gitmodules still points at vanilla LIBERO
git submodule sync third_party/libero
git submodule update --init --recursive
```

> The repoint to LIBERO-Plus lives on the `pi05-libero-plus-eval` branch. Once
> it's merged to `main` you can clone with `--recurse-submodules` directly.

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

Pick **one** of the two options. Both end with the same Python deps; they differ
only in how the env and the native C libraries are provided.

### Option A — `uv` (you have sudo for the §0 system libs)

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
# (scipy is already pulled in transitively by robosuite, so it's not listed here.)
uv pip install Wand scikit-image

export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
```

### Option B — conda (no sudo: conda-forge supplies the C libraries)

Two steps: `conda env create` builds python 3.8 + the native libs from
`environment.yml`; then pip/uv installs the Python wheels into that env. (The
wheels are kept out of `environment.yml` because conda feeds its `pip:` section
through a temp file, which breaks the `-r examples/libero/requirements.txt`
relative paths.)

```bash
# 1) env + native C libs (declarative). Run from the fork root:
conda env create -f examples/libero/environment.yml
conda activate libero_plus

# 2) Python wheels into the conda env -- same set as Option A. uv pip targets the
#    conda env via --python and *install* (not sync) leaves conda's packages alone:
PY="$CONDA_PREFIX/bin/python"
uv pip install --python "$PY" \
  -r examples/libero/requirements.txt -r third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install --python "$PY" -e packages/openpi-client -e third_party/libero
uv pip install --python "$PY" Wand scikit-image   # scipy is already pinned in requirements.txt

export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
export MAGICK_HOME="$CONDA_PREFIX"   # so Wand loads the conda ImageMagick, not a system one
```

> Plain `pip` works too (`pip install -r ... --extra-index-url ...`), but `uv pip`
> is used here for the same cross-index resolution as Option A.
>
> **Pin skew to watch:** openpi's lockfile pins `robosuite==1.4.1` while
> LIBERO-Plus's `requirements.txt` pins `1.4.0` (and adds robomimic/transformers/
> bddl). `--index-strategy=unsafe-best-match` lets the resolver pick a consistent
> set; if it still objects to the conflicting pins, install
> `examples/libero/requirements.txt` first, then `uv pip install --python "$PY" -e
> third_party/libero` and let its `setup.py` resolve the rest.
>
> **`usd-core` (~300 MB):** LIBERO-Plus's `requirements.txt` pins `usd-core>=25.5`.
> `usd-core 25.8` ships a `cp38` manylinux wheel so it installs fine on Python 3.8
> Linux — just expect a slow download on first install; nothing to configure.
>
> If `import wand` still can't find ImageMagick, also
> `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"`.

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
