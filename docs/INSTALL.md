# Installation

Tested with **Python 3.11**, **PyTorch 2.5.1**, and **CUDA 12.4** on Linux.

```bash
conda create -n dvlt python=3.11 && conda activate dvlt
conda install pytorch=2.5.1 torchvision pytorch-cuda=12.4 -c pytorch -c nvidia -c conda-forge

git clone <YOUR-DVLT-REMOTE>.git dvlt && cd dvlt
pip install -e .[all]
```

Extras (pick what you need): `pip install -e .[demos|dev|all]`

- `demos` — Gradio demos (`gradio`, `onnxruntime-gpu`, `trimesh`)
- `dev`   — `black`, `ruff`, `pre-commit`, `pytest`
- `all`   — both

## Baselines (optional)

The baseline wrappers in `src/dvlt/model/{vggt,vggt_omega,da3,mapanything,pi3}/`
import the upstream packages at runtime. Install only the ones you plan to
evaluate:

```bash
# VGGT — code: commercial-use-friendly; checkpoints: original
# is non-commercial, VGGT-1B-Commercial allows commercial use. See upstream LICENSE.
pip install git+https://github.com/facebookresearch/vggt.git --no-deps

# VGGT-Omega — FAIR Noncommercial Research License (non-commercial / research-only;
# applies to both code and checkpoints). See upstream LICENSE.
pip install git+https://github.com/facebookresearch/vggt-omega.git --no-deps

# Depth-Anything-3 — Apache-2.0
pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git --no-deps

# MapAnything — Apache-2.0
pip install git+https://github.com/facebookresearch/map-anything.git --no-deps
pip install uniception==0.1.6

# Pi3 / Pi3X — BSD-3-Clause (clone into third_party/Pi3 first)
git clone https://github.com/<pi3-upstream>/Pi3.git third_party/Pi3
pip install -e third_party/Pi3 --no-deps
```

See [THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md) §"Upstream packages
used for evaluation" for license details — verify each upstream's terms suit
your use case before installing.

## Data backend (`dataverse`)

DVLT's dataset parsers depend on the `dataverse` package.

```bash
git clone <YOUR-DATAVERSE-REMOTE>.git third_party/dataverse
pip install -e third_party/dataverse
```

Some datasets need extra dependencies — install the matching extras as needed,
for example:

```bash
pip install -e 'third_party/dataverse[kubric]'
```

The Gradio demos and `dvlt.scripts.visualize` work without `dataverse`; the
full training stack does not.

## Sanity check

```bash
python -c "import dvlt; print('dvlt OK')"
pytest -q tests/data/datasets/
```
