# Dataset Preparation

All dataset paths are resolved from a single config variable, `user.data_root`.
It defaults to `<project root>/datasets`, and can be overridden with the CLI
flag `user.data_root=/path/to/data`. The per-dataset configs under
[`src/dvlt/config/experiments/data/`](../../src/dvlt/config/experiments/data/)
read from `${user.data_root}/{train,test}/<dataset>`.

Supported datasets and per-dataset preparation:

| Dataset | Use | Preparation | Config |
|---|---|---|---|
| DTU | eval | download only (Spann3R mvsnet, see below) | [`dtu.yaml`](../../src/dvlt/config/experiments/data/test_datasets/dtu.yaml) |
| ETH3D | eval | [`preprocess_eth3d.md`](../../src/dvlt/scripts/preprocess/preprocess_eth3d.md) | [`eth3d.yaml`](../../src/dvlt/config/experiments/data/test_datasets/eth3d.yaml) |
| 7Scenes | eval | [`preprocess_7scenes.md`](../../src/dvlt/scripts/preprocess/preprocess_7scenes.md) | [`sevenscenes.yaml`](../../src/dvlt/config/experiments/data/test_datasets/sevenscenes.yaml) |
| NuScenes | eval | [`preprocess_nuscenes.md`](../../src/dvlt/scripts/preprocess/preprocess_nuscenes.md) | [`nuscenes.yaml`](../../src/dvlt/config/experiments/data/test_datasets/nuscenes.yaml) |
| ScanNet++ | train + eval | [`preprocess_scannetpp.md`](../../src/dvlt/scripts/preprocess/preprocess_scannetpp.md) | [`scannetpp.yaml`](../../src/dvlt/config/experiments/data/test_datasets/scannetpp.yaml) |

Preprocessing scripts live under
[`src/dvlt/scripts/preprocess/`](../../src/dvlt/scripts/preprocess/) and are run
as modules, e.g. `python -m dvlt.scripts.preprocess.<dataset>.<script>`. See the
per-dataset docs above for download + preprocessing details.

---

## DTU (evaluation, download only)

Follow Spann3R for DTU download (mvsnet preprocessed):
<https://github.com/HengyiWang/spann3r/blob/main/docs/data_preprocess.md#dtu>

Expected layout, one directory per scan, placed under `${user.data_root}/test/dtu`:

```text
test/dtu/
└── scanN/
    ├── images/{00000000.jpg, 00000001.jpg, ...}
    ├── cams/{00000000_cam.txt, ...}        # MVSnet camera format (world-to-camera)
    ├── depths/{00000000.npy, ...}          # depth in millimetres
    └── binary_masks/{00000000.png, ...}    # foreground masks
```

Consumed by [`test_datasets/dtu.yaml`](../../src/dvlt/config/experiments/data/test_datasets/dtu.yaml)
(`DTU`).

---

## ETH3D (evaluation)

High-resolution multi-view benchmark. Download the undistorted DSLR images +
calibration and the GT depth, then undistort the distorted GT depth onto the
pinhole cameras. Lands under `${user.data_root}/test/eth3d`.

See [`preprocess_eth3d.md`](../../src/dvlt/scripts/preprocess/preprocess_eth3d.md)
for download + preprocessing. Consumed by
[`test_datasets/eth3d.yaml`](../../src/dvlt/config/experiments/data/test_datasets/eth3d.yaml).

---

## 7Scenes (evaluation)

Microsoft 7-Scenes RGB-D (640×480). Download the seven scenes and generate the
projected pseudo-GT depth (via SimpleRecon). Lands under
`${user.data_root}/test/sevenscenes`. Intrinsics are fixed (focal 525, principal
point at image centre).

See [`preprocess_7scenes.md`](../../src/dvlt/scripts/preprocess/preprocess_7scenes.md)
for download + preprocessing. Consumed by
[`test_datasets/sevenscenes.yaml`](../../src/dvlt/config/experiments/data/test_datasets/sevenscenes.yaml).

---

## NuScenes (evaluation)

Autonomous-driving dataset. Download **v1.0-trainval** (contains the 50 eval
scenes) and convert to one `scene.h5` per scene (CAM_FRONT images, sparse LiDAR
depth, camera extrinsics + intrinsics). Lands under
`${user.data_root}/test/nuscenes/trainval/<scene>/scene.h5`.

See [`preprocess_nuscenes.md`](../../src/dvlt/scripts/preprocess/preprocess_nuscenes.md)
for download + preprocessing. Consumed by
[`test_datasets/nuscenes.yaml`](../../src/dvlt/config/experiments/data/test_datasets/nuscenes.yaml).

---

## ScanNet++ (training + evaluation)

Indoor DSLR captures. Render per-frame metric depth from the aligned GT mesh and
rectify the fisheye images/depth to a pinhole model. The train split
(`nvs_sem_train`) lands under `${user.data_root}/train/scannetpp`, the eval split
(`nvs_sem_val`) under `${user.data_root}/test/scannetpp`.

See [`preprocess_scannetpp.md`](../../src/dvlt/scripts/preprocess/preprocess_scannetpp.md)
for download + preprocessing. Consumed by
[`train_datasets/scannetpp.yaml`](../../src/dvlt/config/experiments/data/train_datasets/scannetpp.yaml)
and
[`test_datasets/scannetpp.yaml`](../../src/dvlt/config/experiments/data/test_datasets/scannetpp.yaml).

---

## Training datasets

Only **ScanNet++** is included as a runnable training dataset in this release.
The other training-dataset configs under
[`train_datasets/`](../../src/dvlt/config/experiments/data/train_datasets/) are
kept so that the per-dataset frame/view-sampling configuration stays visible
(`view_ranking`, `view_sampling`, `window_size_ratio`, `max_depth_thresh`,
etc.), but their loaders and preprocess scripts are WIP.

---

## Data pipeline (overview)

Parsers convert dataset-specific formats into a unified representation in three
stages:

1. **Parse + per-frame preprocess** - `TrainDataset.get_data()` (training) or
   `EvalDataset.__getitem__()` (eval): sample frames, read raw data, then
   resize/crop each frame and update intrinsics/depth/world-points accordingly.
   Output: per-frame numpy arrays.
2. **Sequence-level processing** - `MultiSourceDataset._build_sample()`: color
   augmentations (training), scene normalization (centre + scale factor), and
   conversion to PyTorch tensors.
3. **Batch assembly** - `default_collate_fn()`: stack samples into a batch with
   field-specific padding.

The batched sample the model consumes (keys are `DataField`s):

```python
{
    DataField.IMAGES:        torch.Tensor,  # [B, S, C, H, W], float32, [0,1]
    DataField.DEPTHS:        torch.Tensor,  # [B, S, H, W],    float32
    DataField.EXTRINSICS_C2W:torch.Tensor,  # [B, S, 4, 4],    float32 (normalized)
    DataField.INTRINSICS:    torch.Tensor,  # [B, S, 3, 3],    float32
    DataField.WORLD_POINTS:  torch.Tensor,  # [B, S, H, W, 3], float32 (normalized)
    DataField.POINT_MASKS:   torch.Tensor,  # [B, S, H, W],    bool
    DataField.SCALE_FACTOR:  torch.Tensor,  # [B],             float32
}
```
