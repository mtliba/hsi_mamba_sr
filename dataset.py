# dataset_any.py
import os, glob, random, warnings
from typing import Optional, Tuple, List
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# Optional backends for .mat/.h5
try:
    import scipy.io as sio  # for MATLAB v<=7.2 .mat
except Exception:
    sio = None
try:
    import h5py  # for MATLAB v7.3 (HDF5) and generic .h5
except Exception:
    h5py = None


# -------------------- core utilities --------------------

def _to_DHW(arr: np.ndarray) -> np.ndarray:
    """Ensure array is (D,H,W). Accepts (H,W,D), (D,H,W), or squeezable 4D (H,W,D,1)/(D,H,W,1)."""
    a = np.asarray(arr)
    # drop singleton dims at ends
    while a.ndim > 3 and (a.shape[-1] == 1 or a.shape[0] == 1):
        # prefer dropping last singleton first
        if a.shape[-1] == 1:
            a = a.squeeze(-1)
        elif a.shape[0] == 1:
            a = a.squeeze(0)
        else:
            break

    if a.ndim == 2:  # single band
        a = a[None, ...]  # (1,H,W)

    if a.ndim != 3:
        raise ValueError(f"Expected 3D array for HSI, got shape {a.shape}")

    # If (H,W,D) -> transpose to (D,H,W)
    if a.shape[0] not in (1, 4, 8, 16, 31, 32, 64, 93, 128, 224, 240) and a.shape[2] in (1, 3, 4, 8, 16, 31, 32, 64, 93, 128, 224, 240):
        # Heuristic: more likely (H,W,D)
        a = np.transpose(a, (2, 0, 1))
    elif a.shape[2] not in (1, 3, 4, 8, 16, 31, 32, 64, 93, 128, 224, 240) and a.shape[0] in (1, 3, 4, 8, 16, 31, 32, 64, 93, 128, 224, 240):
        # Already (D,H,W)
        pass
    else:
        # If ambiguous (e.g., both plausible), prefer treating first dim as D when H and W are equal
        if a.shape[1] == a.shape[2] and a.shape[0] <= a.shape[1]:
            pass  # assume (D,H,W)
        else:
            # fallback: assume (H,W,D)
            a = np.transpose(a, (2, 0, 1))
    return a.astype(np.float32, copy=False)


def _normalize_auto(a: np.ndarray) -> np.ndarray:
    """Map to [0,1] with simple, safe heuristics."""
    if np.issubdtype(a.dtype, np.integer):
        maxv = np.iinfo(a.dtype).max
        if maxv == 0:
            return a.astype(np.float32)
        return (a.astype(np.float32) / float(maxv)).clip(0.0, 1.0)
    # float
    a = a.astype(np.float32, copy=False)
    m = np.nanmax(a)
    if not np.isfinite(m) or m <= 0:
        return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    # If range looks already [0,1] (or <=1.2), leave it; else scale by max
    if m <= 1.2:
        return np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
    return (np.nan_to_num(a, nan=0.0) / m).clip(0.0, 1.0)


def load_hsi(path: str, var: Optional[str] = None, norm: str = "auto") -> np.ndarray:
    """
    Load an HSI cube and return (D,H,W) float32 in [0,1] if norm='auto'.
    Supports: .mat (both <=7.2 and v7.3 HDF5), .h5/.hdf5, .npy, .npz.
    """
    ext = os.path.splitext(path)[1].lower()
    arr = None

    if ext == ".npy":
        arr = np.load(path)
    elif ext == ".npz":
        npz = np.load(path)
        # Heuristic: pick largest numeric array
        key = max((k for k in npz.files), key=lambda k: np.prod(npz[k].shape))
        arr = npz[key]
    elif ext == ".mat":
        # Try old MAT first
        loaded = None
        if sio is not None:
            try:
                loaded = sio.loadmat(path)
                # filter meta keys
                candidates: List[Tuple[str, np.ndarray]] = []
                for k, v in loaded.items():
                    if k.startswith("__"):
                        continue
                    a = np.array(v)
                    if np.issubdtype(a.dtype, np.number) and a.ndim >= 3:
                        candidates.append((k, a))
                if var is not None and var in loaded:
                    arr = np.array(loaded[var])
                elif candidates:
                    # prefer commonly used names
                    pref = ["cube", "hsi", "I", "img", "image", "data", "Y", "HSI"]
                    best = None
                    for p in pref:
                        for k, a in candidates:
                            if k.lower() == p.lower():
                                best = a; break
                        if best is not None: break
                    if best is None:
                        # pick largest
                        best = max(candidates, key=lambda kv: np.prod(kv[1].shape))[1]
                    arr = best
            except Exception:
                loaded = None
        # Try HDF5-style (MAT v7.3) if not found
        if arr is None and h5py is not None:
            try:
                with h5py.File(path, "r") as f:
                    # collect datasets
                    ds_list = []
                    def visit(name, obj):
                        if isinstance(obj, h5py.Dataset) and obj.dtype.kind in "fiu" and obj.ndim >= 3:
                            ds_list.append((name, obj.shape))
                    f.visititems(visit)
                    if var is not None and var in f:
                        arr = np.array(f[var])
                    elif ds_list:
                        # pick largest dataset
                        name = max(ds_list, key=lambda x: np.prod(x[1]))[0]
                        arr = np.array(f[name])
            except Exception:
                pass
        if arr is None:
            raise RuntimeError("Could not read .mat file. Install scipy and/or h5py, or specify --var_name.")
    elif ext in (".h5", ".hdf5"):
        if h5py is None:
            raise RuntimeError("Install h5py to read .h5/.hdf5 files.")
        with h5py.File(path, "r") as f:
            if var is not None and var in f:
                arr = np.array(f[var])
            else:
                ds_list = []
                def visit(name, obj):
                    if isinstance(obj, h5py.Dataset) and obj.dtype.kind in "fiu" and obj.ndim >= 3:
                        ds_list.append((name, obj.shape))
                f.visititems(visit)
                if not ds_list:
                    raise RuntimeError("No 3D numeric dataset found in HDF5 file.")
                name = max(ds_list, key=lambda x: np.prod(x[1]))[0]
                arr = np.array(f[name])
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    # shape -> (D,H,W)
    arr = _to_DHW(arr)

    # sanitize
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)

    # normalize
    if norm == "auto":
        arr = _normalize_auto(arr)
    elif norm == "none":
        arr = arr.astype(np.float32, copy=False)
    else:
        warnings.warn(f"Unknown norm '{norm}', using auto.")
        arr = _normalize_auto(arr)

    return arr  # (D,H,W) float32


def make_lr(hr: torch.Tensor, scale: int) -> torch.Tensor:
    """Bicubic downsample per band. hr: (D,H,W) -> lr: (D,H/s,W/s)"""
    D, H, W = hr.shape
    x = hr.view(D, 1, H, W)
    lr = F.interpolate(x, scale_factor=1/scale, mode="bicubic", align_corners=False)
    return lr.view(D, H // scale, W // scale).contiguous()


def random_crop(hr: torch.Tensor, size_hr: Optional[int]) -> torch.Tensor:
    """Random spatial crop on HR cube. hr: (D,H,W)"""
    if size_hr is None:
        return hr
    D, H, W = hr.shape
    if H < size_hr or W < size_hr:
        raise ValueError(f"Patch {size_hr} exceeds image size {(H,W)}")
    y = random.randint(0, H - size_hr)
    x = random.randint(0, W - size_hr)
    return hr[:, y:y+size_hr, x:x+size_hr].contiguous()


# -------------------- Dataset --------------------

class HSIDatasetAny(Dataset):
    """
    Loads HR cubes from a folder of .mat/.h5/.hdf5/.npy/.npz files.
    Returns (lr, hr) tensors: (D,h,w) in [0,1] if norm='auto'.
    """
    def __init__(
        self,
        root: str,
        scale: int,
        split: str = "train",
        patch_hr: Optional[int] = None,
        val_ratio: float = 0.1,
        var_name: Optional[str] = None,
        norm: str = "auto",
        allowed_exts: Tuple[str, ...] = (".mat", ".h5", ".hdf5", ".npy", ".npz"),
    ):
        super().__init__()
        self.root, self.scale, self.patch_hr = root, scale, patch_hr
        self.var_name, self.norm = var_name, norm
        files = []
        for ext in allowed_exts:
            files.extend(glob.glob(os.path.join(root, f"*{ext}")))
        files = sorted(files)
        if len(files) == 0:
            raise FileNotFoundError(f"No HSI files found in {root} (ext={allowed_exts})")

        n_val = max(1, int(len(files) * val_ratio))
        self.files = files[-n_val:] if split == "val" else files[:-n_val]
        if len(self.files) == 0:  # if too few files, put at least one in split
            self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        hr_np = load_hsi(path, var=self.var_name, norm=self.norm)   # (D,H,W)
        hr = torch.from_numpy(hr_np)                                # float32 [0,1]
        if self.patch_hr is not None:
            hr = random_crop(hr, self.patch_hr)
        lr = make_lr(hr, self.scale)
        return lr, hr
