"""Derivation of the image tasks of the kaggle package (NoisyOffice, PatchCamelyon); see kaggle_derive.py."""
from __future__ import annotations

import gzip
import hashlib
import io
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .kaggle_derive import npz_bytes

# --- Denoising dirty documents: UCI NoisyOffice, simulated part ---------------------------------------------

NOISY_NAME = re.compile(r"^Font([fnL][tsr][em])_Noise([fwcp])_(TR|VA|TE)\.png$")
CLEAN_NAME = re.compile(r"^Font([fnL][tsr][em])_Clean_(TR|VA|TE)\.png$")


def _grey(payload):
    """Grey levels of a PNG; the noisy files are RGB with three equal channels, the clean files are mode L."""
    from PIL import Image
    with Image.open(io.BytesIO(payload)) as image:
        array = np.asarray(image)
    if array.ndim == 3:
        assert (array[..., 0] == array[..., 1]).all() and (array[..., 1] == array[..., 2]).all()
        array = array[..., 0]
    return np.ascontiguousarray(array, dtype=np.uint8)


def noisyoffice(raw, test_partition="TE"):
    """Pairs each of the 216 simulated noisy images with the clean grey-level image of the same font and text.

    Each NoisyOffice partition (TR, VA, TE) is one text passage typeset in 18 fonts, and every typeset page
    appears with 4 noise types, so splitting by partition is splitting by text: the TE passage (72 noisy
    images) is the test set and TR + VA (144 noisy images) the training set. The images are presented in a
    seeded order that carries no font, noise or partition information."""
    noisy, clean = {}, {}
    with zipfile.ZipFile(raw["noisyoffice"]) as archive:
        for name in sorted(archive.namelist()):
            parts = name.split("/")
            if "SimulatedNoisyOffice" not in parts or not parts[-1].endswith(".png"):
                continue
            folder, base = parts[-2], parts[-1]
            if folder == "simulated_noisy_images_grayscale":
                font, noise, part = NOISY_NAME.match(base).groups()
                noisy[(font, noise, part)] = _grey(archive.read(name))
            elif folder == "clean_images_grayscale":
                font, part = CLEAN_NAME.match(base).groups()
                clean[(font, part)] = _grey(archive.read(name))
    assert len(noisy) == 216 and len(clean) == 54, (len(noisy), len(clean))
    for (font, noise, part), image in noisy.items():
        assert image.shape == clean[(font, part)].shape, (font, noise, part)
    keys = sorted(noisy)
    rng = np.random.RandomState(0)
    train = [k for k in keys if k[2] != test_partition]
    test = [k for k in keys if k[2] == test_partition]
    train = [train[i] for i in rng.permutation(len(train))]
    test = [test[i] for i in rng.permutation(len(test))]

    def bundle(arrays):
        return npz_bytes({f"a{i}": a for i, a in enumerate(arrays)})
    return {
        "noisyoffice/public/train_noisy.npz": bundle([noisy[k] for k in train]),
        "noisyoffice/public/train_clean.npz": bundle([clean[(k[0], k[2])] for k in train]),
        "noisyoffice/public/test_noisy.npz": bundle([noisy[k] for k in test]),
        "noisyoffice/private/test_clean.npz": bundle([clean[(k[0], k[2])] for k in test]),
    }, {"train_images": len(train), "test_images": len(test), "test_partition": test_partition,
        "test_order": ["Font{}_Noise{}_{}".format(*k) for k in test],
        "test_pixels": int(sum(noisy[k].size for k in test)),
        "shapes": sorted({str(a.shape) for a in clean.values()})}


# --- Histopathologic cancer detection: PatchCamelyon ----------------------------------------------------------

def _h5_dataset(path, key):
    import h5py
    with h5py.File(path, "r") as handle:
        return np.asarray(handle[key])


def _gunzip(path):
    target = Path(str(path)[:-3]) if str(path).endswith(".gz") else Path(str(path) + ".raw")
    if not target.is_file():
        partial = target.with_name(target.name + ".part")
        with gzip.open(path, "rb") as source, open(partial, "wb") as out:
            shutil.copyfileobj(source, out, 1 << 22)
        partial.replace(target)
    return target


def pcam(raw, crop=48, test_fraction=0.2, seed=0, chunk=4096):
    """Pools the PCam 'valid' and 'test' files (Zenodo names; 183 whole-slide images, disjoint between them),
    removes exact duplicate patches (SHA-1 of the 96 x 96 x 3 pixels), keeps the central crop x crop pixels,
    and puts all patches of seeded-randomly chosen slides into the test set until it holds test_fraction."""
    import h5py
    lo = (96 - crop) // 2
    images, labels, slides, seen = [], [], [], {}
    duplicates = conflicting = 0
    for split in ("valid", "test"):
        y = _h5_dataset(_gunzip(raw[f"pcam-{split}-y"]), "y").reshape(-1).astype(np.int64)
        meta = pd.read_csv(raw[f"pcam-{split}-meta"], index_col=0)
        assert len(meta) == len(y) and (meta["center_tumor_patch"].to_numpy() == y.astype(bool)).all()
        x_path = _gunzip(raw[f"pcam-{split}-x"])
        with h5py.File(x_path, "r") as handle:
            x = handle["x"]
            assert x.shape == (len(y), 96, 96, 3) and x.dtype == np.uint8
            for start in range(0, len(y), chunk):
                block = np.asarray(x[start:start + chunk])
                for offset, patch in enumerate(block):
                    i = start + offset
                    digest = hashlib.sha1(patch.tobytes()).hexdigest()
                    if digest in seen:
                        duplicates += 1
                        conflicting += int(labels[seen[digest]] != y[i])
                        continue
                    seen[digest] = len(labels)
                    images.append(patch[lo:lo + crop, lo:lo + crop, :].copy())
                    labels.append(int(y[i]))
                    slides.append(meta["wsi"].iloc[i])
        x_path.unlink()  # the decompressed HDF5 (about 0.9 GB) is not kept
    images, labels, slides = np.stack(images), np.array(labels, dtype=np.int64), np.array(slides)
    names = sorted(set(slides))
    counts = pd.Series(slides).value_counts()
    test_slides, total = set(), 0
    for i in np.random.RandomState(seed).permutation(len(names)):
        if total >= test_fraction * len(labels):
            break
        test_slides.add(names[i])
        total += int(counts[names[i]])
    is_test = np.isin(slides, sorted(test_slides))
    rng = np.random.RandomState(seed + 1)
    train_idx = np.flatnonzero(~is_test)[rng.permutation(int((~is_test).sum()))]
    test_idx = np.flatnonzero(is_test)[rng.permutation(int(is_test.sum()))]
    return {
        "pcam/public/train.npz": npz_bytes({"images": images[train_idx], "labels": labels[train_idx]}),
        "pcam/public/test.npz": npz_bytes({"images": images[test_idx]}),
        "pcam/private/test_labels.npz": npz_bytes({"labels": labels[test_idx], "slides": slides[test_idx].astype("U40")}),
    }, {"patches": int(len(labels)), "duplicates_removed": duplicates, "duplicate_label_conflicts": conflicting,
        "slides": len(names), "test_slides": len(test_slides), "train": int(len(train_idx)), "test": int(len(test_idx)),
        "train_positive_fraction": round(float(labels[train_idx].mean()), 4),
        "test_positive_fraction": round(float(labels[test_idx].mean()), 4)}
