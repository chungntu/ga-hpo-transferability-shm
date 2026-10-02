"""
Leakage-free data module.

Replaces datasetManagement.py / datasetManagement_QUGS.py. The three leaks
documented in HANDOFF_REVISION.md section 3.6 are addressed here:

  L1  sensor channels of one measurement setup were treated as independent
      samples and shuffled across train/val/test  ->  we split at RECORD
      level (a record = one .mat setup for Z24, one state file for QUGS),
      so every channel of a record stays inside one split.

  L2  a single continuous QUGS record was cut into 5 chunks that were then
      spread over the splits  ->  QUGS uses a TEMPORAL split: disjoint,
      contiguous time zones, windows never cross a zone boundary.

  L3  StandardScaler was fitted on the whole dataset  ->  the scaler here
      is fitted on the training windows only.

Every window carries its provenance (dataset, class, record, sensor, start
sample). `assert_no_leakage` re-derives the disjointness from that
provenance and is called on every load.

Public API
----------
load_records(dataset, classes, sensor_mode)      -> list[Record]
build_splits(records, ...)                       -> SplitData
load_dataset(name, ...)                          -> SplitData   (convenience)
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np
import scipy.io

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # repository root
Z24_DIR = os.path.join(ROOT, "data", "DatasetPDT")
QUGS_DIR = os.path.join(ROOT, "data", "qugs")

WINDOW_DEFAULT = 65536

# Class sets used in the paper.
Z24_CLASSES_5 = ["01", "03", "04", "05", "06"]
Z24_CLASSES_15 = ["01", "03", "04", "05", "06", "07", "09", "10",
                  "11", "12", "13", "14", "15", "16", "17"]


# ----------------------------------------------------------------------
# records
# ----------------------------------------------------------------------

@dataclass
class Record:
    """One physically simultaneous measurement: (T, C) over C sensors."""
    dataset: str          # 'z24' | 'qugs-A' | 'qugs-B'
    class_name: str
    class_idx: int
    record_id: str        # unique within dataset, e.g. '01setup03'
    data: np.ndarray      # (T, C) float32
    sensor_ids: list      # length C, sensor label per column

    @property
    def n_samples(self):
        return self.data.shape[0]


def _first_ndarray(mat_dict):
    for k, v in mat_dict.items():
        if not k.startswith("__") and isinstance(v, np.ndarray):
            return v
    return None


def load_records(dataset, classes=None, sensor_mode="all", max_sensors=None):
    """
    dataset     : 'z24' | 'qugs-A' | 'qugs-B'
    classes     : list of class names; None -> dataset default
    sensor_mode : z24 -> '5' (the 5 fixed sensors in avt/Processed) or
                  'all' (33 channels in avt/*.mat)
                  qugs -> 'all' (30 accelerometers)
    """
    if dataset == "z24":
        recs = _load_z24(classes or Z24_CLASSES_15, sensor_mode)
    elif dataset in ("qugs-A", "qugs-B"):
        recs = _load_qugs(dataset[-1], classes)
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    if max_sensors is not None:
        # paper section 3.3: QUGS Small-scale uses sensors 01-05, Full-scale 01-15
        for r in recs:
            r.data = r.data[:, :max_sensors]
            r.sensor_ids = r.sensor_ids[:max_sensors]
    return recs


def _load_z24(classes, sensor_mode):
    records = []
    for class_idx, cname in enumerate(classes):
        if sensor_mode == "5":
            cdir = os.path.join(Z24_DIR, cname, "avt", "Processed")
        elif sensor_mode == "all":
            cdir = os.path.join(Z24_DIR, cname, "avt")
        else:
            raise ValueError(f"z24 sensor_mode must be '5' or 'all', got {sensor_mode!r}")
        if not os.path.isdir(cdir):
            raise FileNotFoundError(cdir)

        for fname in sorted(f for f in os.listdir(cdir) if f.endswith(".mat")):
            mat = scipy.io.loadmat(os.path.join(cdir, fname))
            arr = mat["data"] if "data" in mat else _first_ndarray(mat)
            if arr is None:
                raise ValueError(f"no array inside {fname}")
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(f"{fname}: expected 2-D, got {arr.shape}")
            # stored as (time, channel)
            rid = os.path.splitext(fname)[0].replace("_R", "")
            records.append(Record(
                dataset="z24",
                class_name=cname,
                class_idx=class_idx,
                record_id=f"{cname}/{rid}",
                data=arr,
                sensor_ids=[f"ch{j:02d}" for j in range(arr.shape[1])],
            ))
    return records


def _load_qugs(subset, classes):
    sdir = os.path.join(QUGS_DIR, subset)
    if not os.path.isdir(sdir):
        raise FileNotFoundError(f"{sdir} — run prepare_qugs.py first")
    available = sorted(f for f in os.listdir(sdir) if f.endswith(".npy"))
    names = [os.path.splitext(f)[0] for f in available]
    if classes is None:
        classes = names
    records = []
    for class_idx, cname in enumerate(classes):
        path = os.path.join(sdir, f"{cname}.npy")
        arr = np.load(path).astype(np.float32, copy=False)
        records.append(Record(
            dataset=f"qugs-{subset}",
            class_name=cname,
            class_idx=class_idx,
            record_id=f"{subset}/{cname}",
            data=arr,
            sensor_ids=[f"acc{j:02d}" for j in range(arr.shape[1])],
        ))
    return records


# ----------------------------------------------------------------------
# splitting
# ----------------------------------------------------------------------

@dataclass
class SplitData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    width: int
    n_classes: int
    meta: dict = field(default_factory=dict)      # provenance per split
    scaler: tuple = (0.0, 1.0)

    def summary(self):
        return (f"train {self.X_train.shape} | val {self.X_val.shape} | "
                f"test {self.X_test.shape} | {self.n_classes} classes")


def _window_starts(length, window, stride, lo=0, hi=None):
    hi = length if hi is None else hi
    starts = list(range(lo, hi - window + 1, stride))
    return starts


def _pad_to(arr1d, target):
    """Pad a 1-D signal to `target` by repeating its last value (as in the original code)."""
    if len(arr1d) >= target:
        return arr1d[:target]
    pad = np.full(target - len(arr1d), arr1d[-1], dtype=arr1d.dtype)
    return np.concatenate([arr1d, pad])


def build_splits(records, seed=0, window_length=WINDOW_DEFAULT,
                 split_scheme="record", fracs=(0.7, 0.1, 0.2), stride=None,
                 scaler="global", pad_short_records=True, test_records=None,
                 fold=None, multichannel=False, verbose=True):
    """
    split_scheme
        'record'   : whole records go to one split (Z24: several setups/class).
        'temporal' : each record is cut into three contiguous, disjoint time
                     zones (QUGS: a single long record per state).
        'campaign' : train/val are two time zones of `records`; the test set is
                     every window of `test_records`, a separate measurement
                     campaign (QUGS Dataset A -> Dataset B).
        'loso'     : leave-one-setup-out CV for Z24; needs `fold`.
        'random_window' : DELIBERATELY LEAKY control; see the branch below.

    multichannel
        False (default): one sample per (record, channel, window), shape (T, 1).
            This is what the submitted paper's code does, and it discards the
            fact that the channels were recorded SIMULTANEOUSLY.
        True: one sample per (record, window) carrying every channel, shape
            (T, C). Phase and amplitude relations between measurement points
            become available to the model. Only valid when channel i means the
            same physical location in every record -- true for the five Z24
            sensors kept fixed across setups, and for all 30 QUGS joints, but
            NOT for the 33 channels of an arbitrary Z24 setup, where channel i
            is a different position in different setups.

    Returns a SplitData whose X are (N, window_length, 1) float32, scaled with
    statistics computed on the training windows only.
    """
    stride = stride or window_length              # non-overlapping by default
    rng = np.random.default_rng(seed)
    n_classes = len(set(r.class_idx for r in records))

    buckets = {"train": [], "val": [], "test": []}   # list of (window, label, prov)

    def emit(split, rec, ch, start, sig):
        buckets[split].append((
            sig,
            rec.class_idx,
            dict(dataset=rec.dataset, class_name=rec.class_name,
                 record_id=rec.record_id, sensor=rec.sensor_ids[ch],
                 start=int(start), end=int(start + window_length)),
        ))

    def emit_mc(split, rec, start, block):
        buckets[split].append((
            block,
            rec.class_idx,
            dict(dataset=rec.dataset, class_name=rec.class_name,
                 record_id=rec.record_id, sensor="+".join(rec.sensor_ids),
                 start=int(start), end=int(start + window_length)),
        ))

    def windows_from(rec, ch, lo, hi, split):
        """Emit windows of one channel, or -- when multichannel and ch == 0 --
        one multi-channel block per window. `ch` is ignored in that case, so
        callers can keep looping over channels unchanged."""
        T = rec.data.shape[0]
        hi = min(hi, T)
        if multichannel:
            if ch != 0:
                return
            if hi - lo < window_length:
                if not pad_short_records or lo != 0:
                    return
                block = np.stack([_pad_to(rec.data[lo:hi, c], window_length)
                                  for c in range(rec.data.shape[1])], axis=1)
                emit_mc(split, rec, lo, block)
                return
            for s in _window_starts(T, window_length, stride, lo, hi):
                emit_mc(split, rec, s, rec.data[s:s + window_length, :])
            return
        if hi - lo < window_length:
            if not pad_short_records or lo != 0:
                return
            sig = _pad_to(rec.data[lo:hi, ch], window_length)
            emit(split, rec, ch, lo, sig)
            return
        for s in _window_starts(T, window_length, stride, lo, hi):
            emit(split, rec, ch, s, rec.data[s:s + window_length, ch])

    if split_scheme == "record":
        by_class = {}
        for r in records:
            by_class.setdefault(r.class_idx, []).append(r)
        for cidx, recs in sorted(by_class.items()):
            recs = sorted(recs, key=lambda r: r.record_id)
            order = rng.permutation(len(recs))
            n = len(recs)
            n_tr = max(1, int(round(fracs[0] * n)))
            n_va = max(1, int(round(fracs[1] * n)))
            if n_tr + n_va >= n:                       # keep at least one test record
                n_va = max(1, min(n_va, n - n_tr - 1))
                n_tr = max(1, n - n_va - 1)
            assign = (["train"] * n_tr + ["val"] * n_va +
                      ["test"] * (n - n_tr - n_va))
            for pos, split in zip(order, assign):
                rec = recs[pos]
                for ch in range(rec.data.shape[1]):
                    windows_from(rec, ch, 0, rec.data.shape[0], split)

    elif split_scheme == "temporal":
        for rec in sorted(records, key=lambda r: r.record_id):
            T = rec.data.shape[0]
            b1 = int(T * fracs[0])
            b2 = int(T * (fracs[0] + fracs[1]))
            zones = (("train", 0, b1), ("val", b1, b2), ("test", b2, T))
            for ch in range(rec.data.shape[1]):
                for split, lo, hi in zones:
                    windows_from(rec, ch, lo, hi, split)
    elif split_scheme == "random_window":
        # DELIBERATELY LEAKY. Every window of every record is pooled and split
        # at random, so windows and channels of one measurement setup land in
        # train AND test. This is the protocol of the submitted code and of much
        # of the Z24 literature; it exists here ONLY as a control, to measure how
        # much of the published accuracy the protocol alone explains.
        # assert_no_leakage deliberately does not police this scheme.
        pool = []
        for rec in sorted(records, key=lambda r: r.record_id):
            T = rec.data.shape[0]
            starts = _window_starts(T, window_length, stride, 0, T)
            if not starts and pad_short_records:
                # records shorter than one window contribute a single padded
                # window, exactly as windows_from does for the other schemes.
                # Z24 records are 65530 samples, so at window 65536 this branch
                # carries 9 of the 44 records; dropping them silently would
                # remove most of one class.
                starts = [0]
            for ch in range(rec.data.shape[1]):
                if multichannel and ch != 0:
                    continue
                for st in starts:
                    pool.append((rec, ch, st))
        order = rng.permutation(len(pool))
        n_pool = len(pool)
        n_tr = int(round(fracs[0] * n_pool))
        n_va = int(round(fracs[1] * n_pool))
        for pos, idx in enumerate(order):
            rec, ch, st = pool[idx]
            split = ("train" if pos < n_tr else
                     "val" if pos < n_tr + n_va else "test")
            if multichannel:
                block = rec.data[st:st + window_length, :]
                if block.shape[0] < window_length:
                    block = np.stack([_pad_to(rec.data[st:, c], window_length)
                                      for c in range(rec.data.shape[1])], axis=1)
                emit_mc(split, rec, st, block)
            else:
                sig = rec.data[st:st + window_length, ch]
                if sig.shape[0] < window_length:
                    sig = _pad_to(rec.data[st:, ch], window_length)
                emit(split, rec, ch, st, sig)

    elif split_scheme == "loso":
        # Leave-one-setup-out cross-validation for Z24. Each structural state
        # was measured in 9 separate setups; a single record-level split puts
        # only ONE record in validation, which no confidence interval can
        # rescue. Fold f holds out setup f as test and setup f+1 as validation,
        # for every class, so all 9 setups are used exactly once as test across
        # the folds. Still strictly record-disjoint.
        #
        # Class 03 has 8 setups rather than 9, so with 9 folds its first setup
        # is tested twice (folds 0 and 8) while the others are tested once.
        # The fold index is taken modulo each class's own setup count, so no
        # fold is ever empty; the slight imbalance is reported with the results
        # rather than hidden.
        if fold is None:
            raise ValueError("split_scheme='loso' needs fold=")
        by_class = {}
        for r in records:
            by_class.setdefault(r.class_idx, []).append(r)
        for cidx, recs in sorted(by_class.items()):
            recs = sorted(recs, key=lambda r: r.record_id)
            n = len(recs)
            i_test, i_val = fold % n, (fold + 1) % n
            for i, rec in enumerate(recs):
                split = ("test" if i == i_test else
                         "val" if i == i_val else "train")
                for ch in range(rec.data.shape[1]):
                    windows_from(rec, ch, 0, rec.data.shape[0], split)

    elif split_scheme == "campaign":
        # train/val are temporal zones of the SOURCE campaign records; the whole
        # TARGET campaign is the test set. Used for QUGS, where each damage
        # state has exactly one record so a record-level split is impossible:
        # Dataset B is a separate measurement campaign, which is the strongest
        # independence available (HANDOFF section 4.3 / checklist #3).
        if test_records is None:
            raise ValueError("split_scheme='campaign' needs test_records")
        f_tr = fracs[0] / (fracs[0] + fracs[1])
        for rec in sorted(records, key=lambda r: r.record_id):
            T = rec.data.shape[0]
            b1 = int(T * f_tr)
            for ch in range(rec.data.shape[1]):
                windows_from(rec, ch, 0, b1, "train")
                windows_from(rec, ch, b1, T, "val")
        for rec in sorted(test_records, key=lambda r: r.record_id):
            for ch in range(rec.data.shape[1]):
                windows_from(rec, ch, 0, rec.data.shape[0], "test")
    else:
        raise ValueError(f"unknown split_scheme {split_scheme!r}")

    for k in buckets:
        if not buckets[k]:
            raise RuntimeError(f"split '{k}' is empty — check fracs/window_length")

    def stack(split):
        sigs, labels, prov = zip(*buckets[split])
        X = np.stack(sigs).astype(np.float32)
        y = np.asarray(labels, dtype=np.int64)
        return X, y, list(prov)

    X_train, y_train, m_train = stack("train")
    X_val, y_val, m_val = stack("val")
    X_test, y_test, m_test = stack("test")

    # --- scaler: every variant is leakage-free (fitted on TRAIN ONLY, or
    #     computed independently per window).
    #       'global'       one mean/std over all training windows
    #       'per_timestep' one mean/std per time index, fitted on train
    #                      (closest to the original StandardScaler, which was
    #                      however fitted on the whole dataset)
    #       'per_window'   instance normalisation; uses no cross-sample
    #                      statistics at all, but discards amplitude
    if scaler == "global":
        mu = float(X_train.mean())
        sd = float(X_train.std()) or 1.0
        for X in (X_train, X_val, X_test):
            X -= mu
            X /= sd
        scaler_info = ("global", mu, sd)
    elif scaler == "per_timestep":
        mu_t = X_train.mean(axis=0)
        sd_t = X_train.std(axis=0)
        sd_t[sd_t == 0] = 1.0
        for X in (X_train, X_val, X_test):
            X -= mu_t
            X /= sd_t
        scaler_info = ("per_timestep", float(mu_t.mean()), float(sd_t.mean()))
    elif scaler == "per_window":
        for X in (X_train, X_val, X_test):
            m = X.mean(axis=1, keepdims=True)
            s = X.std(axis=1, keepdims=True)
            s[s == 0] = 1.0
            X -= m
            X /= s
        scaler_info = ("per_window", 0.0, 1.0)
    elif scaler in ("global_all", "per_timestep_all"):
        # LEAKY BY DESIGN, for the protocol-reproduction control only: the
        # statistics are fitted on train+val+test together, reproducing the
        # submitted code's StandardScaler().fit_transform() call on the whole
        # dataset before splitting (leak L3).
        pool = np.concatenate([X_train, X_val, X_test], axis=0)
        if scaler == "global_all":
            mu_a = float(pool.mean())
            sd_a = float(pool.std()) or 1.0
            for X in (X_train, X_val, X_test):
                X -= mu_a
                X /= sd_a
            scaler_info = (scaler, mu_a, sd_a)
        else:
            mu_t = pool.mean(axis=0)
            sd_t = pool.std(axis=0)
            sd_t[sd_t == 0] = 1.0
            for X in (X_train, X_val, X_test):
                X -= mu_t
                X /= sd_t
            scaler_info = (scaler, float(mu_t.mean()), float(sd_t.mean()))
        print(f"[shm_data][CONTROL] scaler={scaler!r} is fitted on ALL splits "
              f"by design (reproduces leak L3)", flush=True)
    else:
        raise ValueError(f"unknown scaler {scaler!r}")
    mu, sd = scaler_info[1], scaler_info[2]

    def add_axis(X):
        # single-channel windows are stored (N, T) and need an explicit channel
        # axis; multi-channel blocks are already (N, T, C).
        return X if X.ndim == 3 else X[..., None]

    data = SplitData(
        X_train=add_axis(X_train), y_train=y_train,
        X_val=add_axis(X_val), y_val=y_val,
        X_test=add_axis(X_test), y_test=y_test,
        width=window_length, n_classes=n_classes,
        meta=dict(train=m_train, val=m_val, test=m_test,
                  seed=seed, split_scheme=split_scheme, fracs=list(fracs),
                  stride=stride, window_length=window_length,
                  scaler=scaler_info[0], fold=fold,
                  multichannel=bool(multichannel)),
        scaler=(mu, sd),
    )

    assert_no_leakage(data)
    if verbose:
        print(f"[shm_data] {data.summary()}  scheme={split_scheme} seed={seed} "
              f"scaler={scaler_info[0]}(mu={mu:.4g}, sd={sd:.4g})", flush=True)
    return data


# ----------------------------------------------------------------------
# leakage check
# ----------------------------------------------------------------------

def assert_no_leakage(data: SplitData):
    """Re-derive disjointness from the stored provenance. Raises on any overlap."""
    scheme = data.meta["split_scheme"]
    if scheme == "random_window":
        print("[shm_data][CONTROL] split_scheme='random_window' is deliberately "
              "leaky; no-leakage assertions skipped by design", flush=True)
        return True
    if scheme == "campaign":
        # test must come from a campaign that contributes nothing to train/val
        src = {m["dataset"] for m in data.meta["train"]} | {m["dataset"] for m in data.meta["val"]}
        tgt = {m["dataset"] for m in data.meta["test"]}
        if src & tgt:
            raise AssertionError(f"LEAKAGE: campaign(s) {sorted(src & tgt)} in both "
                                 f"train/val and test")
        # and train vs val must still be disjoint in time, record by record
        spans = {}
        for split in ("train", "val"):
            for m in data.meta[split]:
                k = (m["dataset"], m["record_id"], m["sensor"])
                lo, hi = spans.setdefault((split, k), (m["start"], m["end"]))
                spans[(split, k)] = (min(lo, m["start"]), max(hi, m["end"]))
        for (split, k), (lo, hi) in spans.items():
            if split != "train":
                continue
            other = spans.get(("val", k))
            if other and lo < other[1] and other[0] < hi:
                raise AssertionError(f"LEAKAGE: train/val time overlap on {k}: "
                                     f"({lo},{hi}) vs {other}")
        return True

    keys = {}
    for split in ("train", "val", "test"):
        if scheme in ("record", "loso"):
            keys[split] = {(m["dataset"], m["record_id"]) for m in data.meta[split]}
        else:                                   # temporal: identical windows
            keys[split] = {(m["dataset"], m["record_id"], m["sensor"],
                            m["start"]) for m in data.meta[split]}

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = keys[a] & keys[b]
        if overlap:
            raise AssertionError(
                f"LEAKAGE: {len(overlap)} shared keys between {a} and {b}; "
                f"example {sorted(overlap)[:3]}")

    if scheme == "temporal":
        # no window may straddle a zone used by another split
        spans = {s: {} for s in ("train", "val", "test")}
        for split in spans:
            for m in data.meta[split]:
                k = (m["dataset"], m["record_id"], m["sensor"])
                lo, hi = spans[split].setdefault(k, (m["start"], m["end"]))
                spans[split][k] = (min(lo, m["start"]), max(hi, m["end"]))
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            for k, (alo, ahi) in spans[a].items():
                if k in spans[b]:
                    blo, bhi = spans[b][k]
                    if alo < bhi and blo < ahi:
                        raise AssertionError(
                            f"LEAKAGE: overlapping time spans on {k}: "
                            f"{a}=({alo},{ahi}) {b}=({blo},{bhi})")

    # labels must cover the same classes
    for split in ("val", "test"):
        missing = set(np.unique(data.y_train)) - set(np.unique(getattr(data, f"y_{split}")))
        if missing:
            print(f"[shm_data][WARN] classes {sorted(missing)} absent from {split}")
    return True


# ----------------------------------------------------------------------
# convenience
# ----------------------------------------------------------------------

QUGS_5 = [f"state{i:02d}" for i in range(1, 6)]
QUGS_15 = [f"state{i:02d}" for i in range(1, 16)]

# A QUGS record is 262144 samples = exactly 4 windows of 65536, so the
# temporal zones are 50/25/25: 2 train windows, 1 val, 1 test per sensor.
# 70/10/20 would leave the val and test zones shorter than one window.
# Sensor/class counts follow the submitted paper (section 3.2.1, 3.3, Table 5):
#   Z24  small  5 fixed sensors, 5 classes    | full  all 33 channels, 15 classes
#   QUGS small  sensors 01-05, cases 01-05    | full  sensors 01-15, cases 01-15
DATASETS = {
    # name           dataset   classes    sensors  max_s  scheme      fracs
    "z24_small":   ("z24",    Z24_CLASSES_5,  "5",   None, "record",   (0.7, 0.1, 0.2)),
    "z24_full":    ("z24",    Z24_CLASSES_15, "all", None, "record",   (0.7, 0.1, 0.2)),
    # multi-channel formulation. Z24 uses ONLY the five sensors kept fixed
    # across setups: channel i of one setup is a different physical position in
    # another setup, so a multi-channel model over all 33 would learn a spatial
    # arrangement that does not exist consistently.
    "z24_small_mc":  ("z24",  Z24_CLASSES_5,  "5",   None, "record",   (0.7, 0.1, 0.2)),
    "z24_full_mc":   ("z24",  Z24_CLASSES_15, "5",   None, "record",   (0.7, 0.1, 0.2)),
    # deliberately leaky controls: identical to the rows above except that the
    # split is random over windows.
    "z24_small_leaky": ("z24", Z24_CLASSES_5,  "5",   None, "random_window", (0.7, 0.1, 0.2)),
    "z24_full_leaky":  ("z24", Z24_CLASSES_15, "all", None, "random_window", (0.7, 0.1, 0.2)),
    # QUGS trains and validates on campaign A and is tested on campaign B,
    # which the submitted paper never used.
    "qugs_small":  ("qugs-A", QUGS_5,         "all", 5,    "campaign", (0.75, 0.25, 0.0)),
    "qugs_full":   ("qugs-A", QUGS_15,        "all", 15,   "campaign", (0.75, 0.25, 0.0)),
    "qugs_small_mc": ("qugs-A", QUGS_5,       "all", 5,    "campaign", (0.75, 0.25, 0.0)),
    "qugs_full_mc":  ("qugs-A", QUGS_15,      "all", 15,   "campaign", (0.75, 0.25, 0.0)),
    # the within-campaign temporal split, kept for comparison
    "qugs_small_A": ("qugs-A", QUGS_5,        "all", 5,    "temporal", (0.5, 0.25, 0.25)),
    "qugs_full_A":  ("qugs-A", QUGS_15,       "all", 15,   "temporal", (0.5, 0.25, 0.25)),
}


# Full-scale scenarios yield far more windows than the study needs (z24_full
# alone gives 92k training windows). Taking every 4th window keeps uniform
# coverage of every record while cutting the grid cost by 4x. It changes the
# sampling density only, never which records land in which split.
# z24_full uses 8x rather than 4x: it is by far the heaviest scenario (33
# channels x 9 setups x 15 classes) and at 8x it still yields ~11.6k training
# windows, about 770 per class. Sampling density need not match across
# datasets -- the records differ in length and count anyway -- and every
# analysis compares ranks WITHIN a scenario before comparing across.
STRIDE_MULT = {"z24_full": 8, "qugs_full": 4, "qugs_full_A": 4,
               "z24_full_leaky": 8, "qugs_full_mc": 2}
# multi-channel presets already emit C times fewer samples, so no extra thinning
MULTICHANNEL = {"z24_small_mc", "z24_full_mc", "qugs_small_mc", "qugs_full_mc"}


def load_dataset(name, seed=0, window_length=WINDOW_DEFAULT, **kw):
    if name not in DATASETS:
        raise ValueError(f"unknown dataset preset {name!r}; have {list(DATASETS)}")
    ds, classes, sensors, max_sensors, scheme, fracs = DATASETS[name]
    kw.setdefault("fracs", fracs)
    if name in STRIDE_MULT:
        kw.setdefault("stride", window_length * STRIDE_MULT[name])
    if name in MULTICHANNEL:
        kw.setdefault("multichannel", True)
    records = load_records(ds, classes, sensors, max_sensors=max_sensors)
    if scheme == "campaign":
        target = "qugs-B" if ds == "qugs-A" else "qugs-A"
        kw.setdefault("test_records",
                      load_records(target, classes, sensors, max_sensors=max_sensors))
    data = build_splits(records, seed=seed, window_length=window_length,
                        split_scheme=scheme, **kw)
    data.meta["preset"] = name
    return data


def provenance_digest(data: SplitData):
    """Short hash of the split, to record in the results CSV."""
    payload = {s: sorted(f"{m['record_id']}|{m['sensor']}|{m['start']}"
                         for m in data.meta[s]) for s in ("train", "val", "test")}
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("preset", choices=list(DATASETS))
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    d = load_dataset(a.preset, seed=a.seed)
    print(d.summary())
    print("digest", provenance_digest(d))
    for s in ("train", "val", "test"):
        recs = sorted({m["record_id"] for m in d.meta[s]})
        print(f"  {s:5s}: {len(d.meta[s])} windows from {len(recs)} records")
