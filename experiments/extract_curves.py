"""
Extract the per-epoch validation curves from results/details/ into one compact file.

results/details/ is 94 MB across ~7000 files, almost all of it raw predictions and
confusion matrices, and it is excluded from the repository. But the per-epoch
validation curves inside it are what lets hpo_offline.py simulate budget-allocating
search exactly rather than approximately, so without them one result section cannot
be reproduced. The curves alone compress to a couple of megabytes, small enough to
version.

Writes results/curves.json:  run_id -> {experiment, model, dataset, seed, config,
val_acc curve, final metrics}
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  _os.pardir, "src"))
import csv, glob, json, os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
RESULTS = os.path.join(HERE, "results")
GENES = {"wavenet": ["lr", "filters", "res_per_block", "n_blocks"],
         "cnn1d": ["lr", "filters", "n_blocks", "kernel_size"]}


def main():
    meta = {}
    with open(os.path.join(RESULTS, "runs.csv"), newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            meta[r["run_id"]] = r

    out, skipped = {}, 0
    for p in sorted(glob.glob(os.path.join(RESULTS, "details", "*.json"))):
        rid = os.path.splitext(os.path.basename(p))[0]
        m = meta.get(rid)
        if not m or m.get("model") not in GENES:
            skipped += 1
            continue
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        hist = d.get("history") or []
        if not hist:
            skipped += 1
            continue
        model = m["model"]
        try:
            cfg = {g: (float(m[g]) if g == "lr" else int(m[g])) for g in GENES[model]}
        except (ValueError, KeyError):
            skipped += 1
            continue
        out[rid] = dict(
            experiment=m["experiment"], model=model, dataset=m["dataset_train"],
            seed=int(m["seed"]), config=cfg,
            val_acc=[round(h["val_acc"], 6) for h in hist],
            final=dict(val_acc=float(m["val_acc"]), test_acc=float(m["test_acc"]),
                       test_macro_f1=float(m["test_macro_f1"])),
        )

    dst = os.path.join(RESULTS, "curves.json")
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(dict(note="per-epoch validation curves extracted from "
                            "results/details/, which is not versioned",
                       runs=out), fh, separators=(",", ":"))
    mb = os.path.getsize(dst) / 1e6
    print(f"{len(out)} runs written, {skipped} skipped -> results/curves.json "
          f"({mb:.1f} MB)")


if __name__ == "__main__":
    main()
