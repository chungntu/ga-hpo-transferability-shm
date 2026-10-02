"""
Third diagnostic: is the 65536-sample window the reason WaveNet cannot
generalise across records?

What we know so far, all on the leakage-free Z24 small-scale split:
  * 72/72 grid configs sit at chance after 15 epochs;
  * with 200 epochs the network reaches train acc 0.83 but test 0.11, i.e.
    it memorises the training records and does not transfer across setups;
  * a PSD + logistic-regression baseline on the SAME split reaches 0.71 test.

The baseline compresses 65536 raw points into 128 spectral features. The
network is asked to learn from 150 training examples of dimension 65536.
Shortening the window trades sequence length for sample count without
touching the split: the records assigned to train/val/test stay exactly the
same, only the number of windows cut from them changes. assert_no_leakage
re-checks this on every load.

window 65536 -> 1 window per channel per record   (150 train windows)
window  8192 -> 8                                  (1200)
window  2048 -> 32                                 (4800)
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  _os.pardir, _os.pardir, "src"))
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import json, os
from runlog import log_run, new_run_id
from shm_data import load_dataset, provenance_digest
from wavenet_torch import train_eval

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root
BASE = dict(lr=1e-4, filters=8, res_per_block=8, n_blocks=2)   # inside paper Table 2
WINDOWS = [65536, 8192, 2048]
EPOCHS = 60

def main():
    rows = []
    for w in WINDOWS:
        data = load_dataset("z24_small", seed=0, window_length=w)
        digest = provenance_digest(data)
        n_sensors = len({m["sensor"] for m in data.meta["train"]})
        print(f"\n=== window={w} ({data.X_train.shape[0]} train windows) ===", flush=True)
        m, ex = train_eval(data, epochs=EPOCHS, head="flatten", seed=0,
                           verbose=0, early_stop_flag=0, **BASE)
        log_run(dict(
            run_id=new_run_id("diag"), experiment="diag_window",
            dataset_train="z24_small", dataset_eval="z24_small",
            n_classes=data.n_classes, n_sensors=n_sensors, window_length=w,
            split_scheme=data.meta["split_scheme"], split_digest=digest,
            config_source="window_probe",
            notes=f"window probe; same record split, {data.X_train.shape[0]} train windows",
            **{k: m[k] for k in ("lr","filters","res_per_block","n_blocks",
               "epochs","epochs_run","batch_size","head","n_params","seed",
               "val_acc","val_acc_best","test_acc","test_macro_f1",
               "test_balanced_acc","stopped_early","train_time_sec","gpu_name")},
        ), ex)
        h = ex["history"]
        rows.append(dict(window=w, n_train=int(data.X_train.shape[0]),
                         n_test=int(data.X_test.shape[0]),
                         n_params=m["n_params"], final_train_acc=h[-1]["acc"],
                         val_acc=m["val_acc"], val_acc_best=m["val_acc_best"],
                         test_acc=m["test_acc"], test_macro_f1=m["test_macro_f1"],
                         test_balanced_acc=m["test_balanced_acc"],
                         train_time_sec=m["train_time_sec"]))
        r = rows[-1]
        print(f"  train {r['final_train_acc']:.3f}  val {r['val_acc']:.3f}"
              f" (best {r['val_acc_best']:.3f})  test {r['test_acc']:.3f}"
              f"  macroF1 {r['test_macro_f1']:.3f}  {r['train_time_sec']:.0f}s", flush=True)

    with open(os.path.join(HERE,"results","diag_window.json"),"w",encoding="utf-8") as f:
        json.dump(dict(base=BASE, epochs=EPOCHS, rows=rows), f, indent=2)
    print("\n" + "="*72)
    print(f"{'window':>7} {'ntrain':>7} {'ntest':>6} {'train':>7} {'val':>7} {'test':>7} {'macroF1':>8}")
    for r in rows:
        print(f"{r['window']:>7} {r['n_train']:>7} {r['n_test']:>6} "
              f"{r['final_train_acc']:>7.3f} {r['val_acc']:>7.3f} "
              f"{r['test_acc']:>7.3f} {r['test_macro_f1']:>8.3f}")
    print("chance 0.200 | PSD+logreg (65536 window) 0.711 test")

if __name__ == "__main__":
    main()
