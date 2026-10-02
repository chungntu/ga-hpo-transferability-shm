"""
Follow-up diagnostic: after diag_budget_head.py showed the network is
UNDERfitting (train acc 0.45 after 100 epochs, GAP head dead at 1.5k params),
separate the two remaining explanations for chance-level accuracy:

  * not enough epochs, or
  * a learning rate too small to move the model at all.

The paper's Table 2 caps lr at 1e-4. 3e-4 and 1e-3 sit OUTSIDE that space on
purpose: if the task only becomes learnable above the declared ceiling, then
the search space itself was misspecified, which is a finding in its own right
and must be reported rather than quietly fixed.

Flatten head throughout (GAP is ruled out). Early stopping off.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  _os.pardir, _os.pardir, "src"))
import json, os
from runlog import log_run, new_run_id
from shm_data import load_dataset, provenance_digest
from wavenet_torch import train_eval

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root
BASE = dict(filters=4, res_per_block=8, n_blocks=2)
EPOCHS = 200
LRS = [7e-5, 3e-4, 1e-3]

def main():
    data = load_dataset("z24_small", seed=0)
    digest = provenance_digest(data)
    n_sensors = len({m["sensor"] for m in data.meta["train"]})
    rows = []
    for lr in LRS:
        print(f"\n=== lr={lr:g} epochs={EPOCHS} ===", flush=True)
        m, ex = train_eval(data, lr=lr, epochs=EPOCHS, head="flatten", seed=0,
                           verbose=0, early_stop_flag=0, **BASE)
        log_run(dict(
            run_id=new_run_id("diag"), experiment="diag_lr_budget",
            dataset_train="z24_small", dataset_eval="z24_small",
            n_classes=data.n_classes, n_sensors=n_sensors,
            window_length=data.width, split_scheme=data.meta["split_scheme"],
            split_digest=digest, config_source="paper_table3_small_lr_probe",
            notes="lr probe; 3e-4 and 1e-3 are outside paper Table 2",
            **{k: m[k] for k in ("lr","filters","res_per_block","n_blocks",
               "epochs","epochs_run","batch_size","head","n_params","seed",
               "val_acc","val_acc_best","test_acc","test_macro_f1",
               "test_balanced_acc","stopped_early","train_time_sec","gpu_name")},
        ), ex)
        h = ex["history"]
        rows.append(dict(lr=lr, final_train_acc=h[-1]["acc"],
                         best_train_acc=max(x["acc"] for x in h),
                         val_acc=m["val_acc"], val_acc_best=m["val_acc_best"],
                         test_acc=m["test_acc"], test_macro_f1=m["test_macro_f1"],
                         train_curve=[x["acc"] for x in h],
                         val_curve=[x["val_acc"] for x in h]))
        print(f"  train {h[-1]['acc']:.3f} (best {rows[-1]['best_train_acc']:.3f})"
              f"  val {m['val_acc']:.3f} (best {m['val_acc_best']:.3f})"
              f"  test {m['test_acc']:.3f}  macroF1 {m['test_macro_f1']:.3f}", flush=True)
    with open(os.path.join(HERE,"results","diag_lr_budget.json"),"w",encoding="utf-8") as f:
        json.dump(dict(base=BASE, epochs=EPOCHS, rows=rows), f, indent=2)
    print("\n" + "="*62)
    print(f"{'lr':>8} {'train':>7} {'val':>7} {'bestval':>8} {'test':>7} {'macroF1':>8}")
    for r in rows:
        print(f"{r['lr']:>8.0e} {r['final_train_acc']:>7.3f} {r['val_acc']:>7.3f} "
              f"{r['val_acc_best']:>8.3f} {r['test_acc']:>7.3f} {r['test_macro_f1']:>8.3f}")
    print(f"chance 0.200 | PSD+logreg 0.711 test")


if __name__ == "__main__":
    main()
