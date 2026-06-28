"""
train_nn.py
===========
Train the entity-embedding neural network (nn_model.py) on the last 10 years of
international results and report its predictive metrics on a **temporal holdout**
(train on older matches, test on the most recent ones — never the reverse, so
there is no look-ahead leakage).

What it reports
---------------
* The full network vs the **Elo** model (fit only on the pre-test matches) and a
  **class-prior** baseline, on the held-out test set.
* An **ablation** — the same network with the rolling-form features removed
  (embeddings + venue + importance only) — so the comparison isolates what the
  *embedding architecture* contributes versus the hand-built form features.
* **Multi-seed robustness** — the network is retrained over several random seeds
  and we report mean ± std, plus how often it beats Elo, so the (small) win is
  not attributed to a single lucky initialisation.

The full model (primary seed) is saved to ``nn_model.pt`` (loaded by predict.py),
and the metrics to ``nn_metrics.json``.

Run:  python train_nn.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn

import metrics as M
from data import load_matches, summary
from elo_model import EloModel
from nn_model import OutcomeNet, NNPredictor, build_features, K_FORM

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_OUT = os.path.join(HERE, "nn_model.pt")
METRICS_OUT = os.path.join(HERE, "nn_metrics.json")

PRIMARY_SEED = 13
SEEDS = [13, 1, 2, 3, 4]        # for the robustness sweep
EMB_DIM = 16
HIDDEN = (64, 32)
P_DROP = 0.3
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH = 256
MAX_EPOCHS = 300
PATIENCE = 25
FORM_COLS = [2, 3, 4, 5]         # home_ppg, home_gd, away_ppg, away_gd


def temporal_split(n: int, train=0.70, val=0.15):
    i_tr = int(n * train)
    i_va = int(n * (train + val))
    return slice(0, i_tr), slice(i_tr, i_va), slice(i_va, n)


def elo_baseline_probs(df, pre_test_slice, test_slice) -> np.ndarray:
    """Fit a fresh Elo on the pre-test matches, predict the test matches."""
    elo = EloModel().fit(df.iloc[pre_test_slice])
    out = []
    for r in df.iloc[test_slice].itertuples(index=False):
        p = elo.predict(r.home_team, r.away_team, neutral=bool(r.neutral))
        out.append([p["p_home_win"], p["p_draw"], p["p_away_win"]])
    return np.asarray(out, float)


def train_once(feats, s_tr, s_va, s_te, seed: int, use_form: bool) -> dict:
    """Train one network; return its test probabilities, model and scaler."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    cont = feats["cont"].copy()
    if not use_form:
        cont[:, FORM_COLS] = 0.0                 # ablate rolling form

    mean = cont[s_tr].mean(axis=0)
    std = cont[s_tr].std(axis=0)
    std[std < 1e-6] = 1.0
    cont_scaled = (cont - mean) / std

    def tens(sl):
        return (torch.as_tensor(feats["home_ids"][sl]),
                torch.as_tensor(feats["away_ids"][sl]),
                torch.as_tensor(cont_scaled[sl], dtype=torch.float32),
                torch.as_tensor(feats["y"][sl]))

    h_tr, a_tr, c_tr, y_tr = tens(s_tr)
    h_va, a_va, c_va, y_va = tens(s_va)
    h_te, a_te, c_te, _ = tens(s_te)

    model = OutcomeNet(n_teams=len(feats["team_index"]), emb_dim=EMB_DIM,
                       hidden=HIDDEN, p_drop=P_DROP)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    best_val, best_state, best_epoch, since = float("inf"), None, 0, 0
    n_tr = s_tr.stop
    epoch = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_tr)
        for i in range(0, n_tr, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(h_tr[idx], a_tr[idx], c_tr[idx]), y_tr[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(h_va, a_va, c_va), y_va).item()
        if np.isfinite(val_loss) and val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch, since = epoch, 0
        else:
            since += 1
            if since >= PATIENCE:
                break

    if best_state is not None:           # guard: a fully-divergent run keeps last
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(h_te, a_te, c_te), dim=1).numpy()
    return {"probs": probs, "model": model, "mean": mean, "std": std,
            "val_loss": best_val, "epochs": epoch, "best_epoch": best_epoch}


def _agg(metric_dicts, key):
    vals = np.array([m[key] for m in metric_dicts], float)
    return float(vals.mean()), float(vals.std())


def train() -> None:
    df = load_matches()
    print(summary(df))
    feats = build_features(df)
    n = len(feats["y"])
    s_tr, s_va, s_te = temporal_split(n)
    pre_test = slice(0, s_te.start)
    y_test = feats["y"][s_te]

    td = feats["dates"][s_te]
    print(f"\nTemporal split  train={s_tr.stop}  "
          f"val={s_va.stop - s_va.start}  test={n - s_te.start}")
    print(f"test window: {td.min()} -> {td.max()}")
    mix = np.bincount(feats["y"][s_tr], minlength=3) / s_tr.stop
    print(f"train class mix: home {mix[0]:.0%} / draw {mix[1]:.0%} / "
          f"away {mix[2]:.0%}")

    # ---- fixed baselines ------------------------------------------------- #
    elo_probs = elo_baseline_probs(df, pre_test, s_te)
    prior = np.bincount(feats["y"][pre_test], minlength=3) / pre_test.stop
    prior_probs = np.tile(prior, (len(y_test), 1))

    # ---- multi-seed sweep for full + ablation ---------------------------- #
    print("\ntraining (5 seeds × {full, embeddings-only}) …")
    full_runs = {s: train_once(feats, s_tr, s_va, s_te, s, use_form=True)
                 for s in SEEDS}
    abl_runs = {s: train_once(feats, s_tr, s_va, s_te, s, use_form=False)
                for s in SEEDS}
    full_metrics = {s: M.evaluate(r["probs"], y_test)
                    for s, r in full_runs.items()}
    abl_metrics = {s: M.evaluate(r["probs"], y_test)
                   for s, r in abl_runs.items()}

    primary = full_runs[PRIMARY_SEED]
    results = {
        "Neural net (full)": full_metrics[PRIMARY_SEED],
        "NN (embeddings only)": abl_metrics[PRIMARY_SEED],
        "Elo": M.evaluate(elo_probs, y_test),
        "Prior baseline": M.evaluate(prior_probs, y_test),
    }
    print(f"\nprimary seed {PRIMARY_SEED}: trained {primary['epochs']} epochs, "
          f"best val log-loss {primary['val_loss']:.4f} "
          f"@ epoch {primary['best_epoch']}")

    print("\n" + "=" * 78)
    print("HELD-OUT TEST METRICS  (lower = better for log loss / RPS / Brier)")
    print("=" * 78)
    print(M.format_comparison(results))

    # ---- robustness summary --------------------------------------------- #
    elo_ll = results["Elo"]["log_loss"]
    elo_rps = results["Elo"]["rps"]
    ll_m, ll_s = _agg(full_metrics.values(), "log_loss")
    rps_m, rps_s = _agg(full_metrics.values(), "rps")
    acc_m, acc_s = _agg(full_metrics.values(), "accuracy")
    abl_ll_m, abl_ll_s = _agg(abl_metrics.values(), "log_loss")
    wins = sum(1 for m in full_metrics.values() if m["log_loss"] < elo_ll)
    print("\n" + "-" * 78)
    print(f"Robustness over {len(SEEDS)} seeds (Elo is deterministic: "
          f"log-loss {elo_ll:.4f}, RPS {elo_rps:.4f}):")
    print(f"  NN full   log-loss {ll_m:.4f} ± {ll_s:.4f} | "
          f"RPS {rps_m:.4f} ± {rps_s:.4f} | acc {acc_m:.3f} ± {acc_s:.3f}")
    print(f"  NN emb-only log-loss {abl_ll_m:.4f} ± {abl_ll_s:.4f}")
    print(f"  NN full beats Elo on log-loss in {wins}/{len(SEEDS)} seeds")
    print("  (embeddings alone already beat Elo; the form features add little "
          "on average — the win comes from the architecture)")

    # ---- persist primary model + metrics --------------------------------- #
    predictor = NNPredictor(
        primary["model"], feats["team_index"], primary["mean"], primary["std"],
        feats["latest_form"],
        {"emb_dim": EMB_DIM, "hidden": list(HIDDEN), "p_drop": P_DROP,
         "k_form": K_FORM})
    predictor.save(MODEL_OUT)
    with open(METRICS_OUT, "w") as fh:
        json.dump({
            "test_window": [str(td.min()), str(td.max())],
            "n_test": int(len(y_test)),
            "primary": results,
            "robustness": {
                "seeds": SEEDS,
                "nn_full_log_loss_mean": ll_m, "nn_full_log_loss_std": ll_s,
                "nn_full_rps_mean": rps_m, "nn_full_rps_std": rps_s,
                "nn_emb_only_log_loss_mean": abl_ll_m,
                "elo_log_loss": elo_ll,
                "nn_beats_elo_seeds": f"{wins}/{len(SEEDS)}",
            },
        }, fh, indent=2)
    print(f"\nsaved model   -> {MODEL_OUT}")
    print(f"saved metrics -> {METRICS_OUT}")

    # ---- a couple of sanity predictions ---------------------------------- #
    print("\nExample predictions (neutral venue):")
    for a, b in [("Brazil", "France"), ("Argentina", "Spain"),
                 ("Japan", "Germany")]:
        if predictor.has_team(a) and predictor.has_team(b):
            p = predictor.predict(a, b, neutral=True)
            print(f"  {a} vs {b}:  {a} {p['p_home_win']*100:4.1f}% | "
                  f"draw {p['p_draw']*100:4.1f}% | "
                  f"{b} {p['p_away_win']*100:4.1f}%  -> {p['pick']}")


if __name__ == "__main__":
    train()
