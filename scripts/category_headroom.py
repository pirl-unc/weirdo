#!/usr/bin/env python3
"""Headroom checks for the species-category task on the validation fold.

Is the modest accuracy of the shipped MLP a limit of the model or of the input? Each
subcommand trains a different learner on the family-aware splits from
``category_data.py`` and scores it on the validation fold (the test fold is never read):

* ``mlp`` - scikit-learn MLPs on the 592 composition features: the old squared loss with a
  sigmoid, cross-entropy, cross-entropy plus per-position one-hot residues, and a wider net.
* ``cnn`` - a small PyTorch CNN over the 8 residues (needs ``torch``).
* ``gbm`` - gradient boosting (``HistGradientBoostingClassifier``) on the 592 features for
  selected categories.

Example::

    python scripts/category_headroom.py mlp --splits data/category-splits --out mlp.json
    python scripts/category_headroom.py cnn --splits data/category-splits --out cnn.json
    python scripts/category_headroom.py gbm --splits data/category-splits --out gbm.json
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import category_eval as ce  # noqa: E402
from category_data import CATEGORIES, load_split  # noqa: E402

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
_RESIDUE_INDEX = np.zeros(256, dtype=np.int64)
for _i, _aa in enumerate(AMINO_ACIDS):
    _RESIDUE_INDEX[ord(_aa)] = _i


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def residue_codes(kmers):
    """(n, k) integer residue indices of equal-length k-mers."""
    return _RESIDUE_INDEX[np.frombuffer("".join(kmers).encode(), dtype=np.uint8)].reshape(len(kmers), -1)


def one_hot(kmers):
    codes = residue_codes(kmers)
    out = np.zeros((len(kmers), codes.shape[1], 20), dtype=np.float32)
    out[np.arange(len(kmers))[:, None], np.arange(codes.shape[1])[None, :], codes] = 1.0
    return out.reshape(len(kmers), -1)


def composition_features(kmers):
    from weirdo.scorers.peptide_features import extract_features_batch

    return extract_features_batch(kmers, 8, True)


def summarize(P, Y, prevalence):
    """Mean and per-category AUC, log loss, and ECE for probability matrix P."""
    auc = ce.auc_per_category(P, Y)
    loss = ce.log_loss_per_category(P, Y)
    baseline = ce.prevalence_log_loss(Y, prevalence)
    ece = [ce.expected_calibration_error(P[:, j], Y[:, j]) for j in range(Y.shape[1])]
    return {
        "mean_auc": float(np.mean(auc)), "mean_log_loss": float(np.mean(loss)),
        "mean_log_loss_prevalence": float(np.mean(baseline)), "mean_ece": float(np.mean(ece)),
        "auc": {c: float(a) for c, a in zip(CATEGORIES, auc)},
    }


def run_mlp(args):
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.preprocessing import StandardScaler

    train, Ytr = load_split(args.splits, "train", args.train_n, seed=args.seed)
    val, Yva = load_split(args.splits, "val", args.val_n, seed=args.seed)
    prevalence = Ytr.mean(axis=0)
    base_train = composition_features(train)
    base_val = composition_features(val)
    scaler = StandardScaler().fit(base_train)
    base_train, base_val = scaler.transform(base_train).astype(np.float32), scaler.transform(base_val).astype(np.float32)
    onehot_train, onehot_val = one_hot(train), one_hot(val)
    results = []
    for name in args.variants.split(","):
        use_position = "pos" in name
        hidden = (512, 256, 128) if name.endswith("wide") else (256, 128, 64)
        X_train = np.hstack([base_train, onehot_train]) if use_position else base_train
        X_val = np.hstack([base_val, onehot_val]) if use_position else base_val
        legacy = name == "legacy_mse"
        kwargs = dict(hidden_layer_sizes=hidden, learning_rate_init=1e-3, alpha=1e-4, max_iter=1,
                      shuffle=False, warm_start=True, random_state=args.seed)
        model = MLPRegressor(**kwargs) if legacy else MLPClassifier(**kwargs)
        rng = np.random.RandomState(args.seed)
        best, bad, best_state, history = np.inf, 0, None, []
        for epoch in range(1, args.epochs + 1):
            order = rng.permutation(len(X_train))
            for start in range(0, len(order), args.batch_size):
                index = order[start:start + args.batch_size]
                if legacy:
                    model.partial_fit(X_train[index], Ytr[index])
                else:
                    model.partial_fit(X_train[index], Ytr[index], classes=np.arange(Ytr.shape[1]))
            raw = np.vstack([
                model.predict(X_val[s:s + 50000]) if legacy else model.predict_proba(X_val[s:s + 50000])
                for s in range(0, len(X_val), 50000)
            ])
            P = 1.0 / (1.0 + np.exp(-raw)) if legacy else raw  # the old head squashed regression output
            stats = summarize(P, Yva, prevalence)
            criterion = -stats["mean_auc"] if legacy else stats["mean_log_loss"]
            history.append({"epoch": epoch, **{k: stats[k] for k in ("mean_auc", "mean_log_loss", "mean_ece")}})
            log(f"{name} epoch {epoch}: AUC {stats['mean_auc']:.4f} log loss {stats['mean_log_loss']:.4f} "
                f"(base rate {stats['mean_log_loss_prevalence']:.4f}) ECE {stats['mean_ece']:.4f}")
            if criterion < best - 1e-5:
                best, bad, best_stats = criterion, 0, stats
                best_state = (copy.deepcopy(model.coefs_), copy.deepcopy(model.intercepts_))
            else:
                bad += 1
                if bad >= args.patience:
                    break
        results.append({"variant": name, "hidden": hidden, "train_n": len(train), "best_validation": best_stats,
                        "history": history})
        Path(args.out).write_text(json.dumps(results, indent=1))


def run_cnn(args):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self, n_out, emb=32, ch=128, hidden=256):
            super().__init__()
            self.emb = nn.Embedding(20, emb)
            self.conv1 = nn.Conv1d(emb, ch, 3, padding=1)
            self.conv2 = nn.Conv1d(ch, ch, 3, padding=1)
            self.head = nn.Sequential(nn.Linear(2 * ch + 8 * emb, hidden), nn.ReLU(), nn.Dropout(0.1),
                                      nn.Linear(hidden, n_out))

        def forward(self, x):
            e = self.emb(x)
            h = F.relu(self.conv2(F.relu(self.conv1(e.transpose(1, 2)))))
            return self.head(torch.cat([h.max(-1).values, h.mean(-1), e.flatten(1)], 1))

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(args.seed)
    train, Ytr = load_split(args.splits, "train", args.train_n, seed=args.seed)
    val, Yva = load_split(args.splits, "val", args.val_n, seed=args.seed)
    X_train, X_val = torch.from_numpy(residue_codes(train)), torch.from_numpy(residue_codes(val))
    Y_train = torch.from_numpy(Ytr)
    prevalence = Ytr.mean(axis=0)
    model = Net(len(CATEGORIES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history, best = [], None
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(X_train))
        for start in range(0, len(order), args.batch_size):
            index = order[start:start + args.batch_size]
            loss = F.binary_cross_entropy_with_logits(model(X_train[index].to(device)), Y_train[index].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = torch.cat([model(X_val[s:s + 20000].to(device)).cpu() for s in range(0, len(X_val), 20000)])
        stats = summarize(torch.sigmoid(logits).numpy().astype(np.float64), Yva, prevalence)
        history.append({"epoch": epoch, **{k: stats[k] for k in ("mean_auc", "mean_log_loss", "mean_ece")}})
        log(f"cnn epoch {epoch}: AUC {stats['mean_auc']:.4f} log loss {stats['mean_log_loss']:.4f}")
        if best is None or stats["mean_log_loss"] < best["mean_log_loss"]:
            best = stats
    Path(args.out).write_text(json.dumps({"train_n": len(train), "best_validation": best, "history": history}, indent=1))


def run_gbm(args):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import log_loss, roc_auc_score

    train, Ytr = load_split(args.splits, "train", args.train_n, seed=args.seed)
    val, Yva = load_split(args.splits, "val", args.val_n, seed=args.seed)
    X_train = composition_features(train)
    X_val = composition_features(val)
    results = {}
    for name in args.categories.split(","):
        j = CATEGORIES.index(name)
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                                               early_stopping=True, validation_fraction=0.1,
                                               n_iter_no_change=15, random_state=args.seed)
        model.fit(X_train, Ytr[:, j])
        p = model.predict_proba(X_val)[:, 1]
        results[name] = {
            "auc": float(roc_auc_score(Yva[:, j], p)), "log_loss": float(log_loss(Yva[:, j], p)),
            "log_loss_prevalence": float(log_loss(Yva[:, j], np.full(len(p), Ytr[:, j].mean()))),
            "iterations": int(model.n_iter_),
        }
        log(f"gbm {name}: {json.dumps(results[name])}")
        Path(args.out).write_text(json.dumps({"train_n": len(train), "categories": results}, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func, train_n, epochs in (("mlp", run_mlp, 1_000_000, 20), ("cnn", run_cnn, 3_000_000, 8),
                                        ("gbm", run_gbm, 500_000, 0)):
        command = sub.add_parser(name)
        command.add_argument("--splits", required=True)
        command.add_argument("--out", required=True)
        command.add_argument("--train-n", type=int, default=train_n)
        command.add_argument("--val-n", type=int, default=200_000 if name != "cnn" else 300_000)
        command.add_argument("--seed", type=int, default=0)
        command.set_defaults(func=func)
        if name in ("mlp", "cnn"):
            command.add_argument("--epochs", type=int, default=epochs)
            command.add_argument("--batch-size", type=int, default=1024 if name == "mlp" else 2048)
        if name == "mlp":
            command.add_argument("--patience", type=int, default=3)
            command.add_argument("--variants", default="legacy_mse,bce_592,bce_592_pos,bce_592_pos_wide")
        if name == "cnn":
            command.add_argument("--lr", type=float, default=2e-3)
        if name == "gbm":
            command.add_argument("--categories", default="viruses,bacteria,human,plants")
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
