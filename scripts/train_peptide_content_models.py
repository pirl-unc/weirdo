#!/usr/bin/env python3
"""Train variable-length peptide-content models on exact taxonomy membership.

This script compares a deployable composition MLP, a small residue CNN, a small
Transformer, and frozen ESM-2 8M embeddings.  All models consume the complete
8--15mer rather than averaging predictions over 8mer windows.  The outputs are
independent sigmoid membership scores.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from weirdo.peptide_content import summarize_peptide_content


ROOT = Path(__file__).resolve().parents[1]
DATA_SCRIPT = ROOT / "scripts" / "peptide_content_data.py"


def load_data_module(path: str | Path):
    """Import one split-builder module and validate its compact public API."""
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(f"weirdo_data_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not import split builder {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "CATEGORIES") or not hasattr(module, "load_split"):
        raise ValueError(f"Split builder {path} must define CATEGORIES and load_split().")
    return module


pcd = load_data_module(DATA_SCRIPT)
CATEGORIES = tuple(pcd.CATEGORIES)
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: index + 1 for index, aa in enumerate(AA)}
_MICROBIAL_CANDIDATES = ("viruses", "bacteria", "archaea", "fungi", "protists")
MICROBIAL_CATEGORIES = tuple(name for name in _MICROBIAL_CANDIDATES if name in CATEGORIES)


def set_data_module(module) -> None:
    """Switch model output shape and derived score groups to one split builder."""
    global pcd, CATEGORIES, MICROBIAL_CATEGORIES
    pcd = module
    CATEGORIES = tuple(module.CATEGORIES)
    MICROBIAL_CATEGORIES = tuple(
        name for name in _MICROBIAL_CANDIDATES if name in CATEGORIES
    )
    if not MICROBIAL_CATEGORIES or "mammals" not in CATEGORIES:
        raise ValueError(
            "Split builder categories must include 'mammals' and at least one microbial head."
        )


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _sha256_path(path: str | Path) -> str:
    """Hash a split file or a directory of split files deterministically."""
    path = Path(path)
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        digest.update(str(item.relative_to(path) if path.is_dir() else item.name).encode())
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def provenance(splits: str | Path) -> dict:
    versions = {}
    for package in ("numpy", "pandas", "scikit-learn", "torch", "transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {
        "split_sha256": _sha256_path(splits),
        "training_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_revision": revision,
        "dependency_versions": versions,
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def encode_peptides(peptides: list[str], max_length: int = 15):
    import torch

    tokens = torch.zeros((len(peptides), max_length), dtype=torch.long)
    mask = torch.zeros((len(peptides), max_length), dtype=torch.bool)
    for row, peptide in enumerate(peptides):
        length = min(len(peptide), max_length)
        if length:
            tokens[row, :length] = torch.tensor(
                [AA_TO_INDEX.get(aa, 0) for aa in peptide[:length]], dtype=torch.long
            )
            mask[row, :length] = True
    return tokens, mask


def _masked_pool(values, mask):
    import torch

    weights = mask.unsqueeze(-1).to(values.dtype)
    mean = (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    maximum = values.masked_fill(~mask.unsqueeze(-1), -torch.inf).max(dim=1).values
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    return torch.cat((mean, maximum), dim=1)


def build_torch_model(architecture: str, n_outputs: int, max_length: int = 15):
    import torch
    from torch import nn

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(len(AA_TO_INDEX) + 1, 48, padding_idx=0)
            self.convolutions = nn.ModuleList(
                nn.Conv1d(48, 64, kernel_size=k, padding=k // 2) for k in (3, 5, 7)
            )
            self.head = nn.Sequential(
                nn.Linear(64 * 3 * 2 + 1, 192),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(192, n_outputs),
            )

        def forward(self, tokens, mask):
            embedded = self.embedding(tokens).transpose(1, 2)
            pooled = []
            for convolution in self.convolutions:
                values = torch.nn.functional.gelu(convolution(embedded)).transpose(1, 2)
                pooled.append(_masked_pool(values, mask))
            length = mask.sum(dim=1, keepdim=True).to(embedded.dtype) / max_length
            return self.head(torch.cat((*pooled, length), dim=1))

    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            width = 96
            self.embedding = nn.Embedding(len(AA_TO_INDEX) + 1, width, padding_idx=0)
            self.position = nn.Parameter(torch.zeros(1, max_length, width))
            nn.init.normal_(self.position, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=width,
                nhead=4,
                dim_feedforward=256,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=3)
            self.head = nn.Sequential(
                nn.LayerNorm(width * 2 + 1),
                nn.Linear(width * 2 + 1, 192),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(192, n_outputs),
            )

        def forward(self, tokens, mask):
            values = self.embedding(tokens) + self.position[:, : tokens.shape[1]]
            values = self.encoder(values, src_key_padding_mask=~mask)
            pooled = _masked_pool(values, mask)
            length = mask.sum(dim=1, keepdim=True).to(values.dtype) / max_length
            return self.head(torch.cat((pooled, length), dim=1))

    if architecture == "cnn":
        return CNN()
    if architecture == "transformer":
        return Transformer()
    raise ValueError(f"Unknown Torch architecture {architecture!r}")


def build_esm_model(
    n_outputs: int,
    model_name: str,
    freeze: bool,
    *,
    encoder_config=None,
):
    from torch import nn
    from transformers import AutoModel

    class ESMHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = (
                AutoModel.from_config(encoder_config)
                if encoder_config is not None
                else AutoModel.from_pretrained(model_name)
            )
            width = int(self.encoder.config.hidden_size)
            self.head = nn.Sequential(
                nn.LayerNorm(width * 2 + 1),
                nn.Linear(width * 2 + 1, 256),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(256, n_outputs),
            )
            self.freeze_encoder = freeze
            if freeze:
                for parameter in self.encoder.parameters():
                    parameter.requires_grad = False

        def forward(self, input_ids, attention_mask, special_tokens_mask):
            if self.freeze_encoder:
                self.encoder.eval()
            values = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
            residue_mask = attention_mask.bool() & ~special_tokens_mask.bool()
            pooled = _masked_pool(values, residue_mask)
            length = residue_mask.sum(dim=1, keepdim=True).to(values.dtype) / 15.0
            return self.head(__import__("torch").cat((pooled, length), dim=1))

    return ESMHead()


class PeptideDataset:
    def __init__(self, peptides: list[str], labels: np.ndarray):
        self.peptides = peptides
        self.labels = np.asarray(labels, dtype=np.float32)

    def __len__(self):
        return len(self.peptides)

    def __getitem__(self, index):
        return self.peptides[index], self.labels[index]


class PeptideCollator:
    """Pickle-safe batch collation for multiprocessing data loaders."""

    def __init__(self, architecture: str, tokenizer=None):
        self.architecture = architecture
        self.tokenizer = tokenizer

    def __call__(self, batch):
        import torch

        peptides, labels = zip(*batch)
        if self.architecture.startswith("esm"):
            encoded = self.tokenizer(
                list(peptides),
                padding=True,
                return_tensors="pt",
                return_special_tokens_mask=True,
            )
            features = {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "special_tokens_mask": encoded["special_tokens_mask"],
            }
        else:
            tokens, mask = encode_peptides(list(peptides))
            features = {"tokens": tokens, "mask": mask}
        return features, torch.tensor(np.asarray(labels), dtype=torch.float32)


def make_collate(architecture: str, tokenizer=None):
    return PeptideCollator(architecture, tokenizer)


def predict_torch(model, loader, device):
    import torch

    model.eval()
    probabilities = []
    labels = []
    loss_total = 0.0
    element_count = 0
    criterion = torch.nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for features, target in loader:
            features = {key: value.to(device) for key, value in features.items()}
            target = target.to(device)
            logits = model(**features)
            loss_total += float(criterion(logits, target).cpu()) * target.numel()
            element_count += target.numel()
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(target.cpu().numpy())
    return (
        np.vstack(probabilities),
        np.vstack(labels),
        loss_total / max(element_count, 1),
    )


def train_torch(
    architecture: str,
    train: tuple[list[str], np.ndarray],
    val: tuple[list[str], np.ndarray],
    args,
    output: Path,
):
    import torch
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = None
    if architecture.startswith("esm"):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.esm_model)
        model = build_esm_model(
            len(CATEGORIES), args.esm_model, freeze=architecture == "esm_frozen"
        )
        batch_size = args.esm_batch_size
        learning_rate = (
            args.esm_learning_rate
            if architecture == "esm_frozen"
            else args.esm_finetune_learning_rate
        )
    else:
        model = build_torch_model(architecture, len(CATEGORIES))
        batch_size = args.batch_size
        learning_rate = args.learning_rate
    model.to(device)
    collate = make_collate(architecture, tokenizer)
    train_loader = DataLoader(
        PeptideDataset(*train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )
    val_loader = DataLoader(
        PeptideDataset(*val),
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_loss = math.inf
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        if architecture == "esm_frozen":
            model.encoder.eval()
        total_loss = 0.0
        seen = 0
        started = time.time()
        for features, target in train_loader:
            features = {key: value.to(device, non_blocking=True) for key, value in features.items()}
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(**features)
                loss = criterion(logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu()) * target.numel()
            seen += target.numel()
        _, _, val_loss = predict_torch(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(seen, 1),
            "val_loss": val_loss,
            "seconds": time.time() - started,
        }
        history.append(row)
        log(f"{architecture} epoch {epoch}: {row}")
        if val_loss < best_loss - args.min_delta:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "weights.pt")
    if architecture.startswith("esm"):
        tokenizer.save_pretrained(output / "tokenizer")
        model.encoder.config.save_pretrained(output / "encoder_config")
    config = {
        "format": "weirdo-peptide-content-v1",
        "architecture": architecture,
        "categories": list(CATEGORIES),
        "length_range": [8, 15],
        "esm_model": args.esm_model if architecture.startswith("esm") else None,
        "training": {
            "seed": getattr(args, "architecture_seed", args.seed),
            "epochs": len(history),
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "best_val_loss": best_loss,
            "history": history,
        },
        "provenance": getattr(args, "provenance", {}),
    }
    (output / "config.json").write_text(json.dumps(config, indent=2))
    return model, tokenizer, config


def make_loader(peptides, labels, architecture, tokenizer, batch_size, workers=0):
    from torch.utils.data import DataLoader

    return DataLoader(
        PeptideDataset(peptides, labels),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=make_collate(architecture, tokenizer),
    )


def expected_calibration_error(y_true, probabilities, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (probabilities >= low) & (
            probabilities <= high if high == 1.0 else probabilities < high
        )
        if selected.any():
            result += selected.mean() * abs(
                probabilities[selected].mean() - y_true[selected].mean()
            )
    return float(result)


def evaluate(probabilities: np.ndarray, labels: np.ndarray, peptides: list[str]) -> dict:
    result = {"n": len(labels), "categories": {}, "by_length": {}}
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    for index, category in enumerate(CATEGORIES):
        y = labels[:, index]
        p = clipped[:, index]
        result["categories"][category] = {
            "prevalence": float(y.mean()),
            "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            "average_precision": float(average_precision_score(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "brier": float(brier_score_loss(y, p)),
            "ece": expected_calibration_error(y, p),
        }
    aucs = [row["auc"] for row in result["categories"].values() if row["auc"] is not None]
    result["mean_auc"] = float(np.mean(aucs))

    index = {name: i for i, name in enumerate(CATEGORIES)}
    microbial_indices = [index[name] for name in MICROBIAL_CATEGORIES]
    content_scores = summarize_peptide_content(
        probabilities, CATEGORIES, microbial_categories=MICROBIAL_CATEGORIES
    )
    microbial_probability = content_scores["microbial_like"]
    microbial_truth = labels[:, microbial_indices].max(axis=1).astype(bool)
    mammal_truth = labels[:, index["mammals"]].astype(bool)
    eligible = microbial_truth ^ mammal_truth
    target = microbial_truth[eligible].astype(int)
    score = content_scores["microbial_foreign_like"]
    result["summary"] = {
        "definition": "max(virus,bacteria,archaea,fungi,protists) * (1 - mammals)",
        "n_microbial_only": int((microbial_truth & ~mammal_truth).sum()),
        "n_mammal_only": int((mammal_truth & ~microbial_truth).sum()),
        "n_shared_excluded": int((microbial_truth & mammal_truth).sum()),
        "microbial_foreign_like_auc": (
            float(roc_auc_score(target, score[eligible])) if len(np.unique(target)) == 2 else None
        ),
        "microbial_like_auc": (
            float(roc_auc_score(target, microbial_probability[eligible]))
            if len(np.unique(target)) == 2 else None
        ),
    }
    host_heads = {"mammal_host_viruses", "other_host_viruses"}
    if host_heads <= set(index):
        mammal_host_index = index["mammal_host_viruses"]
        other_host_index = index["other_host_viruses"]
        mammal_host_truth = labels[:, mammal_host_index].astype(bool)
        other_host_truth = labels[:, other_host_index].astype(bool)
        host_eligible = mammal_host_truth ^ other_host_truth
        host_target = mammal_host_truth[host_eligible].astype(int)
        mammal_host_probability = probabilities[:, mammal_host_index]
        other_host_probability = probabilities[:, other_host_index]
        host_margin = mammal_host_probability - other_host_probability
        result["viral_host_summary"] = {
            "n_mammal_host_only": int((mammal_host_truth & ~other_host_truth).sum()),
            "n_other_host_only": int((other_host_truth & ~mammal_host_truth).sum()),
            "n_shared_excluded": int((mammal_host_truth & other_host_truth).sum()),
            "mammal_head_auc_within_viruses": float(
                roc_auc_score(host_target, mammal_host_probability[host_eligible])
            ),
            "mammal_minus_other_auc_within_viruses": float(
                roc_auc_score(host_target, host_margin[host_eligible])
            ),
        }
    lengths = np.asarray([len(peptide) for peptide in peptides])
    for length in sorted(set(lengths.tolist())):
        selected = lengths == length
        if selected.sum() < 2:
            continue
        result["by_length"][str(length)] = evaluate_summary_only(
            probabilities[selected], labels[selected]
        )
    return result


def evaluate_summary_only(probabilities, labels):
    index = {name: i for i, name in enumerate(CATEGORIES)}
    microbial_indices = [index[name] for name in MICROBIAL_CATEGORIES]
    content_scores = summarize_peptide_content(
        probabilities, CATEGORIES, microbial_categories=MICROBIAL_CATEGORIES
    )
    microbial_truth = labels[:, microbial_indices].max(axis=1).astype(bool)
    mammal_truth = labels[:, index["mammals"]].astype(bool)
    eligible = microbial_truth ^ mammal_truth
    target = microbial_truth[eligible].astype(int)
    score = content_scores["microbial_foreign_like"]
    return {
        "n": int(eligible.sum()),
        "auc": float(roc_auc_score(target, score[eligible])) if len(np.unique(target)) == 2 else None,
    }


def train_composition(train, val, args, output):
    from weirdo.scorers import MLPScorer

    scorer = MLPScorer(
        k=15,
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        alpha=args.weight_decay,
        learning_rate_init=args.learning_rate,
        batch_size=args.batch_size,
        random_state=getattr(args, "architecture_seed", args.seed),
        early_stopping=True,
        loss="bce",
        positional=False,
        whole_peptide=True,
    )
    scorer.train(
        train[0],
        train[1],
        val_peptides=val[0],
        val_labels=val[1],
        target_categories=list(CATEGORIES),
        epochs=args.epochs,
        n_iter_no_change=args.patience,
        cache_features=True,
        verbose=True,
    )
    output.mkdir(parents=True, exist_ok=True)
    scorer.save(output)
    return scorer


def subset(split, n, seed):
    if n <= 0 or n >= len(split[0]):
        return split
    chosen = np.random.RandomState(seed).choice(len(split[0]), n, replace=False)
    return [split[0][i] for i in chosen], split[1][chosen]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--splits", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--data-script",
        default=str(DATA_SCRIPT),
        help="Split-builder Python module used to create --splits (default: %(default)s)",
    )
    parser.add_argument(
        "--architectures",
        default="composition_mlp,cnn,transformer,esm_frozen",
        help="comma-separated: composition_mlp, cnn, transformer, esm_frozen, esm_finetune",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--esm-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--esm-learning-rate", type=float, default=3e-4)
    parser.add_argument("--esm-finetune-learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-val", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--esm-model", default="facebook/esm2_t6_8M_UR50D")
    args = parser.parse_args()
    set_data_module(load_data_module(args.data_script))
    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    args.provenance = provenance(args.splits)
    train = subset(pcd.load_split(args.splits, "train"), args.max_train, args.seed)
    val = subset(pcd.load_split(args.splits, "val"), args.max_val, args.seed + 1)
    test = subset(pcd.load_split(args.splits, "test"), args.max_test, args.seed + 2)
    log(f"data sizes train={len(train[0]):,} val={len(val[0]):,} test={len(test[0]):,}")

    reports = {}
    architectures = [item.strip() for item in args.architectures.split(",") if item.strip()]
    for architecture in architectures:
        # Re-seed per architecture so changing the requested architecture
        # order cannot change the initialization of models that were already
        # in the run.  The offset is deterministic and independent of Python's
        # randomized string hash.
        architecture_offset = sum((index + 1) * ord(char) for index, char in enumerate(architecture))
        args.architecture_seed = args.seed + architecture_offset
        set_seed(args.architecture_seed)
        log(f"training {architecture}")
        model_out = out / architecture
        if architecture == "composition_mlp":
            model = train_composition(train, val, args, model_out)
            probabilities = model.predict_proba(test[0])
            config = {"architecture": architecture, "categories": list(CATEGORIES)}
        else:
            model, tokenizer, config = train_torch(
                architecture, train, val, args, model_out
            )
            import torch

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            test_loader = make_loader(
                test[0],
                test[1],
                architecture,
                tokenizer,
                (args.esm_batch_size if architecture.startswith("esm") else args.batch_size) * 2,
                args.workers,
            )
            probabilities, _, test_loss = predict_torch(model, test_loader, device)
            config["test_loss"] = test_loss
        report = evaluate(np.asarray(probabilities), test[1], test[0])
        report["config"] = config
        report["provenance"] = args.provenance
        reports[architecture] = report
        (model_out / "evaluation.json").write_text(json.dumps(report, indent=2))
        log(
            f"{architecture}: mean AUC={report['mean_auc']:.3f}; "
            f"microbial foreignness AUC="
            f"{report['summary']['microbial_foreign_like_auc']:.3f}"
        )
    summary = {
        "data": str(args.splits),
        "categories": list(CATEGORIES),
        "models": {
            name: {
                "mean_auc": report["mean_auc"],
                "microbial_foreign_like_auc": report["summary"]["microbial_foreign_like_auc"],
                "microbial_like_auc": report["summary"]["microbial_like_auc"],
            }
            for name, report in reports.items()
        },
        "provenance": args.provenance,
    }
    (out / "comparison.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
