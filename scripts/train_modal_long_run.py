#!/usr/bin/env python3
"""Train WEIRDO for a long run on Modal and export learned weights.

This script trains `MLPScorer` remotely, saves the model directory to a Modal
volume, and packages it as a `.tar.gz` artifact that can be distributed.

Examples
--------
Seed full SwissProt data once into the Modal data volume:
    modal volume put weirdo-data-cache data/swissprot-8mers.csv downloads/swissprot-8mers.csv --force

Run long training remotely:
    modal run scripts/train_modal_long_run.py --model-name swissprot-mlp-modal --epochs 1000

Download resulting weights to a local file:
    modal run scripts/train_modal_long_run.py --model-name swissprot-mlp-modal --output-archive ./swissprot-mlp-modal.tar.gz
"""

from __future__ import annotations

import json
import modal


APP_NAME = "weirdo-train-long-run"
ARTIFACTS_VOLUME_NAME = "weirdo-model-artifacts"
DATA_CACHE_VOLUME_NAME = "weirdo-data-cache"
WEIRDO_GIT_REF = "60d88139b2aa880108060d7b9dccecfb441151cc"

DEFAULT_TARGET_CATEGORIES = (
    "archaea,bacteria,fungi,human,invertebrates,mammals,plants,rodents,vertebrates,viruses"
)
DEFAULT_SWISSPROT_PATH = "/root/.weirdo/downloads/swissprot-8mers.csv"


def _build_image() -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git")
        .run_commands("python -m pip install --upgrade pip")
        .run_commands(
            "git clone https://github.com/pirl-unc/weirdo.git /root/weirdo "
            f"&& cd /root/weirdo && git checkout {WEIRDO_GIT_REF}"
        )
        .workdir("/root/weirdo")
        .run_commands("python -m pip install .")
    )


image = _build_image()
app = modal.App(APP_NAME, image=image)

artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)
data_cache_volume = modal.Volume.from_name(DATA_CACHE_VOLUME_NAME, create_if_missing=True)


@app.function(
    cpu=16,
    memory=65536,
    timeout=12 * 60 * 60,
    volumes={
        "/artifacts": artifacts_volume,
        "/root/.weirdo": data_cache_volume,
    },
)
def train_remote(
    model_name: str,
    max_samples: int = 0,
    epochs: int = 500,
    learning_rate: float = 1e-3,
    hidden_layers_csv: str = "256,128,64",
    target_categories_csv: str = DEFAULT_TARGET_CATEGORIES,
    swissprot_path: str = DEFAULT_SWISSPROT_PATH,
    seed: int = 42,
) -> dict[str, object]:
    """Train model remotely and persist model + archive in volume.

    Use ``max_samples=0`` to train on all available rows.
    """
    from pathlib import Path
    import tarfile

    from weirdo.model_manager import ModelManager
    from weirdo.scorers import MLPScorer, SwissProtReference

    target_categories = [x.strip() for x in target_categories_csv.split(",") if x.strip()]
    hidden_layers = tuple(int(x.strip()) for x in hidden_layers_csv.split(",") if x.strip())

    data_path = Path(swissprot_path)
    if not data_path.exists():
        raise FileNotFoundError(
            "SwissProt CSV not found in Modal volume at "
            f"{data_path}. Seed it once with:\n"
            "  modal volume put weirdo-data-cache data/swissprot-8mers.csv "
            "downloads/swissprot-8mers.csv --force"
        )

    scorer = MLPScorer(
        k=8,
        hidden_layer_sizes=hidden_layers,
        random_state=seed,
    )
    if max_samples <= 0:
        # Stream all rows from CSV to keep memory bounded for full SwissProt runs.
        ref = SwissProtReference(
            data_path=str(data_path),
            auto_download=False,
            lazy=True,
        ).load()

        def row_iterator_factory():
            for kmer, cats in ref.iter_kmers_with_categories():
                yield kmer, [1.0 if cats.get(cat, False) else 0.0 for cat in target_categories]

        scorer.train_streaming(
            row_iterator_factory=row_iterator_factory,
            target_categories=target_categories,
            epochs=epochs,
            learning_rate=learning_rate,
            verbose=True,
        )
    else:
        ref = SwissProtReference(
            data_path=str(data_path),
            auto_download=False,
            lazy=True,
        ).load()
        peptides, labels = ref.get_training_data(
            target_categories=target_categories,
            multi_label=True,
            max_samples=max_samples,
            shuffle=True,
            seed=seed,
        )
        scorer.train(
            peptides=peptides,
            labels=labels,
            target_categories=target_categories,
            epochs=epochs,
            learning_rate=learning_rate,
            verbose=True,
        )

    manager = ModelManager(model_dir=Path("/artifacts/models"))
    saved_model_dir = manager.save(scorer, name=model_name, overwrite=True)

    archive_path = Path("/artifacts") / f"{model_name}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(saved_model_dir, arcname=model_name)

    metadata = {
        "model_name": model_name,
        "saved_model_dir": str(saved_model_dir),
        "archive_path": str(archive_path),
        "swissprot_path": str(data_path),
        "n_train": scorer._metadata.get("n_train"),
        "n_epochs": scorer._metadata.get("n_epochs"),
        "final_train_loss": scorer._metadata.get("final_train_loss"),
        "final_val_loss": scorer._metadata.get("final_val_loss"),
        "training_history": scorer.training_history,
    }
    with open(Path("/artifacts") / f"{model_name}.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Ensure files are visible to other functions in the same app run.
    artifacts_volume.commit()

    return metadata


@app.function(
    timeout=30 * 60,
    volumes={"/artifacts": artifacts_volume},
)
def read_archive_bytes(model_name: str) -> bytes:
    """Read archived model bytes from volume."""
    from pathlib import Path

    # Pull latest writes from other containers before reading.
    artifacts_volume.reload()

    archive_path = Path("/artifacts") / f"{model_name}.tar.gz"
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    return archive_path.read_bytes()


@app.local_entrypoint()
def main(
    model_name: str = "swissprot-mlp-modal",
    max_samples: int = 0,
    epochs: int = 500,
    learning_rate: float = 1e-3,
    hidden_layers: str = "256,128,64",
    target_categories: str = DEFAULT_TARGET_CATEGORIES,
    swissprot_path: str = DEFAULT_SWISSPROT_PATH,
    seed: int = 42,
    output_archive: str = "",
):
    """Run remote training and optionally pull model weights locally."""
    result = train_remote.remote(
        model_name=model_name,
        max_samples=max_samples,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_layers_csv=hidden_layers,
        target_categories_csv=target_categories,
        swissprot_path=swissprot_path,
        seed=seed,
    )
    print(json.dumps(result, indent=2))

    if output_archive:
        from pathlib import Path

        model_bytes = read_archive_bytes.remote(model_name=model_name)
        output_path = Path(output_archive)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(model_bytes)
        print(f"Wrote model archive to: {output_path}")
