#!/usr/bin/env python3
"""Build and train the variable-length peptide-content models on Modal.

The raw SwissProt ``.dat.gz`` files live in the existing ``weirdo-data-cache``
volume.  Model artifacts and reports are written to ``weirdo-model-artifacts``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import modal


APP_NAME = "weirdo-peptide-content"
DATA_VOLUME_NAME = "weirdo-data-cache"
ARTIFACT_VOLUME_NAME = "weirdo-model-artifacts"
REMOTE_ROOT = "/root/weirdo"
REMOTE_DAT = "/data/downloads/swissprot-dat-files"
REMOTE_SPLITS = "/data/peptide-content-splits-v1"
REMOTE_MODELS = "/artifacts/peptide-content-v1"
REMOTE_EVALUATION = "/artifacts/peptide-content-v1/immunogenicity"
REMOTE_REFERENCE_DATA = "/data/weirdo/downloads/reference-proteomes"
REMOTE_REFERENCE_SPLITS = "/data/reference-proteome-splits-v1"
REMOTE_REFERENCE_MODELS = "/artifacts/reference-proteome-v1"
REMOTE_REFERENCE_EVALUATION = "/artifacts/reference-proteome-v1/tcell-evaluation"
REMOTE_TRANSFER_SCRIPT = "scripts/evaluate_tcell_transfer.py"
REMOTE_VIRAL_BENCHMARK = f"{REMOTE_ROOT}/data/viral-tcell-benchmark/scored-assays.csv"
REMOTE_HOST_SPLITS = "/data/viral-host-peptide-content-splits-v1"
REMOTE_HOST_MODELS = "/artifacts/viral-host-peptide-content-v1"
REMOTE_HOST_EVALUATION = "/artifacts/viral-host-peptide-content-v1/improve"
REMOTE_NCBI_TAXONOMY = "/data/downloads/ncbi-taxonomy"
REMOTE_IMPROVE = (
    f"{REMOTE_ROOT}/data/improve-validation/"
    "02_2_Validated_neoepitopes_calculated_features_TME.txt"
)
REMOTE_IMPROVE_PUBLISHED = (
    f"{REMOTE_ROOT}/data/improve-validation/neoepitopes_predictions.tsv"
)

LOCAL_ROOT = Path(__file__).resolve().parents[1]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy>=2,<3",
        "scipy>=1.14",
        "pandas>=2.2",
        "scikit-learn>=1.7,<2",
        "torch>=2.4",
        "transformers>=4.48,<6",
        "pyahocorasick>=2,<3",
        "datacache>=1.24",
    )
    .add_local_dir(LOCAL_ROOT / "weirdo", f"{REMOTE_ROOT}/weirdo", copy=True)
    .add_local_file(
        LOCAL_ROOT / "scripts" / "peptide_content_data.py",
        f"{REMOTE_ROOT}/scripts/peptide_content_data.py",
        copy=True,
    )
    .add_local_file(
        LOCAL_ROOT / "scripts" / "viral_host_peptide_content_data.py",
        f"{REMOTE_ROOT}/scripts/viral_host_peptide_content_data.py",
        copy=True,
    )
    .add_local_file(
        LOCAL_ROOT / "scripts" / "reference_proteome_data.py",
        f"{REMOTE_ROOT}/scripts/reference_proteome_data.py",
        copy=True,
    )
    .add_local_file(
        LOCAL_ROOT / "scripts" / "analyze_taxonomy_buckets.py",
        f"{REMOTE_ROOT}/scripts/analyze_taxonomy_buckets.py",
        copy=True,
    )
    .add_local_file(
        LOCAL_ROOT / "scripts" / "train_peptide_content_models.py",
        f"{REMOTE_ROOT}/scripts/train_peptide_content_models.py",
        copy=True,
    )
    .add_local_file(
        LOCAL_ROOT / "scripts" / "evaluate_peptide_content_immunogenicity.py",
        f"{REMOTE_ROOT}/scripts/evaluate_peptide_content_immunogenicity.py",
        copy=True,
    )
    .add_local_file(
        LOCAL_ROOT / "scripts" / "evaluate_tcell_transfer.py",
        f"{REMOTE_ROOT}/scripts/evaluate_tcell_transfer.py",
        copy=True,
    )
    .add_local_file(
        LOCAL_ROOT / "scripts" / "evaluate_improve_foreignness.py",
        f"{REMOTE_ROOT}/scripts/evaluate_improve_foreignness.py",
        copy=True,
    )
    .add_local_file(
        LOCAL_ROOT / "data" / "foreignness-choices" / "dev-scored-assays.csv",
        f"{REMOTE_ROOT}/data/foreignness-choices/dev-scored-assays.csv",
        copy=True,
    )
    .add_local_file(
        LOCAL_ROOT / "data" / "viral-tcell-benchmark" / "scored-assays.csv",
        f"{REMOTE_ROOT}/data/viral-tcell-benchmark/scored-assays.csv",
        copy=True,
    )
    .add_local_file(
        LOCAL_ROOT / "data" / "foreignness-choices" / "dev2-scored-assays.csv",
        f"{REMOTE_ROOT}/data/foreignness-choices/dev2-scored-assays.csv",
        copy=True,
    )
    .add_local_file(
        LOCAL_ROOT / "data" / "foreignness-choices" / "dev3-scored-assays.csv",
        f"{REMOTE_ROOT}/data/foreignness-choices/dev3-scored-assays.csv",
        copy=True,
    )
    .add_local_file(
        LOCAL_ROOT / "data" / "viral-tcell-benchmark" / "iedb-31511069-page-000.json",
        f"{REMOTE_ROOT}/data/viral-tcell-benchmark/iedb-31511069-page-000.json",
        copy=True,
    )
    .add_local_file(
        LOCAL_ROOT / "data" / "viral-tcell-benchmark" / "iedb-33038342-page-000.json",
        f"{REMOTE_ROOT}/data/viral-tcell-benchmark/iedb-33038342-page-000.json",
        copy=True,
    )
    .add_local_file(
        LOCAL_ROOT
        / "data"
        / "improve-validation"
        / "02_2_Validated_neoepitopes_calculated_features_TME.txt",
        REMOTE_IMPROVE,
        copy=True,
    )
    .add_local_file(
        LOCAL_ROOT / "data" / "improve-validation" / "neoepitopes_predictions.tsv",
        REMOTE_IMPROVE_PUBLISHED,
        copy=True,
    )
)

app = modal.App(APP_NAME, image=image)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)


def _ensure_ncbi_taxonomy() -> None:
    directory = Path(REMOTE_NCBI_TAXONOMY)
    nodes = directory / "nodes.dmp"
    merged = directory / "merged.dmp"
    if nodes.is_file() and merged.is_file():
        return
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "taxdmp.zip"
    urllib.request.urlretrieve(
        "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdmp.zip", archive
    )
    with zipfile.ZipFile(archive) as bundle:
        bundle.extract("nodes.dmp", directory)
        bundle.extract("merged.dmp", directory)


@app.function(
    cpu=8,
    memory=32768,
    timeout=60 * 60,
    volumes={"/data": data_volume},
)
def summarize_data() -> dict:
    sys.path.insert(0, f"{REMOTE_ROOT}/scripts")
    import peptide_content_data as pcd

    return pcd.summarize(pcd.load_proteins(Path(REMOTE_DAT)))


@app.function(
    cpu=8,
    memory=32768,
    timeout=60 * 60,
    volumes={"/data": data_volume},
)
def analyze_taxonomy_buckets() -> dict:
    sys.path.insert(0, f"{REMOTE_ROOT}/scripts")
    import analyze_taxonomy_buckets as analysis

    return analysis.analyze(Path(REMOTE_DAT))


@app.function(
    cpu=16,
    memory=65536,
    timeout=8 * 60 * 60,
    volumes={"/data": data_volume},
)
def build_data(
    train_per_stratum_length: int = 10_000,
    val_per_stratum_length: int = 2_000,
    test_per_stratum_length: int = 2_000,
    seed: int = 42,
) -> dict:
    command = [
        "python",
        "scripts/peptide_content_data.py",
        "build",
        "--dat-dir",
        REMOTE_DAT,
        "--out",
        REMOTE_SPLITS,
        "--train-per-stratum-length",
        str(train_per_stratum_length),
        "--val-per-stratum-length",
        str(val_per_stratum_length),
        "--test-per-stratum-length",
        str(test_per_stratum_length),
        "--seed",
        str(seed),
    ]
    subprocess.run(command, cwd=REMOTE_ROOT, check=True)
    data_volume.commit()
    return json.loads((Path(REMOTE_SPLITS) / "metadata.json").read_text())


@app.function(
    cpu=16,
    memory=65536,
    timeout=8 * 60 * 60,
    volumes={"/data": data_volume},
)
def build_viral_host_data(
    train_per_stratum_length: int = 10_000,
    val_per_stratum_length: int = 2_000,
    test_per_stratum_length: int = 2_000,
    seed: int = 42,
) -> dict:
    _ensure_ncbi_taxonomy()
    command = [
        "python",
        "scripts/viral_host_peptide_content_data.py",
        "build",
        "--dat-dir",
        REMOTE_DAT,
        "--taxonomy-dir",
        REMOTE_NCBI_TAXONOMY,
        "--out",
        REMOTE_HOST_SPLITS,
        "--train-per-stratum-length",
        str(train_per_stratum_length),
        "--val-per-stratum-length",
        str(val_per_stratum_length),
        "--test-per-stratum-length",
        str(test_per_stratum_length),
        "--seed",
        str(seed),
    ]
    subprocess.run(command, cwd=REMOTE_ROOT, check=True)
    data_volume.commit()
    return json.loads((Path(REMOTE_HOST_SPLITS) / "metadata.json").read_text())


@app.function(
    cpu=8,
    memory=32768,
    timeout=12 * 60 * 60,
    volumes={"/data": data_volume},
    env={"WEIRDO_DATA_DIR": "/data/weirdo", "PYTHONPATH": REMOTE_ROOT},
)
def download_reference_proteomes(
    manifest: str = "reference-proteomes-v1",
    groups: str = "",
) -> dict:
    """Fetch a frozen complete-reference corpus into the persistent volume."""
    command = [
        "python", "-m", "weirdo.cli", "data", "reference-proteome", "--manifest", manifest
    ]
    selected_groups = [item.strip() for item in groups.split(",") if item.strip()]
    for group in selected_groups:
        command.extend(("--group", group))
    subprocess.run(command, cwd=REMOTE_ROOT, check=True)
    data_volume.commit()
    return {
        "manifest": manifest,
        "groups": selected_groups,
        "reference_dir": REMOTE_REFERENCE_DATA,
    }


@app.function(
    cpu=1,
    memory=1024,
    timeout=10 * 60,
    volumes={"/data": data_volume},
)
def summarize_reference_download(manifest: str = "reference-proteomes-v1") -> dict:
    """Return exact totals from the checksum-verified manifest download record."""
    metadata = json.loads((Path("/data/weirdo") / "metadata.json").read_text())
    records = [
        value
        for key, value in metadata.get("downloads", {}).items()
        if key.startswith("reference-proteome:")
    ]
    if not records:
        raise FileNotFoundError(
            f"No reference proteomes found; run the {manifest!r} download first."
        )

    def totals(selected):
        return {
            "proteomes": len(selected),
            "proteins": sum(record["n_proteins"] for record in selected),
            "amino_acids": sum(record["amino_acids"] for record in selected),
            "compressed_bytes": sum(record["size_bytes"] for record in selected),
        }

    viruses = [record for record in records if record["taxon_id"] == 10239]
    mammals = [record for record in records if record["taxon_id"] != 10239]
    return {
        "manifest": manifest,
        "total": totals(records),
        "mammals": totals(mammals),
        "viruses": totals(viruses),
    }


@app.function(
    cpu=16,
    memory=65536,
    timeout=12 * 60 * 60,
    volumes={"/data": data_volume},
    env={"WEIRDO_DATA_DIR": "/data/weirdo", "PYTHONPATH": REMOTE_ROOT},
)
def build_reference_data(
    manifest: str = "reference-proteomes-v1",
    train_per_stratum_length: int = 10_000,
    val_per_stratum_length: int = 2_000,
    test_per_stratum_length: int = 2_000,
    seed: int = 42,
) -> dict:
    """Build variable-length exact membership labels from every selected FASTA."""
    command = [
        "python",
        "scripts/reference_proteome_data.py",
        "build",
        "--reference-dir",
        REMOTE_REFERENCE_DATA,
        "--manifest",
        manifest,
        "--out",
        REMOTE_REFERENCE_SPLITS,
        "--train-per-stratum-length",
        str(train_per_stratum_length),
        "--val-per-stratum-length",
        str(val_per_stratum_length),
        "--test-per-stratum-length",
        str(test_per_stratum_length),
        "--seed",
        str(seed),
        "--exclude-peptide-csv",
        REMOTE_VIRAL_BENCHMARK,
        "--exclude-role",
        "validation",
    ]
    subprocess.run(command, cwd=REMOTE_ROOT, check=True)
    data_volume.commit()
    return json.loads((Path(REMOTE_REFERENCE_SPLITS) / "metadata.json").read_text())


@app.function(
    cpu=8,
    memory=65536,
    gpu="A10G",
    timeout=12 * 60 * 60,
    volumes={"/data": data_volume, "/artifacts": artifact_volume},
    env={"HF_HOME": "/data/huggingface", "PYTHONPATH": REMOTE_ROOT},
)
def train_models(
    architectures: str = "composition_mlp,cnn,transformer,esm_frozen",
    epochs: int = 20,
    seed: int = 42,
) -> dict:
    command = [
        "python",
        "scripts/train_peptide_content_models.py",
        "--splits",
        REMOTE_SPLITS,
        "--out",
        REMOTE_MODELS,
        "--architectures",
        architectures,
        "--epochs",
        str(epochs),
        "--seed",
        str(seed),
        "--workers",
        "4",
    ]
    subprocess.run(command, cwd=REMOTE_ROOT, check=True)
    data_volume.commit()
    artifact_volume.commit()
    return json.loads((Path(REMOTE_MODELS) / "comparison.json").read_text())


@app.function(
    cpu=8,
    memory=65536,
    gpu="A10G",
    timeout=12 * 60 * 60,
    volumes={"/data": data_volume, "/artifacts": artifact_volume},
    env={"HF_HOME": "/data/huggingface", "PYTHONPATH": REMOTE_ROOT},
)
def train_viral_host_models(
    architectures: str = "esm_finetune",
    epochs: int = 20,
    seed: int = 42,
) -> dict:
    command = [
        "python",
        "scripts/train_peptide_content_models.py",
        "--data-script",
        "scripts/viral_host_peptide_content_data.py",
        "--splits",
        REMOTE_HOST_SPLITS,
        "--out",
        REMOTE_HOST_MODELS,
        "--architectures",
        architectures,
        "--epochs",
        str(epochs),
        "--seed",
        str(seed),
        "--workers",
        "4",
    ]
    subprocess.run(command, cwd=REMOTE_ROOT, check=True)
    data_volume.commit()
    artifact_volume.commit()
    return json.loads((Path(REMOTE_HOST_MODELS) / "comparison.json").read_text())


@app.function(
    cpu=8,
    memory=65536,
    gpu="A10G",
    timeout=12 * 60 * 60,
    volumes={"/data": data_volume, "/artifacts": artifact_volume},
    env={
        "HF_HOME": "/data/huggingface",
        "WEIRDO_DATA_DIR": "/data/weirdo",
        "PYTHONPATH": REMOTE_ROOT,
    },
)
def train_reference_models(
    architectures: str = "composition_mlp,cnn,esm_frozen",
    epochs: int = 20,
    seed: int = 42,
) -> dict:
    """Train on the complete frozen reference-proteome corpus."""
    comparison_path = Path(REMOTE_REFERENCE_MODELS) / "comparison.json"
    previous = (
        json.loads(comparison_path.read_text()) if comparison_path.exists() else None
    )
    command = [
        "python",
        "scripts/train_peptide_content_models.py",
        "--data-script",
        "scripts/reference_proteome_data.py",
        "--splits",
        REMOTE_REFERENCE_SPLITS,
        "--out",
        REMOTE_REFERENCE_MODELS,
        "--architectures",
        architectures,
        "--epochs",
        str(epochs),
        "--seed",
        str(seed),
        "--workers",
        "4",
    ]
    subprocess.run(command, cwd=REMOTE_ROOT, check=True)
    # Training a subset (for example only ``esm_finetune``) must not hide
    # already-registered artifacts from the model registry used by evaluation.
    if previous is not None and comparison_path.exists():
        current = json.loads(comparison_path.read_text())
        for name, metadata in previous.get("models", {}).items():
            current.setdefault("models", {}).setdefault(name, metadata)
        comparison_path.write_text(json.dumps(current, indent=2))
    data_volume.commit()
    artifact_volume.commit()
    return json.loads((Path(REMOTE_REFERENCE_MODELS) / "comparison.json").read_text())


@app.function(
    cpu=16,
    memory=65536,
    gpu="A10G",
    timeout=24 * 60 * 60,
    volumes={"/data": data_volume, "/artifacts": artifact_volume},
    env={
        "HF_HOME": "/data/huggingface",
        "WEIRDO_DATA_DIR": "/data/weirdo",
        "PYTHONPATH": REMOTE_ROOT,
    },
)
def run_reference_pipeline(
    architectures: str = "composition_mlp,cnn,esm_frozen",
    epochs: int = 20,
    train_per_stratum_length: int = 10_000,
    val_per_stratum_length: int = 2_000,
    test_per_stratum_length: int = 2_000,
    seed: int = 42,
) -> dict:
    """Run download, exact split construction, and training in one detached job."""
    subprocess.run(
        ["python", "-m", "weirdo.cli", "data", "reference-proteome", "--manifest"],
        cwd=REMOTE_ROOT,
        check=True,
    )
    data_volume.commit()
    subprocess.run(
        [
            "python",
            "scripts/reference_proteome_data.py",
            "build",
            "--reference-dir",
            REMOTE_REFERENCE_DATA,
            "--out",
            REMOTE_REFERENCE_SPLITS,
            "--train-per-stratum-length",
            str(train_per_stratum_length),
            "--val-per-stratum-length",
            str(val_per_stratum_length),
            "--test-per-stratum-length",
            str(test_per_stratum_length),
            "--seed",
            str(seed),
            "--exclude-peptide-csv",
            REMOTE_VIRAL_BENCHMARK,
            "--exclude-role",
            "validation",
        ],
        cwd=REMOTE_ROOT,
        check=True,
    )
    data_volume.commit()
    subprocess.run(
        [
            "python",
            "scripts/train_peptide_content_models.py",
            "--data-script",
            "scripts/reference_proteome_data.py",
            "--splits",
            REMOTE_REFERENCE_SPLITS,
            "--out",
            REMOTE_REFERENCE_MODELS,
            "--architectures",
            architectures,
            "--epochs",
            str(epochs),
            "--seed",
            str(seed),
            "--workers",
            "4",
        ],
        cwd=REMOTE_ROOT,
        check=True,
    )
    data_volume.commit()
    artifact_volume.commit()
    return json.loads((Path(REMOTE_REFERENCE_MODELS) / "comparison.json").read_text())


@app.function(
    cpu=8,
    memory=32768,
    gpu="A10G",
    timeout=4 * 60 * 60,
    volumes={"/data": data_volume, "/artifacts": artifact_volume},
    env={"HF_HOME": "/data/huggingface", "PYTHONPATH": REMOTE_ROOT},
)
def evaluate_models(
    bootstrap_draws: int = 1000,
    mhc_feature: str = "presentation",
) -> dict:
    if mhc_feature == "affinity":
        names = ["esm_finetune"]
    else:
        names = ["composition_mlp", "cnn", "transformer", "esm_frozen"]
        if (Path(REMOTE_MODELS) / "esm_finetune" / "config.json").exists():
            names.append("esm_finetune")
    model_paths = ",".join(
        f"{REMOTE_MODELS}/{name}" for name in names
    )
    inputs = {
        "broad": f"{REMOTE_ROOT}/data/foreignness-choices/dev-scored-assays.csv",
        "registered_viral": (
            f"{REMOTE_ROOT}/data/viral-tcell-benchmark/scored-assays.csv"
        ),
    }
    reports = {}
    for name, input_path in inputs.items():
        output = f"{REMOTE_EVALUATION}/{name}-{mhc_feature}"
        command = [
            "python",
            "scripts/evaluate_peptide_content_immunogenicity.py",
            "--input",
            input_path,
            "--models",
            model_paths,
            "--out",
            output,
            "--bootstrap-draws",
            str(bootstrap_draws),
            "--mhc-feature",
            mhc_feature,
        ]
        if mhc_feature == "affinity":
            command.extend(("--ablation-architecture", ""))
        subprocess.run(command, cwd=REMOTE_ROOT, check=True)
        reports[name] = json.loads((Path(output) / "report.json").read_text())
    artifact_volume.commit()
    return reports


@app.function(
    cpu=8,
    memory=32768,
    gpu="A10G",
    timeout=2 * 60 * 60,
    volumes={"/data": data_volume, "/artifacts": artifact_volume},
    env={"HF_HOME": "/data/huggingface", "PYTHONPATH": REMOTE_ROOT},
)
def evaluate_viral_host_taxonomy(architecture: str = "esm_finetune") -> dict:
    sys.path.insert(0, f"{REMOTE_ROOT}/scripts")
    import evaluate_peptide_content_immunogenicity as predictor

    data_script = f"{REMOTE_ROOT}/scripts/viral_host_peptide_content_data.py"
    predictor.set_data_script(data_script)
    training = predictor.training
    peptides, labels = training.pcd.load_split(REMOTE_HOST_SPLITS, "test")
    _, predict = predictor._load_predictor(Path(REMOTE_HOST_MODELS) / architecture)
    probabilities = predict(peptides)
    report = training.evaluate(probabilities, labels, peptides)
    output = Path(REMOTE_HOST_MODELS) / architecture / "evaluation.json"
    output.write_text(json.dumps(report, indent=2))
    artifact_volume.commit()
    return report


@app.function(
    cpu=8,
    memory=32768,
    gpu="A10G",
    timeout=6 * 60 * 60,
    volumes={"/artifacts": artifact_volume},
    env={"HF_HOME": "/data/huggingface", "PYTHONPATH": REMOTE_ROOT},
)
def evaluate_viral_host_improve(
    architecture: str = "esm_finetune",
    bootstrap_draws: int = 500,
) -> dict:
    command = [
        "python",
        "scripts/evaluate_improve_foreignness.py",
        "--input",
        REMOTE_IMPROVE,
        "--published-predictions",
        REMOTE_IMPROVE_PUBLISHED,
        "--model",
        f"{REMOTE_HOST_MODELS}/{architecture}",
        "--data-script",
        "scripts/viral_host_peptide_content_data.py",
        "--out",
        REMOTE_HOST_EVALUATION,
        "--bootstrap-draws",
        str(bootstrap_draws),
    ]
    subprocess.run(command, cwd=REMOTE_ROOT, check=True)
    artifact_volume.commit()
    report = json.loads((Path(REMOTE_HOST_EVALUATION) / "report.json").read_text())
    host_names = {
        "p_viruses",
        "p_mammal_host_viruses",
        "p_other_host_viruses",
        "host_virus_max",
        "mammal_host_virus_margin",
        "mammal_host_virus_specific",
        "mammal_host_virus_vs_mammals",
        "mammal_host_virus_vs_human",
    }
    report["viral_host_leaderboard"] = sorted(
        (
            row
            for row in report["results"]
            if row["score"] in host_names
            and row["analysis_set"] == "all mutant peptides"
        ),
        key=lambda row: row["delta_roc_auc"],
        reverse=True,
    )
    return report


@app.function(
    cpu=8,
    memory=65536,
    gpu="A10G",
    timeout=4 * 60 * 60,
    volumes={"/data": data_volume, "/artifacts": artifact_volume},
    env={"HF_HOME": "/data/huggingface", "PYTHONPATH": REMOTE_ROOT},
)
def evaluate_reference_models(
    bootstrap_draws: int = 1000,
    mhc_feature: str = "presentation",
) -> dict:
    """Score the untouched viral T-cell benchmark with reference-trained models."""
    comparison = json.loads((Path(REMOTE_REFERENCE_MODELS) / "comparison.json").read_text())
    names = list(comparison["models"])
    model_paths = ",".join(f"{REMOTE_REFERENCE_MODELS}/{name}" for name in names)
    output = f"{REMOTE_REFERENCE_EVALUATION}/registered-viral-{mhc_feature}"
    command = [
        "python",
        "scripts/evaluate_peptide_content_immunogenicity.py",
        "--data-script",
        "scripts/reference_proteome_data.py",
        "--input",
        REMOTE_VIRAL_BENCHMARK,
        "--models",
        model_paths,
        "--out",
        output,
        "--bootstrap-draws",
        str(bootstrap_draws),
        "--mhc-feature",
        mhc_feature,
        "--ablation-architecture",
        "esm_frozen",
    ]
    subprocess.run(command, cwd=REMOTE_ROOT, check=True)
    artifact_volume.commit()
    return json.loads((Path(output) / "report.json").read_text())


@app.function(
    cpu=8,
    memory=65536,
    gpu="A10G",
    timeout=4 * 60 * 60,
    volumes={"/data": data_volume, "/artifacts": artifact_volume},
    env={"HF_HOME": "/data/huggingface", "PYTHONPATH": REMOTE_ROOT},
)
def evaluate_reference_transfer(bootstrap_draws: int = 200) -> dict:
    """Calibrate response features on broad development data, then test validation."""
    comparison = json.loads((Path(REMOTE_REFERENCE_MODELS) / "comparison.json").read_text())
    names = list(comparison["models"])
    model_paths = ",".join(f"{REMOTE_REFERENCE_MODELS}/{name}" for name in names)
    output = f"{REMOTE_REFERENCE_EVALUATION}/broad-development-transfer"
    command = [
        "python",
        REMOTE_TRANSFER_SCRIPT,
        "--development",
        f"{REMOTE_ROOT}/data/foreignness-choices/dev-scored-assays.csv",
        "--validation",
        REMOTE_VIRAL_BENCHMARK,
        "--models",
        model_paths,
        "--data-script",
        "scripts/reference_proteome_data.py",
        "--out",
        output,
        "--bootstrap-draws",
        str(bootstrap_draws),
    ]
    subprocess.run(command, cwd=REMOTE_ROOT, check=True)
    artifact_volume.commit()
    return json.loads((Path(output) / "report.json").read_text())


@app.local_entrypoint()
def main(
    stage: str = "all",
    architectures: str = "composition_mlp,cnn,transformer,esm_frozen",
    epochs: int = 20,
    train_per_stratum_length: int = 10_000,
    val_per_stratum_length: int = 2_000,
    test_per_stratum_length: int = 2_000,
    seed: int = 42,
    bootstrap_draws: int = 1000,
    mhc_feature: str = "presentation",
):
    if stage not in {
        "analyze", "summarize", "build", "train", "evaluate", "all",
        "references", "reference-build", "reference-train", "reference-evaluate", "reference-transfer", "reference-all",
        "viral-host-build", "viral-host-train", "viral-host-taxonomy-evaluate",
        "viral-host-evaluate", "viral-host-all",
    }:
        raise ValueError(
            "stage must be analyze, summarize, build, train, evaluate, all, references, "
            "reference-build, reference-train, reference-evaluate, reference-transfer, reference-all, "
            "viral-host-build, viral-host-train, viral-host-taxonomy-evaluate, "
            "viral-host-evaluate, or viral-host-all"
        )
    if stage == "analyze":
        print(json.dumps(analyze_taxonomy_buckets.remote(), indent=2))
    if stage == "summarize":
        print(json.dumps(summarize_data.remote(), indent=2))
    if stage in {"build", "all"}:
        result = build_data.remote(
            train_per_stratum_length=train_per_stratum_length,
            val_per_stratum_length=val_per_stratum_length,
            test_per_stratum_length=test_per_stratum_length,
            seed=seed,
        )
        print(json.dumps({"data": result["splits"]}, indent=2))
    if stage in {"train", "all"}:
        result = train_models.remote(
            architectures=architectures,
            epochs=epochs,
            seed=seed,
        )
        print(json.dumps(result, indent=2))
    if stage in {"evaluate", "all"}:
        result = evaluate_models.remote(
            bootstrap_draws=bootstrap_draws,
            mhc_feature=mhc_feature,
        )
        print(json.dumps({name: report["primary"] for name, report in result.items()}, indent=2))
    if stage == "references":
        print(json.dumps(download_reference_proteomes.remote(), indent=2))
    if stage == "reference-build":
        result = build_reference_data.remote(
            train_per_stratum_length=train_per_stratum_length,
            val_per_stratum_length=val_per_stratum_length,
            test_per_stratum_length=test_per_stratum_length,
            seed=seed,
        )
        print(json.dumps({"reference_data": result["splits"]}, indent=2))
    if stage == "reference-train":
        print(
            json.dumps(
                train_reference_models.remote(
                    architectures=architectures, epochs=epochs, seed=seed
                ),
                indent=2,
            )
        )
    if stage == "reference-evaluate":
        result = evaluate_reference_models.remote(
            bootstrap_draws=bootstrap_draws, mhc_feature=mhc_feature
        )
        print(json.dumps(result["registered_validation_transfer"], indent=2))
    if stage == "reference-transfer":
        result = evaluate_reference_transfer.remote(bootstrap_draws=bootstrap_draws)
        print(json.dumps(result["results"], indent=2))
    if stage == "reference-all":
        print(
            json.dumps(
                run_reference_pipeline.remote(
                    architectures=architectures,
                    epochs=epochs,
                    train_per_stratum_length=train_per_stratum_length,
                    val_per_stratum_length=val_per_stratum_length,
                    test_per_stratum_length=test_per_stratum_length,
                    seed=seed,
                ),
                indent=2,
            )
        )
    if stage in {"viral-host-build", "viral-host-all"}:
        result = build_viral_host_data.remote(
            train_per_stratum_length=train_per_stratum_length,
            val_per_stratum_length=val_per_stratum_length,
            test_per_stratum_length=test_per_stratum_length,
            seed=seed,
        )
        print(json.dumps({"viral_host_data": result["splits"]}, indent=2))
    if stage in {"viral-host-train", "viral-host-all"}:
        result = train_viral_host_models.remote(
            architectures=architectures, epochs=epochs, seed=seed
        )
        print(json.dumps(result, indent=2))
    if stage in {"viral-host-taxonomy-evaluate", "viral-host-all"}:
        result = evaluate_viral_host_taxonomy.remote(architecture="esm_finetune")
        print(json.dumps(result["viral_host_summary"], indent=2))
    if stage in {"viral-host-evaluate", "viral-host-all"}:
        result = evaluate_viral_host_improve.remote(
            architecture="esm_finetune", bootstrap_draws=bootstrap_draws
        )
        print(json.dumps(result["viral_host_leaderboard"], indent=2))
