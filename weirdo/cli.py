"""Command-line interface for Weirdo.

Usage:
    weirdo score PEPTIDE...     # Species-category probabilities and foreignness
    weirdo data status          # Show data status
    weirdo data download        # Download SwissProt from UniProt, build the reference data
    weirdo data clear           # Clear all data
    weirdo score --model NAME PEPTIDE...  # Score peptides with another model
    weirdo models list          # List trained models
    weirdo models available     # List downloadable pretrained models
    weirdo models download NAME # Download pretrained model weights
    weirdo models train         # Train a new model
    weirdo models info NAME     # Show model info
"""

import sys
import argparse

from .data_manager import DATASETS as _DATASETS
from .model_manager import DEFAULT_CATEGORY_MODEL
from .reduced_alphabet import alphabets

DATASET_NAMES = tuple(_DATASETS)


def _translate_sequence(sequence, alphabet):
    """Translate a sequence using a reduced alphabet mapping."""
    return "".join([alphabet.get(aa, aa) for aa in sequence])


def create_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog='weirdo',
        description='Weirdo: Widely Estimated Immunological Recognition and Detection of Outliers',
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # -------------------------------------------------------------------------
    # Data management commands
    # -------------------------------------------------------------------------
    data_parser = subparsers.add_parser('data', help='Manage reference data')
    data_subparsers = data_parser.add_subparsers(dest='data_command', help='Data commands')

    # data list (aliased as 'ls' and 'status')
    list_parser = data_subparsers.add_parser('list', help='List datasets with status')
    data_subparsers.add_parser('ls', help='Alias for list')
    data_subparsers.add_parser('status', help='Alias for list')

    # data cohorts
    cohorts_parser = data_subparsers.add_parser(
        'cohorts',
        help='List the T-cell evaluation datasets with sources, builders and local status',
    )
    cohorts_parser.add_argument('--root', default='.', help='Repository root holding data/ and reports/')
    cohorts_parser.add_argument('--checksums', action='store_true', help='Print SHA-256 of local files')

    # data download
    download_parser = data_subparsers.add_parser(
        'download',
        help='Download SwissProt from UniProt and build the reference data',
        description=(
            'Downloads the SwissProt taxonomic-division files from UniProt (about 670 MB, '
            'checked against the sizes and MD5s UniProt lists) and builds the dataset from them. '
            'swissprot-8mers also builds the k-mer table (several GB, some minutes).'
        ),
    )
    download_parser.add_argument(
        'dataset',
        nargs='?',
        default='swissprot-8mers',
        choices=sorted(DATASET_NAMES),
        help='Dataset to build (default: %(default)s)',
    )
    download_parser.add_argument(
        '--all',
        action='store_true',
        help='Download all available datasets',
    )
    download_parser.add_argument(
        '-f', '--force',
        action='store_true',
        help='Force re-download even if already present',
    )

    host_reference_parser = data_subparsers.add_parser(
        'reference-proteome',
        help='Download a complete UniProt reference proteome for exact host-self lookup',
    )
    host_reference_parser.add_argument(
        'host',
        nargs='?',
        help='Host name (human, dog, cat, mouse, rat, or cattle)',
    )
    host_reference_parser.add_argument(
        '--list',
        action='store_true',
        help='List built-in host aliases and packaged frozen selections',
    )
    host_reference_parser.add_argument(
        '--proteome',
        metavar='UPID',
        help='Resolve and download any UniProt reference-proteome accession (for example UP000009136)',
    )
    host_reference_parser.add_argument(
        '--manifest',
        nargs='?',
        const='reference-proteomes-v1',
        metavar='NAME',
        help='Download every verified reference in a frozen selection (default: reference-proteomes-v1)',
    )
    host_reference_parser.add_argument(
        '--group',
        action='append',
        default=[],
        metavar='GROUP',
        help='Limit --manifest to a group, e.g. mammals, viruses, primates, or carnivora; repeatable',
    )
    host_reference_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show the number of selected manifest proteomes without downloading',
    )
    host_reference_parser.add_argument(
        '-f', '--force',
        action='store_true',
        help='Re-download even if a verified local copy exists',
    )

    # data clear
    clear_parser = data_subparsers.add_parser('clear', help='Clear downloaded data')
    clear_parser.add_argument(
        '--downloads',
        action='store_true',
        help='Clear downloaded dataset files (default behavior)',
    )
    clear_parser.add_argument(
        '--all',
        action='store_true',
        help='Clear downloads and reset data metadata',
    )
    clear_parser.add_argument(
        '-y', '--yes',
        action='store_true',
        help='Skip confirmation prompt',
    )

    # data path
    path_parser = data_subparsers.add_parser('path', help='Show path to data directory')

    # -------------------------------------------------------------------------
    # Score command
    # -------------------------------------------------------------------------
    score_parser = subparsers.add_parser(
        'score',
        help='Predict species-category probabilities and foreignness for peptides',
    )
    score_parser.add_argument(
        'peptides',
        nargs='+',
        help='Peptide sequences to score',
    )
    score_parser.add_argument(
        '-m', '--model',
        default=DEFAULT_CATEGORY_MODEL,
        help='Model name (bundled or trained); default: %(default)s',
    )
    score_parser.add_argument(
        '--combine',
        choices=['max', 'sum'],
        default='max',
        help='Combine probabilities within the pathogen and self groups by their largest '
             'value or their sum (multi-label models only; default: %(default)s)',
    )
    score_parser.add_argument(
        '--csv',
        action='store_true',
        help='Print the result as CSV with no extra text (multi-label models only)',
    )

    # -------------------------------------------------------------------------
    # Translate command (legacy)
    # -------------------------------------------------------------------------
    translate_parser = subparsers.add_parser(
        'translate',
        help='Translate amino acid sequences to reduced alphabets',
    )
    translate_inputs = translate_parser.add_mutually_exclusive_group(required=True)
    translate_inputs.add_argument('--input-fasta')
    translate_inputs.add_argument('--input-sequence')
    translate_parser.add_argument(
        '-a', '--alphabet',
        dest='alphabet',
        help='Reduced alphabet name',
        choices=tuple(alphabets.keys()),
        required=True,
    )

    # -------------------------------------------------------------------------
    # Setup command
    # -------------------------------------------------------------------------
    setup_parser = subparsers.add_parser('setup', help='Initial setup - download reference data')

    # -------------------------------------------------------------------------
    # Model management commands
    # -------------------------------------------------------------------------
    models_parser = subparsers.add_parser('models', help='Manage trained ML models')
    models_subparsers = models_parser.add_subparsers(dest='models_command', help='Model commands')

    # models list
    models_list_parser = models_subparsers.add_parser('list', help='List trained models')
    models_subparsers.add_parser('ls', help='Alias for list')

    # models info
    models_info_parser = models_subparsers.add_parser('info', help='Show model details')
    models_info_parser.add_argument('name', help='Model name')

    # models delete
    models_delete_parser = models_subparsers.add_parser('delete', help='Delete a trained model')
    models_delete_parser.add_argument('name', help='Model name to delete')
    models_delete_parser.add_argument('-y', '--yes', action='store_true', help='Skip confirmation')

    # models train
    models_train_parser = models_subparsers.add_parser('train', help='Train a new model')
    models_train_parser.add_argument(
        '--type',
        default='mlp',
        choices=['mlp'],
        help='Model type to train (default: mlp)',
    )
    models_train_parser.add_argument(
        '--data',
        required=True,
        help='Training data CSV (columns: peptide, label or peptide + category columns)',
    )
    models_train_parser.add_argument(
        '--val-data',
        help='Validation data CSV (optional)',
    )
    models_train_parser.add_argument(
        '--name',
        help='Name for saved model (default: auto-generated)',
    )
    models_train_parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Training epochs (default: 100)',
    )
    models_train_parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate (default: 1e-3)',
    )
    models_train_parser.add_argument(
        '--k',
        type=int,
        default=8,
        help='K-mer size (default: 8)',
    )
    models_train_parser.add_argument(
        '--hidden-layers',
        default='256,128,64',
        help='Hidden layer sizes, comma-separated (default: 256,128,64)',
    )
    models_train_parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing model with same name',
    )

    # models path
    models_path_parser = models_subparsers.add_parser('path', help='Show models directory')

    # models scorers
    models_scorers_parser = models_subparsers.add_parser('scorers', help='List available scorer types')

    # models available
    models_available_parser = models_subparsers.add_parser(
        'available',
        help='List downloadable pretrained models',
    )

    # models download
    models_download_parser = models_subparsers.add_parser(
        'download',
        help='Download pretrained model weights',
    )
    models_download_parser.add_argument(
        'name',
        nargs='?',
        help='Pretrained model name (or custom local name when using --url)',
    )
    models_download_parser.add_argument(
        '--url',
        help='Custom URL to a .tar.gz model artifact',
    )
    models_download_parser.add_argument(
        '--save-as',
        help='Local model name when downloading with --url',
    )
    models_download_parser.add_argument(
        '--sha256',
        help='Expected SHA256 checksum for archive verification',
    )
    models_download_parser.add_argument(
        '-f', '--force',
        action='store_true',
        help='Overwrite existing local model with same name',
    )

    return parser


def cmd_data_list(args):
    """Handle: weirdo data list/ls/status"""
    from .data_manager import get_data_manager
    dm = get_data_manager()
    dm.print_status()
    return 0


def cmd_data_cohorts(args):
    """Handle: weirdo data cohorts"""
    from pathlib import Path
    from .evaluation_cohorts import cohort_status
    for row in cohort_status(Path(args.root), with_checksums=args.checksums):
        print(f"{row['name']}  [{row['role']}]")
        print(f"  {row['description']}")
        print(f"  source:  {row['source']}")
        if row['reference']:
            print(f"  ref:     {row['reference']}")
        if row['builder']:
            print(f"  builder: {row['builder']}")
        print(f"  reports: {', '.join(row['reports'])}")
        for rel, state in row['present'].items():
            label = state if isinstance(state, str) else ('present' if state else 'missing')
            print(f"  file:    {rel}  {label}")
    return 0


def cmd_data_download(args):
    """Handle: weirdo data download"""
    from .data_manager import get_data_manager, DATASETS
    dm = get_data_manager()

    try:
        if args.all:
            dm.download_all(force=args.force)
        else:
            if args.dataset not in DATASETS:
                print(f"Unknown dataset: {args.dataset}")
                print(f"Available: {list(DATASETS.keys())}")
                return 1
            dm.download(args.dataset, force=args.force)
    except RuntimeError as e:
        print(f"Error: {e}")
        return 1
    return 0


def cmd_data_reference_proteome(args):
    """Handle: weirdo data reference-proteome [HOST]."""
    from .data_manager import get_data_manager
    from .reference_proteomes import (
        list_reference_proteome_manifests,
        list_reference_proteomes,
        load_reference_proteome_manifest,
        reference_proteome_from_uniprot,
    )

    if args.list:
        if args.host or args.proteome or args.manifest or args.group or args.dry_run:
            print("Provide a host, --proteome, --manifest, or --list; these options cannot be combined.")
            return 1
        for proteome in list_reference_proteomes():
            print(
                f"{proteome.key}\t{proteome.scientific_name}\t"
                f"taxon:{proteome.taxon_id}\t{proteome.proteome_id}"
            )
        print("\nFrozen selections:")
        for name in list_reference_proteome_manifests():
            selection = load_reference_proteome_manifest(name)
            print(f"{name}\t{len(selection.entries)} proteomes\tgroups: {', '.join(selection.groups)}")
        return 0
    targets = sum(bool(value) for value in (args.host, args.proteome, args.manifest))
    if targets > 1:
        print("Provide exactly one of a host, --proteome, or --manifest.")
        return 1
    if args.group and not args.manifest:
        print("--group requires --manifest.")
        return 1
    if args.dry_run and not args.manifest:
        print("--dry-run requires --manifest.")
        return 1
    if not args.host and not args.proteome and not args.manifest:
        print("Provide a host, --proteome, or --manifest, or use --list to see the available references.")
        return 1
    try:
        if args.manifest:
            selection = load_reference_proteome_manifest(args.manifest)
            entries = selection.select(tuple(args.group))
            print(
                f"{selection.name}: {len(entries):,} selected reference proteomes"
                + (f" for {', '.join(args.group)}" if args.group else "")
            )
            if args.dry_run:
                return 0
            get_data_manager().download_reference_proteome_manifest(
                selection.name, groups=tuple(args.group), force=args.force
            )
            return 0
        target = reference_proteome_from_uniprot(args.proteome) if args.proteome else args.host
        get_data_manager().download_host_reference(target, force=args.force)
    except (ConnectionError, RuntimeError, ValueError) as error:
        print(f"Error: {error}")
        return 1
    return 0


def cmd_data_clear(args):
    """Handle: weirdo data clear"""
    if args.downloads and args.all:
        print("Choose either --downloads or --all, not both.")
        return 1

    from .data_manager import get_data_manager
    dm = get_data_manager()

    # Confirm
    if not args.yes:
        status = dm.status()
        size_mb = status['total_size_mb']
        if args.all:
            print(f"This will delete downloads ({size_mb:.1f} MB) and reset metadata.")
        else:
            print(f"This will delete downloads ({size_mb:.1f} MB)")
        response = input("Continue? [y/N] ")
        if response.lower() not in ('y', 'yes'):
            print("Aborted.")
            return 1

    # Clear
    if args.all:
        count = dm.clear_all(include_metadata=True)
        print(f"Deleted {count} downloads and reset metadata")
    else:
        count = dm.clear_all(include_metadata=False)
        print(f"Deleted {count} downloads")

    return 0


def cmd_data_path(args):
    """Handle: weirdo data path"""
    from .data_manager import get_data_manager
    dm = get_data_manager()
    print(dm.data_dir)
    return 0


def cmd_score(args):
    """Handle: weirdo score"""
    from .model_manager import load_model

    csv_output = bool(getattr(args, 'csv', False))
    if not csv_output:
        print(f"Scoring {len(args.peptides)} peptide(s) with model '{args.model}'...")
        print()

    try:
        scorer = load_model(args.model)
        if getattr(scorer, 'target_categories', None):
            combine = getattr(args, 'combine', 'max')
            df = scorer.predict_dataframe(args.peptides, combine=combine)
            if csv_output:
                print(df.to_csv(index=False), end='')
                return 0
            print(df.round(3).to_string(index=False))
            print()
            print("Category columns: independent sigmoid probabilities that the peptide occurs in")
            print("proteins of that category (they need not sum to 1).")
            if combine == 'sum':
                print("foreignness = pathogen_sum / (pathogen_sum + self_sum), the expected number of")
                print("pathogen categories (bacteria, viruses) against self categories (human, rodents,")
                print("mammals) that contain the peptide.")
            else:
                print("foreignness = pathogen_max / (pathogen_max + self_max): the largest probability")
                print("among the pathogen categories (bacteria, viruses) against the largest among the")
                print("self categories (human, rodents, mammals).")
            print("Compare a foreignness value with the reference distributions in the documentation;")
            print("typical peptides score well above 0.5.")
        else:
            scores = scorer.score(args.peptides)
            print(f"{'Peptide':<40} {'Score':>10}")
            print("-" * 52)
            for pep, score in zip(args.peptides, scores):
                display_pep = pep if len(pep) <= 37 else pep[:34] + '...'
                print(f"{display_pep:<40} {score:>10.4f}")
            print()
            print("Higher scores = more foreign")
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {e}")
        return 1

    return 0


def cmd_translate(args):
    """Handle: weirdo translate"""
    alphabet = alphabets[args.alphabet]

    if args.input_sequence:
        result = _translate_sequence(args.input_sequence, alphabet)
        print(f"{args.input_sequence} -> {result}")
    elif args.input_fasta:
        try:
            with open(args.input_fasta, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"Error: FASTA file not found: {args.input_fasta}")
            return 1

        output_lines = []
        sequence_chunks = []
        saw_header = False

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith('>'):
                saw_header = True
                if sequence_chunks:
                    sequence = "".join(sequence_chunks)
                    output_lines.append(_translate_sequence(sequence, alphabet))
                    sequence_chunks = []
                output_lines.append(line)
            else:
                sequence_chunks.append(line)

        if sequence_chunks:
            sequence = "".join(sequence_chunks)
            output_lines.append(_translate_sequence(sequence, alphabet))

        if not saw_header:
            print("Error: FASTA file must contain at least one header line starting with '>'")
            return 1

        print("\n".join(output_lines))

    return 0


def cmd_setup(args):
    """Handle: weirdo setup"""
    from .data_manager import get_data_manager
    dm = get_data_manager()

    print("Weirdo Setup")
    print("=" * 60)
    print()

    # Download data
    print("Step 1: Downloading SwissProt from UniProt and building the reference data...")
    try:
        dm.download('swissprot-8mers')
    except RuntimeError as e:
        print(f"Error: {e}")
        return 1
    print()

    print("Setup complete!")
    print()
    dm.print_status()

    return 0


# -------------------------------------------------------------------------
# Model command handlers
# -------------------------------------------------------------------------

def cmd_models_list(args):
    """Handle: weirdo models list"""
    from .model_manager import get_model_manager
    mm = get_model_manager()
    mm.print_models()
    return 0


def cmd_models_info(args):
    """Handle: weirdo models info NAME"""
    from .model_manager import get_model_manager
    mm = get_model_manager()

    info = mm.get_model_info(args.name)
    if info is None:
        print(f"Model not found: {args.name}")
        return 1

    print(f"Model: {info.name}")
    print("=" * 60)
    print(f"  Type: {info.scorer_type}")
    print(f"  Path: {info.path}")
    if info.created:
        print(f"  Created: {info.created[:19]}")
    print()

    print("Parameters:")
    for key, value in info.params.items():
        print(f"  {key}: {value}")
    print()

    if info.metadata:
        print("Training info:")
        if 'n_train' in info.metadata:
            print(f"  Training samples: {info.metadata['n_train']}")
        if 'n_epochs' in info.metadata:
            print(f"  Epochs trained: {info.metadata['n_epochs']}")
        elif 'n_iter' in info.metadata:
            print(f"  Epochs trained: {info.metadata['n_iter']}")
        if 'final_train_loss' in info.metadata:
            print(f"  Final train loss: {info.metadata['final_train_loss']:.4f}")
        elif 'loss' in info.metadata:
            print(f"  Final train loss: {info.metadata['loss']:.4f}")
        if 'final_val_loss' in info.metadata:
            print(f"  Final val loss: {info.metadata['final_val_loss']:.4f}")
        if 'best_val_loss' in info.metadata:
            print(f"  Best val loss: {info.metadata['best_val_loss']:.4f}")
        elif 'best_loss' in info.metadata:
            print(f"  Best val loss: {info.metadata['best_loss']:.4f}")
        if 'best_val_score' in info.metadata:
            print(f"  Best val score: {info.metadata['best_val_score']:.4f}")

    return 0


def cmd_models_delete(args):
    """Handle: weirdo models delete NAME"""
    from .model_manager import get_model_manager
    mm = get_model_manager()

    info = mm.get_model_info(args.name)
    if info is None:
        print(f"Model not found: {args.name}")
        return 1

    if not args.yes:
        response = input(f"Delete model '{args.name}'? [y/N] ")
        if response.lower() not in ('y', 'yes'):
            print("Aborted.")
            return 1

    if mm.delete(args.name):
        print(f"Deleted model: {args.name}")
    else:
        print(f"Failed to delete model: {args.name}")
        return 1

    return 0


def cmd_models_train(args):
    """Handle: weirdo models train"""
    import csv
    from datetime import datetime

    from .model_manager import get_model_manager
    mm = get_model_manager()

    # Generate default name if not provided
    if args.name:
        model_name = args.name
    else:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        model_name = f"mlp-{timestamp}"

    # Check if model already exists
    if mm.get_model_info(model_name) and not args.overwrite:
        print(f"Model already exists: {model_name}")
        print("Use --overwrite to replace.")
        return 1

    # Load training data
    print(f"Loading training data from {args.data}...")
    peptides = []
    labels = []
    target_categories = None
    with open(args.data, 'r') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or 'peptide' not in reader.fieldnames:
            print("Training CSV must include a 'peptide' column.")
            return 1
        label_columns = [c for c in reader.fieldnames if c != 'peptide']
        if not label_columns:
            print("Training CSV must include at least one label column.")
            return 1

        for row in reader:
            peptides.append(row['peptide'])
            if label_columns == ['label']:
                labels.append(float(row['label']))
            else:
                labels.append([float(row[c]) for c in label_columns])
        if label_columns != ['label']:
            target_categories = label_columns

    print(f"  Loaded {len(peptides)} samples")
    if target_categories:
        print(f"  Target categories: {', '.join(target_categories)}")

    # Load validation data if provided
    val_peptides = None
    val_labels = None
    if args.val_data:
        print(f"Loading validation data from {args.val_data}...")
        val_peptides = []
        val_labels = []
        with open(args.val_data, 'r') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or 'peptide' not in reader.fieldnames:
                print("Validation CSV must include a 'peptide' column.")
                return 1
            val_label_columns = [c for c in reader.fieldnames if c != 'peptide']
            if target_categories:
                if val_label_columns != target_categories:
                    print("Validation label columns must match training labels.")
                    print(f"Expected: {target_categories}")
                    print(f"Found:    {val_label_columns}")
                    return 1
            else:
                if val_label_columns != ['label']:
                    print("Validation CSV must include a single 'label' column.")
                    return 1
            for row in reader:
                val_peptides.append(row['peptide'])
                if target_categories:
                    val_labels.append([float(row[c]) for c in val_label_columns])
                else:
                    val_labels.append(float(row['label']))
        print(f"  Loaded {len(val_peptides)} validation samples")

    # Create model
    from .scorers.mlp import MLPScorer
    hidden_layers = tuple(int(x) for x in args.hidden_layers.split(','))

    print()
    print(f"Training {args.type} model...")
    print(f"  K-mer size: {args.k}")
    print(f"  Hidden layers: {hidden_layers}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print()

    scorer = MLPScorer(
        k=args.k,
        hidden_layer_sizes=hidden_layers,
    )

    # Train
    scorer.train(
        peptides=peptides,
        labels=labels,
        val_peptides=val_peptides,
        val_labels=val_labels,
        epochs=args.epochs,
        learning_rate=args.lr,
        verbose=True,
        target_categories=target_categories,
    )

    # Save
    print()
    print(f"Saving model as '{model_name}'...")
    path = mm.save(scorer, model_name, overwrite=args.overwrite)
    print(f"  Saved to: {path}")

    return 0


def cmd_models_path(args):
    """Handle: weirdo models path"""
    from .model_manager import get_model_manager
    mm = get_model_manager()
    print(mm.model_dir)
    return 0


def cmd_models_available(args):
    """Handle: weirdo models available"""
    from .model_manager import list_bundled_models, list_pretrained_models

    bundled = list_bundled_models()
    if bundled:
        print("Models bundled with WEIRDO (no download needed):")
        print("-" * 60)
        for model in bundled:
            print(f"\n  {model['name']}  ({model['scorer_type']})")
            description = model['metadata'].get('description')
            if description:
                print(f"    {description}")
        print()

    models = list_pretrained_models()
    if not models:
        print("No built-in pretrained models are configured.")
        print("Use custom URL download: weirdo models download --url URL --save-as NAME")
        return 0

    print("Downloadable pretrained models:")
    print("-" * 60)
    for model in models:
        print(f"\n  {model['name']}")
        if model.get('description'):
            print(f"    {model['description']}")
        if model.get('url'):
            print(f"    URL: {model['url']}")
        if model.get('sha256'):
            print(f"    SHA256: {model['sha256']}")
    return 0


def cmd_models_download(args):
    """Handle: weirdo models download"""
    from .model_manager import download_pretrained_model, download_model_from_url

    if args.url:
        if args.save_as and args.name:
            print("Provide either positional NAME or --save-as with --url, not both.")
            return 1
        model_name = args.save_as or args.name
        if not model_name:
            print("When using --url, provide a model name via positional NAME or --save-as.")
            return 1
        try:
            path = download_model_from_url(
                name=model_name,
                url=args.url,
                overwrite=args.force,
                expected_sha256=args.sha256,
            )
        except (FileExistsError, RuntimeError, ValueError) as e:
            print(f"Error: {e}")
            return 1
        print(f"Downloaded model '{model_name}' to: {path}")
        return 0

    if args.save_as or args.sha256:
        print("--save-as and --sha256 can only be used with --url.")
        return 1
    if not args.name:
        print("Provide a pretrained model name or use --url for a custom artifact.")
        return 1

    try:
        path = download_pretrained_model(name=args.name, overwrite=args.force)
    except (FileExistsError, RuntimeError, ValueError) as e:
        print(f"Error: {e}")
        return 1
    print(f"Downloaded pretrained model '{args.name}' to: {path}")
    return 0


def cmd_models_scorers(args):
    """Handle: weirdo models scorers"""
    from .scorers import list_scorers, get_scorer, TrainableScorer

    print("Available scorer types:")
    print()

    scorers = list_scorers()
    for name in scorers:
        scorer_cls = get_scorer(name)
        if issubclass(scorer_cls, TrainableScorer):
            descriptor = "ML-based, requires training"
        else:
            descriptor = "Reference-based, requires fit(reference)"
        print(f"  {name:<20} ({descriptor})")

    print()
    print("Trainable scorers use train() with labeled peptide data.")
    print("Reference-based scorers require a fitted reference dataset.")

    return 0


def run(args_list=None):
    """Main entry point for CLI."""
    if args_list is None:
        args_list = sys.argv[1:]

    parser = create_parser()
    args = parser.parse_args(args_list)

    if args.command is None:
        parser.print_help()
        return 0

    # Data commands
    if args.command == 'data':
        if args.data_command is None:
            # Default to list
            return cmd_data_list(args)
        elif args.data_command in ('list', 'ls', 'status'):
            return cmd_data_list(args)
        elif args.data_command == 'cohorts':
            return cmd_data_cohorts(args)
        elif args.data_command == 'download':
            return cmd_data_download(args)
        elif args.data_command == 'reference-proteome':
            return cmd_data_reference_proteome(args)
        elif args.data_command == 'clear':
            return cmd_data_clear(args)
        elif args.data_command == 'path':
            return cmd_data_path(args)

    elif args.command == 'score':
        return cmd_score(args)

    elif args.command == 'translate':
        return cmd_translate(args)

    elif args.command == 'setup':
        return cmd_setup(args)

    elif args.command == 'models':
        if args.models_command is None:
            return cmd_models_list(args)
        elif args.models_command in ('list', 'ls'):
            return cmd_models_list(args)
        elif args.models_command == 'info':
            return cmd_models_info(args)
        elif args.models_command == 'delete':
            return cmd_models_delete(args)
        elif args.models_command == 'train':
            return cmd_models_train(args)
        elif args.models_command == 'path':
            return cmd_models_path(args)
        elif args.models_command == 'available':
            return cmd_models_available(args)
        elif args.models_command == 'download':
            return cmd_models_download(args)
        elif args.models_command == 'scorers':
            return cmd_models_scorers(args)

    return 0


if __name__ == '__main__':
    sys.exit(run() or 0)
