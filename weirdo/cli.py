"""Command-line interface for WEIRDO.

Usage:
    weirdo data status          # Show data status
    weirdo data download        # Download reference data
    weirdo data clear           # Clear all data
    weirdo score --model NAME PEPTIDE...  # Score peptides
    weirdo models list          # List trained models
    weirdo models train         # Train a new model
    weirdo models info NAME     # Show model info
"""

import sys
import argparse

from .reduced_alphabet import alphabets


def _translate_sequence(sequence, alphabet):
    """Translate a sequence using a reduced alphabet mapping."""
    return "".join([alphabet.get(aa, aa) for aa in sequence])


def create_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog='weirdo',
        description='WEIRDO: Widely Estimated Immunological Recognition and Detection of Outliers',
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

    # data download
    download_parser = data_subparsers.add_parser('download', help='Download reference data')
    download_parser.add_argument(
        'dataset',
        nargs='?',
        default='swissprot-8mers',
        help='Dataset to download (default: swissprot-8mers)',
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
    score_parser = subparsers.add_parser('score', help='Score peptides for foreignness')
    score_parser.add_argument(
        'peptides',
        nargs='+',
        help='Peptide sequences to score',
    )
    score_parser.add_argument(
        '-m', '--model',
        required=True,
        help='Trained model name to use for scoring',
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

    return parser


def cmd_data_list(args):
    """Handle: weirdo data list/ls/status"""
    from .data_manager import get_data_manager
    dm = get_data_manager()
    dm.print_status()


def cmd_data_download(args):
    """Handle: weirdo data download"""
    from .data_manager import get_data_manager, DATASETS
    dm = get_data_manager()

    if args.all:
        dm.download_all(force=args.force)
    else:
        if args.dataset not in DATASETS:
            print(f"Unknown dataset: {args.dataset}")
            print(f"Available: {list(DATASETS.keys())}")
            return 1
        dm.download(args.dataset, force=args.force)
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


def cmd_score(args):
    """Handle: weirdo score"""
    from .model_manager import load_model

    print(f"Scoring {len(args.peptides)} peptide(s) with model '{args.model}'...")
    print()

    try:
        scorer = load_model(args.model)
        if getattr(scorer, 'target_categories', None):
            df = scorer.predict_dataframe(args.peptides)
            print(df.to_string(index=False))
            print()
            print("Foreignness is derived from max(pathogens) vs max(self).")
        else:
            scores = scorer.score(args.peptides)
            print(f"{'Peptide':<40} {'Score':>10}")
            print("-" * 52)
            for pep, score in zip(args.peptides, scores):
                display_pep = pep if len(pep) <= 37 else pep[:34] + '...'
                print(f"{display_pep:<40} {score:>10.4f}")
            print()
            print("Higher scores = more foreign")
    except FileNotFoundError as e:
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

    print("WEIRDO Setup")
    print("=" * 60)
    print()

    # Download data
    print("Step 1: Downloading reference data...")
    dm.download('swissprot-8mers')
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
        elif args.data_command == 'download':
            return cmd_data_download(args)
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
        elif args.models_command == 'scorers':
            return cmd_models_scorers(args)

    return 0


if __name__ == '__main__':
    sys.exit(run() or 0)
