"""Command-line interface for WEIRDO.

Usage:
    weirdo data status          # Show data status
    weirdo data download        # Download reference data
    weirdo data clear           # Clear all data
    weirdo score PEPTIDE        # Score a peptide
    weirdo models list          # List trained models
    weirdo models train         # Train a new model
    weirdo models info NAME     # Show model info
"""

import sys
import argparse

from .reduced_alphabet import alphabets


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
    data_parser = subparsers.add_parser('data', help='Manage reference data and indices')
    data_subparsers = data_parser.add_subparsers(dest='data_command', help='Data commands')

    # data list (aliased as 'ls' and 'status')
    list_parser = data_subparsers.add_parser('list', help='List datasets and indices with status')
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
    clear_parser = data_subparsers.add_parser('clear', help='Clear downloaded data and/or indices')
    clear_parser.add_argument(
        '--downloads',
        action='store_true',
        help='Clear only downloaded data files',
    )
    clear_parser.add_argument(
        '--indices',
        action='store_true',
        help='Clear only built indices',
    )
    clear_parser.add_argument(
        '--all',
        action='store_true',
        help='Clear everything (downloads and indices)',
    )
    clear_parser.add_argument(
        '-y', '--yes',
        action='store_true',
        help='Skip confirmation prompt',
    )

    # data index
    index_parser = data_subparsers.add_parser('index', help='Build or rebuild indices')
    index_parser.add_argument(
        'index_name',
        nargs='?',
        help='Index to build (frequency, set). If not specified, builds all.',
    )
    index_parser.add_argument(
        '-f', '--force',
        action='store_true',
        help='Force rebuild even if already present',
    )
    index_parser.add_argument(
        '--all',
        action='store_true',
        help='Rebuild all indices',
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
        '-p', '--preset',
        default='default',
        help='Scoring preset (default, human, pathogen, etc.)',
    )
    score_parser.add_argument(
        '--auto-download',
        action='store_true',
        help='Automatically download data if not present',
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
    setup_parser = subparsers.add_parser('setup', help='Initial setup - download data and build indices')
    setup_parser.add_argument(
        '--skip-index',
        action='store_true',
        help='Skip building indices (just download data)',
    )

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
        help='Training data CSV (columns: peptide, label)',
    )
    models_train_parser.add_argument(
        '--val-data',
        help='Validation data CSV (optional)',
    )
    models_train_parser.add_argument(
        '--name',
        required=True,
        help='Name for saved model',
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
    from .data_manager import get_data_manager
    dm = get_data_manager()

    # Determine what to clear
    clear_downloads = args.downloads or args.all or (not args.downloads and not args.indices)
    clear_indices = args.indices or args.all or (not args.downloads and not args.indices)

    if not clear_downloads and not clear_indices:
        clear_downloads = clear_indices = True

    # Confirm
    if not args.yes:
        status = dm.status()
        size_mb = status['total_size_mb']
        what = []
        if clear_downloads:
            what.append('downloads')
        if clear_indices:
            what.append('indices')
        print(f"This will delete {' and '.join(what)} ({size_mb:.1f} MB)")
        response = input("Continue? [y/N] ")
        if response.lower() not in ('y', 'yes'):
            print("Aborted.")
            return 1

    # Clear
    if clear_indices:
        count = dm.delete_all_indices()
        print(f"Deleted {count} indices")

    if clear_downloads:
        count = dm.delete_all_downloads()
        print(f"Deleted {count} downloads")

    return 0


def cmd_data_index(args):
    """Handle: weirdo data index"""
    from .data_manager import get_data_manager, INDEX_TYPES
    dm = get_data_manager()

    if args.all:
        dm.rebuild_indices()
    elif args.index_name:
        if args.index_name not in INDEX_TYPES:
            print(f"Unknown index: {args.index_name}")
            print(f"Available: {list(INDEX_TYPES.keys())}")
            return 1
        dm.build_index(args.index_name, force=args.force)
    else:
        # Build all indices
        for name in INDEX_TYPES:
            dm.build_index(name, force=args.force)
    return 0


def cmd_data_path(args):
    """Handle: weirdo data path"""
    from .data_manager import get_data_manager
    dm = get_data_manager()
    print(dm.data_dir)


def cmd_score(args):
    """Handle: weirdo score"""
    from .data_manager import get_data_manager

    # Check if data is available
    dm = get_data_manager(auto_download=args.auto_download, verbose=True)

    if not dm.is_downloaded('swissprot-8mers'):
        if args.auto_download:
            print("Reference data not found. Downloading...")
            dm.download('swissprot-8mers')
        else:
            print("Reference data not found.")
            print("Run: weirdo data download")
            print("Or use: weirdo score --auto-download PEPTIDE")
            return 1

    # Score peptides
    from .api import score_peptides, create_scorer

    print(f"Scoring {len(args.peptides)} peptide(s) with preset '{args.preset}'...")
    print()

    try:
        scorer = create_scorer(args.preset, cache=True)
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
        result = "".join([alphabet.get(aa, aa) for aa in args.input_sequence])
        print(f"{args.input_sequence} -> {result}")
    elif args.input_fasta:
        print("FASTA translation not yet implemented")
        return 1

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

    # Build indices (optional)
    if not args.skip_index:
        print("Step 2: Building indices...")
        dm.build_index('frequency')
        dm.build_index('set')
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
        if 'final_train_loss' in info.metadata:
            print(f"  Final train loss: {info.metadata['final_train_loss']:.4f}")
        if 'best_val_loss' in info.metadata:
            print(f"  Best val loss: {info.metadata['best_val_loss']:.4f}")

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

    # Check for torch
    try:
        import torch
    except ImportError:
        print("Error: PyTorch is required for training ML models.")
        print("Install with: pip install torch")
        return 1

    from .model_manager import get_model_manager
    mm = get_model_manager()

    # Check if model already exists
    if mm.get_model_info(args.name) and not args.overwrite:
        print(f"Model already exists: {args.name}")
        print("Use --overwrite to replace.")
        return 1

    # Load training data
    print(f"Loading training data from {args.data}...")
    peptides = []
    labels = []
    with open(args.data, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            peptides.append(row['peptide'])
            labels.append(float(row['label']))
    print(f"  Loaded {len(peptides)} samples")

    # Load validation data if provided
    val_peptides = None
    val_labels = None
    if args.val_data:
        print(f"Loading validation data from {args.val_data}...")
        val_peptides = []
        val_labels = []
        with open(args.val_data, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                val_peptides.append(row['peptide'])
                val_labels.append(float(row['label']))
        print(f"  Loaded {len(val_peptides)} validation samples")

    # Create model
    print()
    print(f"Training {args.model_type} model...")
    print(f"  K-mer size: {args.k}")
    print(f"  Embedding dim: {args.embedding_dim}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print()

    from .scorers.mlp import MLPScorer
    hidden_layers = tuple(int(x) for x in args.hidden_layers.split(','))
    print(f"  Hidden layers: {hidden_layers}")
    scorer = MLPScorer(
        k=args.k,
        hidden_layer_sizes=hidden_layers,
    )

    print()

    # Train
    scorer.train(
        peptides=peptides,
        labels=labels,
        val_peptides=val_peptides,
        val_labels=val_labels,
        epochs=args.epochs,
        learning_rate=args.lr,
        verbose=True,
    )

    # Save
    print()
    print(f"Saving model as '{args.name}'...")
    path = mm.save(scorer, args.name, overwrite=args.overwrite)
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
    from .scorers import list_scorers

    print("Available scorer types:")
    print()

    scorers = list_scorers()
    for name in scorers:
        if name in ('frequency', 'similarity'):
            print(f"  {name:<20} (lookup-based, no training)")
        else:
            print(f"  {name:<20} (ML-based, requires training)")

    print()
    print("Lookup-based scorers use fit() with a reference dataset.")
    print("ML-based scorers use train() with labeled peptide data.")

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
        elif args.data_command == 'index':
            return cmd_data_index(args)
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
