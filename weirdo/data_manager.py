"""Data management for WEIRDO reference data.

The reference data are built from UniProtKB/Swiss-Prot. ``DataManager.download`` fetches the
SwissProt taxonomic-division files from UniProt with :mod:`datacache` (checking the sizes and
MD5s UniProt lists), builds the protein-sequence table from them and, on request, the k-mer
reference table. Data is stored in ~/.weirdo/ by default.

Example
-------
>>> from weirdo.data_manager import DataManager
>>> dm = DataManager()
>>> dm.download('swissprot-8mers')  # Download SwissProt from UniProt and build the table
>>> dm.status()  # Show what's downloaded
>>> dm.clear_all()  # Remove all data

UniProt data are licensed CC BY 4.0 (https://www.uniprot.org/help/license).
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# Default data directory
DEFAULT_DATA_DIR = Path.home() / '.weirdo'

# Available datasets with their metadata. Both are built from the UniProt SwissProt
# taxonomic-division files (see weirdo.uniprot); there is nothing to download from anywhere
# else.
DATASETS = {
    'swissprot-sequences': {
        'description': 'SwissProt protein sequences with their UniProt taxonomic division (~570k proteins)',
        'filename': 'swissprot-sequences.csv',
        'source': 'UniProt taxonomic-division files (uniprot_sprot_<division>.dat.gz)',
        'download_mb': 670,
        'size_mb': 230,
    },
    'swissprot-8mers': {
        'description': 'SwissProt 8-mer reference data (~100M k-mers), built from swissprot-sequences',
        'filename': 'swissprot-8mers.csv',
        'source': 'built from swissprot-sequences',
        'requires': 'swissprot-sequences',
        'k': 8,
        'download_mb': 0,
        'size_mb': 8000,
    },
}


class DataManager:
    """Manages WEIRDO reference data.

    Parameters
    ----------
    data_dir : str or Path, optional
        Directory for storing data. Defaults to ~/.weirdo/
    auto_download : bool, default=False
        If True, automatically download missing data when needed.
    verbose : bool, default=True
        Print status messages.
    uniprot_url : str, optional
        Where to fetch the SwissProt division files from (default: UniProt's FTP site, or
        ``$WEIRDO_UNIPROT_URL``); useful for a mirror.

    Example
    -------
    >>> dm = DataManager()
    >>> dm.download('swissprot-8mers')
    >>> print(dm.status())
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        auto_download: bool = False,
        verbose: bool = True,
        uniprot_url: Optional[str] = None,
    ):
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.auto_download = auto_download
        self.verbose = verbose
        self.uniprot_url = uniprot_url

        # Subdirectories
        self.downloads_dir = self.data_dir / 'downloads'
        self.uniprot_dir = self.downloads_dir / 'uniprot'  # raw UniProt files, one directory per release
        self.metadata_file = self.data_dir / 'metadata.json'

        # Ensure directories exist
        self._ensure_dirs()

        # Load metadata
        self._metadata = self._load_metadata()

    def _ensure_dirs(self) -> None:
        """Create data directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.downloads_dir.mkdir(exist_ok=True)

    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
            if 'downloads' not in data:
                data['downloads'] = {}
            return data
        return {'downloads': {}}

    def _save_metadata(self) -> None:
        """Save metadata to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self._metadata, f, indent=2, default=str)

    def reset_metadata(self) -> None:
        """Reset in-memory metadata and remove metadata file from disk."""
        self._metadata = {'downloads': {}}
        if self.metadata_file.exists():
            self.metadata_file.unlink()

    def _log(self, msg: str) -> None:
        """Print message if verbose."""
        if self.verbose:
            print(msg)

    # -------------------------------------------------------------------------
    # Dataset queries
    # -------------------------------------------------------------------------

    def list_available_datasets(self) -> List[str]:
        """List available datasets that can be downloaded.

        Returns
        -------
        datasets : list of str
            Names of available datasets.
        """
        return list(DATASETS.keys())

    def get_dataset_info(self, name: str) -> Dict[str, Any]:
        """Get information about a dataset.

        Parameters
        ----------
        name : str
            Dataset name.

        Returns
        -------
        info : dict
            Dataset metadata.
        """
        if name not in DATASETS:
            raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
        return DATASETS[name].copy()

    # -------------------------------------------------------------------------
    # Download management
    # -------------------------------------------------------------------------

    def is_downloaded(self, name: str) -> bool:
        """Check if a dataset is downloaded.

        Parameters
        ----------
        name : str
            Dataset name.

        Returns
        -------
        downloaded : bool
            True if dataset is downloaded.
        """
        if name not in DATASETS:
            raise ValueError(f"Unknown dataset: {name}")

        filepath = self.downloads_dir / DATASETS[name]['filename']
        return filepath.exists() and name in self._metadata.get('downloads', {})

    def get_data_path(self, name: str, auto_download: Optional[bool] = None) -> Path:
        """Get path to a dataset, optionally downloading if missing.

        Parameters
        ----------
        name : str
            Dataset name.
        auto_download : bool, optional
            Override instance auto_download setting.

        Returns
        -------
        path : Path
            Path to the dataset file.

        Raises
        ------
        FileNotFoundError
            If dataset is not downloaded and auto_download is False.
        """
        if name not in DATASETS:
            raise ValueError(f"Unknown dataset: {name}")

        filepath = self.downloads_dir / DATASETS[name]['filename']

        if not filepath.exists():
            should_download = auto_download if auto_download is not None else self.auto_download
            if should_download:
                self.download(name)
            else:
                raise FileNotFoundError(
                    f"Dataset '{name}' not found. Run: weirdo data download {name}\n"
                    f"Or use DataManager(auto_download=True)"
                )

        return filepath

    def download(self, name: str, force: bool = False) -> Path:
        """Download a dataset from UniProt and build it.

        ``'swissprot-sequences'`` downloads the SwissProt division files (about 670 MB)
        and writes the sequence table; ``'swissprot-8mers'`` builds the k-mer table from
        the sequence table (downloading that first when it is missing) and needs several GB
        of disk space and some minutes.

        Parameters
        ----------
        name : str
            Dataset name.
        force : bool, default=False
            Download and build again even if the dataset is present.

        Returns
        -------
        path : Path
            Path to the dataset file.
        """
        if name not in DATASETS:
            raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")

        info = DATASETS[name]
        filepath = self.downloads_dir / info['filename']

        if filepath.exists() and not force:
            self._log(f"Dataset '{name}' already downloaded at {filepath}")
            return filepath

        self._log(f"Building '{name}' ({info['source']})...")
        try:
            if name == 'swissprot-sequences':
                record = self._download_sequences(filepath, force)
            else:
                record = self._build_kmer_table(name, filepath, force)
        except (ConnectionError, OSError, ValueError) as e:
            raise RuntimeError(f"Failed to download '{name}': {e}") from e

        self._metadata.setdefault('downloads', {})[name] = {
            'path': str(filepath),
            'downloaded_at': datetime.now().isoformat(),
            'size_bytes': filepath.stat().st_size,
            **record,
        }
        self._save_metadata()
        self._log(f"  Wrote: {filepath}")
        return filepath

    def _download_sequences(self, filepath: Path, force: bool) -> Dict[str, Any]:
        from .uniprot import build_sequence_table, fetch_swissprot_divisions

        release = fetch_swissprot_divisions(
            self.uniprot_dir,
            base_url=self.uniprot_url,
            force=force,
            show_progress=self.verbose,
        )
        self._log(f"  UniProt release {release.version}; reading the division files...")
        n_proteins = build_sequence_table(
            release.paths, filepath, progress=lambda message: self._log(f"    {message}")
        )
        return {
            'uniprot_release': release.version,
            'source_url': release.base_url,
            'n_proteins': n_proteins,
        }

    def _build_kmer_table(self, name: str, filepath: Path, force: bool) -> Dict[str, Any]:
        from .kmer_table import build_kmer_table

        info = DATASETS[name]
        sequences = self.get_data_path(info['requires'], auto_download=True)
        origin = self._metadata.get('downloads', {}).get(info['requires'], {})
        self._log(f"  Building the {info['k']}-mer table from {sequences}...")
        n_rows = build_kmer_table(
            sequences, filepath, k=info['k'],
            progress=lambda message: self._log(f"    {message}"),
        )
        record = {'built_from': info['requires'], 'k': info['k'], 'n_kmers': n_rows}
        for key in ('uniprot_release', 'source_url'):
            if key in origin:
                record[key] = origin[key]
        return record

    def download_all(self, force: bool = False) -> List[Path]:
        """Download all available datasets.

        Parameters
        ----------
        force : bool, default=False
            Re-download even if already present.

        Returns
        -------
        paths : list of Path
            Paths to downloaded files.
        """
        paths = []
        for name in DATASETS:
            paths.append(self.download(name, force=force))
        return paths

    def delete_download(self, name: str) -> bool:
        """Delete a downloaded dataset.

        Parameters
        ----------
        name : str
            Dataset name.

        Returns
        -------
        deleted : bool
            True if file was deleted.
        """
        if name not in DATASETS:
            raise ValueError(f"Unknown dataset: {name}")

        filepath = self.downloads_dir / DATASETS[name]['filename']
        deleted = False

        if filepath.exists():
            filepath.unlink()
            deleted = True
            self._log(f"Deleted: {filepath}")

        if name in self._metadata.get('downloads', {}):
            del self._metadata['downloads'][name]
            self._save_metadata()

        return deleted

    def delete_all_downloads(self) -> int:
        """Delete all downloaded datasets.

        Returns
        -------
        count : int
            Number of files deleted.
        """
        count = 0
        for name in list(DATASETS.keys()):
            if self.delete_download(name):
                count += 1
        return count

    # -------------------------------------------------------------------------
    # Status and cleanup
    # -------------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Get status of all downloaded data.

        Returns
        -------
        status : dict
            Status information.
        """
        status = {
            'data_dir': str(self.data_dir),
            'downloads': {},
            'total_size_bytes': 0,
        }

        for name in DATASETS:
            filepath = self.downloads_dir / DATASETS[name]['filename']
            if filepath.exists():
                size = filepath.stat().st_size
                status['downloads'][name] = {
                    'path': str(filepath),
                    'size_bytes': size,
                    'size_mb': size / (1024 * 1024),
                    'metadata': self._metadata.get('downloads', {}).get(name, {}),
                }
                status['total_size_bytes'] += size
            else:
                status['downloads'][name] = {'downloaded': False}

        raw_bytes = sum(
            path.stat().st_size for path in self.uniprot_dir.rglob('*') if path.is_file()
        ) if self.uniprot_dir.exists() else 0
        status['raw_uniprot_bytes'] = raw_bytes
        status['total_size_bytes'] += raw_bytes
        status['total_size_mb'] = status['total_size_bytes'] / (1024 * 1024)
        return status

    def print_status(self) -> None:
        """Print human-readable status."""
        status = self.status()

        print(f"\nWEIRDO Data Directory: {status['data_dir']}")
        print(f"Total Disk Usage: {status['total_size_mb']:.1f} MB")

        print("\nDatasets:")
        print("-" * 70)
        for name, info in status['downloads'].items():
            ds_info = DATASETS.get(name, {})
            if info.get('downloaded', True) and 'size_mb' in info:
                meta = info.get('metadata', {})
                downloaded_at = meta.get('downloaded_at', 'unknown')[:10]  # Just date
                print(f"  [x] {name}")
                print(f"      {ds_info.get('description', '')}")
                print(f"      Status: Downloaded ({info['size_mb']:.0f} MB, {downloaded_at})")
            else:
                print(f"  [ ] {name}")
                print(f"      {ds_info.get('description', '')}")
                print(f"      Status: Not downloaded (~{ds_info.get('size_mb', '?')} MB on disk)")

        print()

    def clear_all(self, include_metadata: bool = False) -> int:
        """Delete all downloaded data.

        Parameters
        ----------
        include_metadata : bool, default=False
            If True, also delete metadata.json and reset metadata state.

        Returns
        -------
        count : int
            Number of downloads deleted.
        """
        count = self.delete_all_downloads()
        if self.uniprot_dir.exists():
            shutil.rmtree(self.uniprot_dir)
            self._log(f"Deleted: {self.uniprot_dir}")
        if include_metadata:
            self.reset_metadata()
        return count

    def disk_usage(self) -> int:
        """Get total disk usage in bytes.

        Returns
        -------
        bytes : int
            Total disk usage.
        """
        return self.status()['total_size_bytes']


# Singleton instance for convenience
_default_manager: Optional[DataManager] = None


def get_data_manager(
    data_dir: Optional[Path] = None,
    auto_download: bool = False,
    verbose: bool = True,
) -> DataManager:
    """Get the default DataManager instance.

    Parameters
    ----------
    data_dir : Path, optional
        Override data directory.
    auto_download : bool, default=False
        Enable auto-download.
    verbose : bool, default=True
        Print status messages.

    Returns
    -------
    manager : DataManager
        Data manager instance.
    """
    global _default_manager

    if _default_manager is None or data_dir is not None:
        _default_manager = DataManager(
            data_dir=data_dir,
            auto_download=auto_download,
            verbose=verbose,
        )

    return _default_manager


def ensure_data_available(auto_download: bool = False) -> Path:
    """Ensure reference data is available, optionally downloading.

    Parameters
    ----------
    auto_download : bool, default=False
        Download if not present.

    Returns
    -------
    path : Path
        Path to reference data.
    """
    dm = get_data_manager(auto_download=auto_download)
    return dm.get_data_path('swissprot-8mers')
