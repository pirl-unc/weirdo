"""Data management for WEIRDO reference data and indices.

Handles downloading, caching, and indexing of reference data files.
Data is stored in ~/.weirdo/ by default.

Example
-------
>>> from weirdo.data_manager import DataManager
>>> dm = DataManager()
>>> dm.download('swissprot-8mers')  # Download reference data
>>> dm.status()  # Show what's downloaded
>>> dm.clear_all()  # Remove all data
"""

import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import urlretrieve
from urllib.error import URLError


# Default data directory
DEFAULT_DATA_DIR = Path.home() / '.weirdo'

# Available datasets with their metadata
DATASETS = {
    'swissprot-8mers': {
        'description': 'SwissProt 8-mer reference data (~100M k-mers)',
        'url': 'https://github.com/pirl-unc/weirdo-data/releases/download/v1.0/swissprot-8mers.csv.gz',
        'filename': 'swissprot-8mers.csv',
        'compressed': True,
        'size_mb': 2500,  # Compressed size
        'uncompressed_size_mb': 7500,
        'sha256': None,  # Will be verified if provided
    },
}

# Index types that can be generated
INDEX_TYPES = {
    'frequency': {
        'description': 'Pre-computed frequency lookup for fast scoring',
        'source': 'swissprot-8mers',
    },
    'set': {
        'description': 'K-mer presence set (memory efficient)',
        'source': 'swissprot-8mers',
    },
}


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    """Progress callback for urlretrieve."""
    if total_size > 0:
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f'\r  Downloading: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)')
        sys.stdout.flush()
    else:
        downloaded = block_num * block_size
        mb_downloaded = downloaded / (1024 * 1024)
        sys.stdout.write(f'\r  Downloading: {mb_downloaded:.1f} MB')
        sys.stdout.flush()


class DataManager:
    """Manages WEIRDO reference data and indices.

    Parameters
    ----------
    data_dir : str or Path, optional
        Directory for storing data. Defaults to ~/.weirdo/
    auto_download : bool, default=False
        If True, automatically download missing data when needed.
    verbose : bool, default=True
        Print status messages.

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
    ):
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.auto_download = auto_download
        self.verbose = verbose

        # Subdirectories
        self.downloads_dir = self.data_dir / 'downloads'
        self.indices_dir = self.data_dir / 'indices'
        self.metadata_file = self.data_dir / 'metadata.json'

        # Ensure directories exist
        self._ensure_dirs()

        # Load metadata
        self._metadata = self._load_metadata()

    def _ensure_dirs(self) -> None:
        """Create data directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.downloads_dir.mkdir(exist_ok=True)
        self.indices_dir.mkdir(exist_ok=True)

    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {'downloads': {}, 'indices': {}}

    def _save_metadata(self) -> None:
        """Save metadata to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self._metadata, f, indent=2, default=str)

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

    def list_available_indices(self) -> List[str]:
        """List available index types that can be generated.

        Returns
        -------
        indices : list of str
            Names of available index types.
        """
        return list(INDEX_TYPES.keys())

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
        """Download a dataset.

        Parameters
        ----------
        name : str
            Dataset name.
        force : bool, default=False
            Re-download even if already present.

        Returns
        -------
        path : Path
            Path to downloaded file.
        """
        if name not in DATASETS:
            raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")

        info = DATASETS[name]
        filepath = self.downloads_dir / info['filename']

        if filepath.exists() and not force:
            self._log(f"Dataset '{name}' already downloaded at {filepath}")
            return filepath

        self._log(f"Downloading '{name}'...")
        self._log(f"  Source: {info['url']}")
        self._log(f"  Size: ~{info['size_mb']} MB (compressed)")

        # Download to temp file first
        temp_path = None
        try:
            if info.get('compressed', False):
                # Download compressed file
                temp_fd, temp_path = tempfile.mkstemp(suffix='.gz')
                os.close(temp_fd)
                urlretrieve(info['url'], temp_path, _progress_hook)
                print()  # Newline after progress

                # Decompress
                self._log("  Decompressing...")
                import gzip
                with gzip.open(temp_path, 'rb') as f_in:
                    with open(filepath, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.unlink(temp_path)
            else:
                urlretrieve(info['url'], filepath, _progress_hook)
                print()  # Newline after progress

            # Update metadata
            self._metadata.setdefault('downloads', {})[name] = {
                'path': str(filepath),
                'downloaded_at': datetime.now().isoformat(),
                'size_bytes': filepath.stat().st_size,
            }
            self._save_metadata()

            self._log(f"  Downloaded to: {filepath}")
            return filepath

        except URLError as e:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            raise RuntimeError(f"Failed to download '{name}': {e}")

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

        # Also delete any indices that depend on this dataset
        indices_to_delete = [
            idx_name for idx_name, idx_info in INDEX_TYPES.items()
            if idx_info.get('source') == name
        ]
        for idx_name in indices_to_delete:
            self.delete_index(idx_name)

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
    # Index management
    # -------------------------------------------------------------------------

    def is_indexed(self, name: str) -> bool:
        """Check if an index exists.

        Parameters
        ----------
        name : str
            Index name.

        Returns
        -------
        indexed : bool
            True if index exists.
        """
        if name not in INDEX_TYPES:
            raise ValueError(f"Unknown index: {name}")

        index_path = self.indices_dir / f"{name}.idx"
        return index_path.exists() and name in self._metadata.get('indices', {})

    def get_index_path(self, name: str) -> Path:
        """Get path to an index file.

        Parameters
        ----------
        name : str
            Index name.

        Returns
        -------
        path : Path
            Path to index file.
        """
        if name not in INDEX_TYPES:
            raise ValueError(f"Unknown index: {name}")
        return self.indices_dir / f"{name}.idx"

    def build_index(self, name: str, force: bool = False) -> Path:
        """Build an index from downloaded data.

        Parameters
        ----------
        name : str
            Index name.
        force : bool, default=False
            Rebuild even if already exists.

        Returns
        -------
        path : Path
            Path to index file.
        """
        if name not in INDEX_TYPES:
            raise ValueError(f"Unknown index: {name}. Available: {list(INDEX_TYPES.keys())}")

        index_path = self.indices_dir / f"{name}.idx"

        if index_path.exists() and not force:
            self._log(f"Index '{name}' already exists at {index_path}")
            return index_path

        info = INDEX_TYPES[name]
        source = info['source']

        # Ensure source data is downloaded
        source_path = self.get_data_path(source)

        self._log(f"Building index '{name}'...")
        self._log(f"  Source: {source_path}")

        start_time = time.time()

        if name == 'frequency':
            self._build_frequency_index(source_path, index_path)
        elif name == 'set':
            self._build_set_index(source_path, index_path)
        else:
            raise NotImplementedError(f"Index type '{name}' not implemented")

        elapsed = time.time() - start_time

        # Update metadata
        self._metadata.setdefault('indices', {})[name] = {
            'path': str(index_path),
            'built_at': datetime.now().isoformat(),
            'source': source,
            'size_bytes': index_path.stat().st_size,
            'build_time_seconds': elapsed,
        }
        self._save_metadata()

        self._log(f"  Built in {elapsed:.1f}s: {index_path}")
        return index_path

    def _build_frequency_index(self, source_path: Path, index_path: Path) -> None:
        """Build frequency lookup index."""
        import pickle

        frequencies = {}
        total_count = 0

        with open(source_path, 'r') as f:
            # Skip header
            header = f.readline().strip().split(',')

            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    kmer = parts[1]  # seq column
                    # Count number of True values (categories present)
                    count = sum(1 for p in parts[2:-1] if p == 'True')
                    frequencies[kmer] = count
                    total_count += 1

                if total_count % 10000000 == 0:
                    self._log(f"    Processed {total_count:,} k-mers...")

        # Normalize to frequencies
        max_count = max(frequencies.values()) if frequencies else 1
        frequencies = {k: v / max_count for k, v in frequencies.items()}

        with open(index_path, 'wb') as f:
            pickle.dump(frequencies, f)

    def _build_set_index(self, source_path: Path, index_path: Path) -> None:
        """Build k-mer presence set index."""
        import pickle

        kmers = set()
        total_count = 0

        with open(source_path, 'r') as f:
            # Skip header
            f.readline()

            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    kmer = parts[1]  # seq column
                    kmers.add(kmer)
                    total_count += 1

                if total_count % 10000000 == 0:
                    self._log(f"    Processed {total_count:,} k-mers...")

        with open(index_path, 'wb') as f:
            pickle.dump(kmers, f)

    def delete_index(self, name: str) -> bool:
        """Delete an index.

        Parameters
        ----------
        name : str
            Index name.

        Returns
        -------
        deleted : bool
            True if index was deleted.
        """
        if name not in INDEX_TYPES:
            raise ValueError(f"Unknown index: {name}")

        index_path = self.indices_dir / f"{name}.idx"
        deleted = False

        if index_path.exists():
            index_path.unlink()
            deleted = True
            self._log(f"Deleted index: {index_path}")

        if name in self._metadata.get('indices', {}):
            del self._metadata['indices'][name]
            self._save_metadata()

        return deleted

    def delete_all_indices(self) -> int:
        """Delete all indices.

        Returns
        -------
        count : int
            Number of indices deleted.
        """
        count = 0
        for name in list(INDEX_TYPES.keys()):
            if self.delete_index(name):
                count += 1
        return count

    def rebuild_indices(self) -> List[Path]:
        """Rebuild all indices from downloaded data.

        Returns
        -------
        paths : list of Path
            Paths to rebuilt indices.
        """
        paths = []
        for name in INDEX_TYPES:
            source = INDEX_TYPES[name]['source']
            if self.is_downloaded(source):
                paths.append(self.build_index(name, force=True))
        return paths

    # -------------------------------------------------------------------------
    # Status and cleanup
    # -------------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Get status of all data and indices.

        Returns
        -------
        status : dict
            Status information.
        """
        status = {
            'data_dir': str(self.data_dir),
            'downloads': {},
            'indices': {},
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

        for name in INDEX_TYPES:
            index_path = self.indices_dir / f"{name}.idx"
            if index_path.exists():
                size = index_path.stat().st_size
                status['indices'][name] = {
                    'path': str(index_path),
                    'size_bytes': size,
                    'size_mb': size / (1024 * 1024),
                    'metadata': self._metadata.get('indices', {}).get(name, {}),
                }
                status['total_size_bytes'] += size
            else:
                status['indices'][name] = {'built': False}

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
                print(f"      Status: Not downloaded (~{ds_info.get('size_mb', '?')} MB compressed)")

        print("\nIndices:")
        print("-" * 70)
        for name, info in status['indices'].items():
            idx_info = INDEX_TYPES.get(name, {})
            if info.get('built', True) and 'size_mb' in info:
                meta = info.get('metadata', {})
                built_at = meta.get('built_at', 'unknown')[:10]  # Just date
                print(f"  [x] {name}")
                print(f"      {idx_info.get('description', '')}")
                print(f"      Status: Built ({info['size_mb']:.0f} MB, {built_at})")
            else:
                print(f"  [ ] {name}")
                print(f"      {idx_info.get('description', '')}")
                print(f"      Status: Not built (requires: {idx_info.get('source', '?')})")
        print()

    def clear_all(self) -> Tuple[int, int]:
        """Delete all downloaded data and indices.

        Returns
        -------
        counts : tuple of (int, int)
            Number of (downloads, indices) deleted.
        """
        idx_count = self.delete_all_indices()
        dl_count = self.delete_all_downloads()
        return (dl_count, idx_count)

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
