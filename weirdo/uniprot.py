"""Download SwissProt from UniProt and build the sequence table WEIRDO trains on.

UniProt publishes the manually reviewed UniProtKB/Swiss-Prot entries split into taxonomic
divisions (``uniprot_sprot_<division>.dat.gz``). The ten divisions are WEIRDO's species
categories: archaea, bacteria, fungi, human, invertebrates, mammals, plants, rodents,
vertebrates and viruses.

Files are downloaded with :mod:`datacache`: transfers are retried and can resume, the size
and MD5 listed in UniProt's ``RELEASE.metalink`` are checked, and a file that is already
cached is not downloaded again. Each release is kept in its own directory, so a model can be
traced to the UniProt release it was built from.

Example
-------
>>> from weirdo.uniprot import fetch_swissprot_divisions, build_sequence_table
>>> release = fetch_swissprot_divisions("downloads")           # doctest: +SKIP
>>> build_sequence_table(release.paths, "swissprot-sequences.csv")  # doctest: +SKIP

UniProt data are licensed CC BY 4.0 (https://www.uniprot.org/help/license).
"""

import csv
import gzip
import hashlib
import json
import logging
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, Mapping, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

SWISSPROT_DIVISIONS: Tuple[str, ...] = (
    "archaea", "bacteria", "fungi", "human", "invertebrates",
    "mammals", "plants", "rodents", "vertebrates", "viruses",
)

#: Where UniProt publishes the division files of its current release. Set the
#: ``WEIRDO_UNIPROT_URL`` environment variable (or pass ``base_url``) to use a mirror such as
#: ``https://ftp.ebi.ac.uk/pub/databases/uniprot/current_release/knowledgebase/taxonomic_divisions``.
UNIPROT_BASE_URL = (
    "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/taxonomic_divisions"
)
BASE_URL_ENV = "WEIRDO_UNIPROT_URL"
METALINK_NAME = "RELEASE.metalink"
RELEASE_RECORD = "release.json"
SEQUENCE_COLUMNS = ("identifier", "sequence", "category")

_METALINK_NAMESPACE = "{http://www.metalinker.org/}"
_MAX_METALINK_BYTES = 5_000_000


def division_filename(division: str) -> str:
    """File name of a SwissProt division, e.g. ``uniprot_sprot_viruses.dat.gz``."""
    return f"uniprot_sprot_{division}.dat.gz"


def resolve_base_url(base_url: Optional[str] = None) -> str:
    """The URL of the division files: the argument, ``$WEIRDO_UNIPROT_URL`` or UniProt's."""
    return (base_url or os.environ.get(BASE_URL_ENV) or UNIPROT_BASE_URL).rstrip("/")


@dataclass(frozen=True)
class UniProtFile:
    """Size in bytes and MD5 of a file as listed by UniProt."""

    size: int
    md5: Optional[str] = None


@dataclass(frozen=True)
class UniProtRelease:
    """The release UniProt currently serves and the files it lists."""

    version: str
    files: Mapping[str, UniProtFile]


@dataclass(frozen=True)
class DownloadedRelease:
    """Division files of one UniProt release on disk."""

    version: str
    base_url: str
    paths: Mapping[str, Path]
    files: Mapping[str, UniProtFile] = field(default_factory=dict)


def parse_release_metalink(data: bytes) -> UniProtRelease:
    """Parse UniProt's ``RELEASE.metalink``: the release version and each file's size and MD5."""
    if b"<!DOCTYPE" in data or b"<!ENTITY" in data:
        raise ValueError("Refusing to parse a metalink that declares a DOCTYPE or entities.")
    try:
        root = ET.fromstring(data)
    except ET.ParseError as error:
        raise ValueError(f"UniProt's release metalink is not valid XML: {error}") from error
    version = root.findtext(f"{_METALINK_NAMESPACE}version")
    if not version:
        raise ValueError("UniProt's release metalink does not state a version.")
    files: Dict[str, UniProtFile] = {}
    for element in root.iter(f"{_METALINK_NAMESPACE}file"):
        name = element.get("name")
        size = element.findtext(f"{_METALINK_NAMESPACE}size")
        if not name or size is None:
            continue
        md5 = None
        for digest in element.iter(f"{_METALINK_NAMESPACE}hash"):
            if (digest.get("type") or "").lower() == "md5" and digest.text:
                md5 = digest.text.strip().lower()
        files[name] = UniProtFile(size=int(size), md5=md5)
    return UniProtRelease(version=version.strip(), files=files)


def fetch_release(base_url: Optional[str] = None, timeout: float = 60.0) -> UniProtRelease:
    """Read the release listing from UniProt (a small download, not cached)."""
    from datacache import fetch_bytes

    url = f"{resolve_base_url(base_url)}/{METALINK_NAME}"
    return parse_release_metalink(fetch_bytes(url, timeout=timeout, max_bytes=_MAX_METALINK_BYTES))


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _release_dir(download_dir: Path, version: str) -> Path:
    return download_dir / f"release-{version}"


def _cached_release(
    download_dir: Path, version: str, divisions: Sequence[str]
) -> Optional[DownloadedRelease]:
    """The cached release ``version`` if it holds every requested division, else ``None``."""
    directory = _release_dir(download_dir, version)
    record_path = directory / RELEASE_RECORD
    if not record_path.is_file():
        return None
    try:
        record = json.loads(record_path.read_text())
        files = {name: UniProtFile(**entry) for name, entry in record["files"].items()}
    except (OSError, ValueError, KeyError, TypeError):
        return None
    paths = {}
    for division in divisions:
        name = division_filename(division)
        path = directory / name
        if name not in files or not path.is_file() or path.stat().st_size != files[name].size:
            return None
        paths[division] = path
    return DownloadedRelease(version, str(record.get("base_url", "")), paths, files)


def _newest_cached_release(download_dir: Path, divisions: Sequence[str]) -> Optional[DownloadedRelease]:
    if not download_dir.is_dir():
        return None
    versions = sorted(
        (p.name[len("release-"):] for p in download_dir.glob("release-*") if p.is_dir()), reverse=True
    )
    for version in versions:
        cached = _cached_release(download_dir, version, divisions)
        if cached is not None:
            return cached
    return None


def _network_errors() -> tuple:
    import urllib.error

    import requests

    return (requests.exceptions.RequestException, urllib.error.URLError, OSError)


def fetch_swissprot_divisions(
    download_dir: Union[str, os.PathLike],
    divisions: Optional[Sequence[str]] = None,
    release: Optional[str] = None,
    base_url: Optional[str] = None,
    force: bool = False,
    show_progress: bool = False,
    timeout: float = 60.0,
) -> DownloadedRelease:
    """Download the SwissProt division files of UniProt's current release with ``datacache``.

    Files go to ``<download_dir>/release-<version>/``. A file that is already there with
    the size and MD5 UniProt lists is reused without downloading it again; interrupted
    transfers resume where the server allows it. If UniProt cannot be reached, the newest
    complete cached release is used instead (with a warning).

    Parameters
    ----------
    download_dir : path
        Directory that holds the downloads.
    divisions : sequence of str, optional
        Divisions to fetch (default: all ten).
    release : str, optional
        Require this release, e.g. ``'2023_02'``. UniProt serves the division files of its
        *current* release only, so an older release can only come from the cache.
    base_url : str, optional
        Alternative location of the division files (a mirror).
    force : bool, default=False
        Download again even if the files are cached.
    show_progress : bool, default=False
        Show a progress bar for each download.
    timeout : float, default=60
        Seconds without data before a transfer is retried or abandoned.

    Returns
    -------
    DownloadedRelease
        The release version and the local path of each division file.
    """
    from datacache import fetch_file

    divisions = tuple(divisions or SWISSPROT_DIVISIONS)
    unknown = sorted(set(divisions) - set(SWISSPROT_DIVISIONS))
    if unknown:
        raise ValueError(f"Unknown SwissProt divisions {unknown}; choose from {list(SWISSPROT_DIVISIONS)}")
    download_dir = Path(download_dir)
    base_url = resolve_base_url(base_url)

    if release is not None and not force:
        cached = _cached_release(download_dir, release, divisions)
        if cached is not None:
            return cached

    try:
        listing = fetch_release(base_url, timeout)
    except _network_errors() as error:
        cached = None if (release is not None or force) else _newest_cached_release(download_dir, divisions)
        if cached is None:
            raise ConnectionError(
                f"Could not read UniProt's release listing at {base_url}/{METALINK_NAME} "
                f"({error}) and no complete release is cached in {download_dir}."
            ) from error
        logger.warning(
            "Could not reach UniProt (%s); using the cached release %s.", error, cached.version
        )
        return cached
    if release is not None and listing.version != release:
        raise ValueError(
            f"UniProt serves the division files of release {listing.version} only, and "
            f"release {release} is not cached in {download_dir}."
        )

    directory = _release_dir(download_dir, listing.version)
    paths: Dict[str, Path] = {}
    for division in divisions:
        name = division_filename(division)
        if name not in listing.files:
            raise ValueError(
                f"UniProt release {listing.version} does not list {name} at {base_url}."
            )
        entry = listing.files[name]
        path = Path(fetch_file(
            f"{base_url}/{name}",
            destination=directory / name,
            raw=True,
            expected_size=entry.size,
            force=force,
            timeout=timeout,
            resume="auto",
            show_progress=show_progress,
            record_provenance=True,
        ))
        if entry.md5 is not None and _md5(path) != entry.md5:
            raise ValueError(
                f"{path} does not match the MD5 UniProt lists for {name}; download it again "
                "with force=True."
            )
        paths[division] = path

    record_path = directory / RELEASE_RECORD
    existing = {}
    if record_path.is_file():
        try:
            existing = json.loads(record_path.read_text()).get("files", {})
        except (OSError, ValueError):
            existing = {}
    existing.update({division_filename(d): {"size": listing.files[division_filename(d)].size,
                                            "md5": listing.files[division_filename(d)].md5}
                     for d in divisions})
    record_path.write_text(json.dumps({
        "version": listing.version,
        "base_url": base_url,
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "files": existing,
    }, indent=1))
    return DownloadedRelease(
        listing.version, base_url, paths, {division_filename(d): listing.files[division_filename(d)]
                                           for d in divisions},
    )


def iter_swissprot_sequences(path: Union[str, os.PathLike]) -> Iterator[Tuple[str, str]]:
    """Stream ``(entry name, sequence)`` pairs from a SwissProt ``.dat`` or ``.dat.gz`` file.

    The entry name is the UniProt identifier, e.g. ``1433B_BOVIN``. The sequence length is
    checked against the length stated in the entry.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    identifier: Optional[str] = None
    expected: Optional[int] = None
    chunks = []
    with opener(path, "rb") as handle:
        for line in handle:
            tag = line[:2]
            if tag == b"ID":
                identifier = line.split(None, 2)[1].decode("ascii")
            elif tag == b"SQ":
                expected = int(line.split()[2])
                chunks = []
            elif tag == b"//":
                if identifier is not None and expected is not None:
                    sequence = b"".join(chunks).translate(None, b" \t\r\n")
                    if len(sequence) != expected:
                        raise ValueError(
                            f"{path}: entry {identifier} states {expected} residues but has {len(sequence)}"
                        )
                    yield identifier, sequence.decode("ascii")
                identifier = expected = None
                chunks = []
            elif expected is not None and line[:1] == b" ":
                chunks.append(line)


def build_sequence_table(
    division_paths: Mapping[str, Union[str, os.PathLike]],
    output_path: Union[str, os.PathLike],
    progress: Optional[Callable[[str], None]] = None,
) -> int:
    """Write the protein sequences of the division files as ``identifier,sequence,category``.

    ``category`` is the division, so the table has the label of every protein. The file is
    written to a temporary name and moved into place when it is complete.

    Returns
    -------
    n_proteins : int
        Number of sequences written.
    """
    log = progress or (lambda message: None)
    unknown = sorted(set(division_paths) - set(SWISSPROT_DIVISIONS))
    if unknown:
        raise ValueError(f"Unknown SwissProt divisions: {unknown}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".partial")
    total = 0
    try:
        with open(temporary, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(SEQUENCE_COLUMNS)
            for division in SWISSPROT_DIVISIONS:
                if division not in division_paths:
                    continue
                count = 0
                for identifier, sequence in iter_swissprot_sequences(division_paths[division]):
                    writer.writerow((identifier, sequence, division))
                    count += 1
                total += count
                log(f"{division}: {count:,} proteins")
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return total


def read_sequence_table(source):
    """Read ``identifier,sequence,category`` rows into a DataFrame.

    Accepts a path (also the older layout with an unnamed leading index column) or a DataFrame.
    """
    import pandas as pd

    if hasattr(source, "columns"):
        frame = source
    else:
        frame = pd.read_csv(
            source,
            usecols=lambda column: column in SEQUENCE_COLUMNS,
            dtype=str,
            keep_default_na=False,
        )
    missing = [column for column in SEQUENCE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"The sequence table lacks columns {missing}.")
    return frame[list(SEQUENCE_COLUMNS)]
