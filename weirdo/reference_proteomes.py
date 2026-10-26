"""Fetch complete, pinned UniProt reference proteomes for host-self lookup.

SwissProt is excellent for a compact, reviewed taxonomy corpus, but it is not a
complete proteome for most non-human hosts. This module keeps a small, explicit
manifest of complete UniProt reference proteomes and downloads their FASTA files
through :mod:`datacache`.

The downloaded sequence set is intended for *exact self lookup*. A taxonomy
classifier should sample and balance these records by species instead of using
their raw record counts as training weights.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence, Union
from urllib.parse import urlencode


UNIPROT_REFERENCE_PROTEOME_STREAM_URL = "https://rest.uniprot.org/uniprotkb/stream"
UNIPROT_REFERENCE_PROTEOME_METADATA_URL = "https://rest.uniprot.org/proteomes"
REFERENCE_PROTEOME_RECORD_SUFFIX = ".json"
REFERENCE_PROTEOME_MANIFEST_DIR = Path(__file__).with_name("reference_proteome_manifests")


@dataclass(frozen=True)
class ReferenceProteome:
    """A pinned UniProt reference proteome for one host species."""

    key: str
    scientific_name: str
    taxon_id: int
    proteome_id: str
    catalog_modified: Optional[str] = None


@dataclass(frozen=True)
class ReferenceProteomeManifestEntry:
    """One complete reference proteome and its explicit training memberships."""

    proteome: ReferenceProteome
    labels: tuple[str, ...]


@dataclass(frozen=True)
class ReferenceProteomeManifest:
    """A frozen selection of complete UniProt reference proteomes."""

    name: str
    entries: tuple[ReferenceProteomeManifestEntry, ...]

    @property
    def groups(self) -> tuple[str, ...]:
        """Return every selectable label in stable manifest order."""
        return tuple(dict.fromkeys(label for entry in self.entries for label in entry.labels))

    def select(self, groups: Sequence[str] = ()) -> tuple[ReferenceProteomeManifestEntry, ...]:
        """Return entries that belong to any requested manifest group."""
        selected_groups = tuple(dict.fromkeys(groups))
        if not selected_groups:
            return self.entries
        unknown = sorted(set(selected_groups).difference(self.groups))
        if unknown:
            choices = ", ".join(self.groups)
            raise ValueError(
                f"Unknown group(s) for reference-proteome manifest {self.name!r}: "
                f"{', '.join(unknown)}. Choose from: {choices}."
            )
        wanted = set(selected_groups)
        return tuple(entry for entry in self.entries if wanted.intersection(entry.labels))


# These are selected complete reference proteomes, not searches over all
# UniProtKB records. Keeping the identifiers explicit makes a model or a
# self-reference result reproducible even when UniProt's catalog changes.
REFERENCE_PROTEOMES = (
    ReferenceProteome("human", "Homo sapiens", 9606, "UP000005640"),
    ReferenceProteome("dog", "Canis lupus familiaris", 9615, "UP000805418"),
    ReferenceProteome("cat", "Felis catus", 9685, "UP000823872"),
    ReferenceProteome("mouse", "Mus musculus", 10090, "UP000000589"),
    ReferenceProteome("rat", "Rattus norvegicus", 10116, "UP000002494"),
    ReferenceProteome("cattle", "Bos taurus", 9913, "UP000009136"),
)

_ALIASES = {
    "canis familiaris": "dog",
    "canis lupus familiaris": "dog",
    "felis catus": "cat",
    "mus musculus": "mouse",
    "rattus norvegicus": "rat",
    "homo sapiens": "human",
    "bos taurus": "cattle",
    "bovine": "cattle",
    "cow": "cattle",
}


@dataclass(frozen=True)
class DownloadedReferenceProteome:
    """A verified local reference-proteome FASTA and its content summary."""

    proteome: ReferenceProteome
    path: Path
    sha256: str
    proteins: int
    amino_acids: int
    source_url: str


def list_reference_proteomes() -> tuple[ReferenceProteome, ...]:
    """Return the built-in host reference-proteome manifest."""
    return REFERENCE_PROTEOMES


def reference_proteome_manifest_path(name: str = "reference-proteomes-v1") -> Path:
    """Return a packaged, frozen selection for reproducible future training.

    The selection is intentionally separate from the short host-alias manifest:
    it supplies many species and viral proteomes for corpus construction, while
    host aliases make exact-self lookup convenient.  Callers should record the
    selected FASTA checksums as well as the manifest version.
    """
    if not re.fullmatch(r"[a-z0-9-]+", name):
        raise ValueError(f"Invalid reference-proteome manifest name {name!r}.")
    path = REFERENCE_PROTEOME_MANIFEST_DIR / f"{name}.yaml"
    if not path.is_file():
        raise ValueError(f"Unknown reference-proteome manifest {name!r}.")
    return path


def list_reference_proteome_manifests() -> tuple[str, ...]:
    """Return the names of packaged frozen reference-proteome selections."""
    return tuple(sorted(path.stem for path in REFERENCE_PROTEOME_MANIFEST_DIR.glob("*.yaml")))


def _manifest_values(block: str) -> dict[str, str]:
    """Parse the deliberately small, stable mapping subset used by our manifest."""
    values = {
        key: value.strip().strip("'\"")
        for key, value in re.findall(r"^    ([a-z_]+): (.+)$", block, flags=re.M)
    }
    match = re.search(r"^  - key: (.+)$", block, flags=re.M)
    if match is not None:
        values["key"] = match.group(1).strip().strip("'\"")
    return values


def load_reference_proteome_manifest(
    name: str = "reference-proteomes-v1",
) -> ReferenceProteomeManifest:
    """Load a packaged frozen manifest without a runtime YAML dependency.

    The shipped manifest intentionally uses a narrow YAML subset (mammal mapping
    entries plus indented lists of viral accessions). Parsing that fixed schema
    here keeps ordinary CLI downloads dependency-free while rejecting malformed
    or duplicate accessions before any network access occurs.
    """
    text = reference_proteome_manifest_path(name).read_text(encoding="utf-8")
    manifest_name = re.search(r"^name: ([a-z0-9-]+)$", text, flags=re.M)
    mammals = re.search(r"^mammals:\n(.*?)^viruses:\n", text, flags=re.M | re.S)
    if manifest_name is None or mammals is None or manifest_name.group(1) != name:
        raise ValueError(f"Malformed reference-proteome manifest {name!r}.")

    entries: list[ReferenceProteomeManifestEntry] = []
    for block in re.split(r"(?=^  - key: )", mammals.group(1), flags=re.M):
        values = _manifest_values(block)
        if not values:
            continue
        required = {"key", "scientific_name", "taxon_id", "proteome_id", "taxonomy_leaf"}
        missing = sorted(required.difference(values))
        if missing:
            raise ValueError(
                f"Manifest {name!r} mammal entry is missing: {', '.join(missing)}."
            )
        proteome_id = values["proteome_id"].upper()
        if not re.fullmatch(r"UP\d{9}", proteome_id):
            raise ValueError(f"Manifest {name!r} has invalid proteome ID {proteome_id!r}.")
        try:
            taxon_id = int(values["taxon_id"])
        except ValueError as error:
            raise ValueError(
                f"Manifest {name!r} has invalid taxon ID {values['taxon_id']!r}."
            ) from error
        labels = ("mammals", values["taxonomy_leaf"])
        if "exact_self_host_alias" in values:
            labels += (values["exact_self_host_alias"],)
        entries.append(
            ReferenceProteomeManifestEntry(
                ReferenceProteome(
                    key=values["key"],
                    scientific_name=values["scientific_name"],
                    taxon_id=taxon_id,
                    proteome_id=proteome_id,
                ),
                labels,
            )
        )

    virus_text = text[mammals.end():]
    realm_matches = re.finditer(
        r"^  ([a-z][a-z_]*):\n((?:  - UP\d{9}\n?)+)", virus_text, flags=re.M
    )
    for match in realm_matches:
        realm, listed = match.groups()
        for proteome_id in re.findall(r"^  - (UP\d{9})$", listed, flags=re.M):
            entries.append(
                ReferenceProteomeManifestEntry(
                    ReferenceProteome(
                        key=f"{realm}-{proteome_id.lower()}",
                        scientific_name=f"{realm} reference proteome {proteome_id}",
                        taxon_id=10239,
                        proteome_id=proteome_id,
                    ),
                    ("viruses", realm),
                )
            )

    if not entries:
        raise ValueError(f"Manifest {name!r} contains no reference proteomes.")
    proteome_ids = [entry.proteome.proteome_id for entry in entries]
    if len(proteome_ids) != len(set(proteome_ids)):
        raise ValueError(f"Manifest {name!r} contains duplicate proteome IDs.")
    return ReferenceProteomeManifest(name=name, entries=tuple(entries))


def get_reference_proteome(host: Union[str, ReferenceProteome]) -> ReferenceProteome:
    """Resolve a built-in host name or return an explicit manifest entry."""
    if isinstance(host, ReferenceProteome):
        return host
    normalized = re.sub(r"[_-]+", " ", host.strip().lower())
    normalized = _ALIASES.get(normalized, normalized)
    for proteome in REFERENCE_PROTEOMES:
        if normalized == proteome.key:
            return proteome
    choices = ", ".join(item.key for item in REFERENCE_PROTEOMES)
    raise ValueError(f"Unknown host {host!r}; choose one of: {choices}.")


def reference_proteome_metadata_url(
    proteome_id: str,
    base_url: str = UNIPROT_REFERENCE_PROTEOME_METADATA_URL,
) -> str:
    """Return UniProt's metadata URL for one reference-proteome accession."""
    normalized = proteome_id.upper()
    if not re.fullmatch(r"UP\d{9}", normalized):
        raise ValueError(
            f"Invalid UniProt proteome ID {proteome_id!r}; expected an accession such as "
            "'UP000805418'."
        )
    return f"{base_url.rstrip('/')}/{normalized}"


def reference_proteome_from_uniprot(
    proteome_id: str,
    *,
    timeout: float = 60.0,
    url: Optional[str] = None,
) -> ReferenceProteome:
    """Resolve any UniProt *reference* proteome accession into a manifest entry.

    The catalog lookup is deliberately separate from the built-in aliases. It
    lets callers fetch any suitable species without embedding a changing copy of
    UniProt's whole catalog in this package. The downloaded FASTA's SHA-256 is
    recorded locally, so a training run records the exact bytes it used.
    """
    from datacache import fetch_bytes

    default_metadata_url = reference_proteome_metadata_url(proteome_id)
    metadata_url = url or default_metadata_url
    try:
        metadata = json.loads(
            fetch_bytes(metadata_url, timeout=timeout, max_bytes=5_000_000)
        )
    except (OSError, ValueError, TypeError) as error:
        raise ConnectionError(
            f"Could not read UniProt metadata for proteome {proteome_id!r} ({error})."
        ) from error
    if metadata.get("proteomeType") != "Reference proteome":
        raise ValueError(f"UniProt proteome {proteome_id!r} is not a reference proteome.")
    taxonomy = metadata.get("taxonomy", {})
    scientific_name = taxonomy.get("scientificName")
    taxon_id = taxonomy.get("taxonId")
    if not isinstance(scientific_name, str) or not isinstance(taxon_id, int):
        raise ValueError(f"UniProt metadata for proteome {proteome_id!r} lacks species taxonomy.")
    normalized_id = metadata.get("id", proteome_id).upper()
    for built_in in REFERENCE_PROTEOMES:
        if built_in.proteome_id == normalized_id:
            return built_in
    key = re.sub(r"[^a-z0-9]+", "_", scientific_name.lower()).strip("_")
    return ReferenceProteome(
        key=key,
        scientific_name=scientific_name,
        taxon_id=taxon_id,
        proteome_id=normalized_id,
        catalog_modified=metadata.get("modified"),
    )


def reference_proteome_url(
    proteome: Union[str, ReferenceProteome],
    base_url: str = UNIPROT_REFERENCE_PROTEOME_STREAM_URL,
) -> str:
    """Return the compressed UniProtKB FASTA stream URL for a reference proteome."""
    resolved = get_reference_proteome(proteome)
    parameters = {
        "compressed": "true",
        "format": "fasta",
        "query": f"proteome:{resolved.proteome_id}",
    }
    return f"{base_url}?{urlencode(parameters)}"


def reference_proteome_filename(proteome: Union[str, ReferenceProteome]) -> str:
    """Return the stable local FASTA filename for a manifest entry."""
    resolved = get_reference_proteome(proteome)
    return f"{resolved.key}-{resolved.proteome_id}.fasta.gz"


def iter_fasta_sequences(path: Union[str, os.PathLike]) -> Iterator[tuple[str, str]]:
    """Yield ``(identifier, sequence)`` from a plain or gzip-compressed FASTA file."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    identifier: Optional[str] = None
    chunks: list[str] = []
    with opener(path, "rt", encoding="ascii") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if identifier is not None:
                    if not chunks:
                        raise ValueError(f"{path}: FASTA entry {identifier!r} has no sequence")
                    yield identifier, "".join(chunks)
                identifier = line[1:].split(maxsplit=1)[0]
                if not identifier:
                    raise ValueError(f"{path}: empty FASTA identifier on line {line_number}")
                chunks = []
            elif identifier is None:
                raise ValueError(f"{path}: sequence before a FASTA header on line {line_number}")
            else:
                chunks.append(line)
    if identifier is None:
        raise ValueError(f"{path}: no FASTA entries")
    if not chunks:
        raise ValueError(f"{path}: FASTA entry {identifier!r} has no sequence")
    yield identifier, "".join(chunks)


def fasta_summary(path: Union[str, os.PathLike]) -> tuple[int, int]:
    """Return the number of FASTA records and residues, validating the file."""
    proteins = 0
    amino_acids = 0
    for _, sequence in iter_fasta_sequences(path):
        proteins += 1
        amino_acids += len(sequence)
    return proteins, amino_acids


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_path(path: Path) -> Path:
    return path.with_name(path.name + REFERENCE_PROTEOME_RECORD_SUFFIX)


def _read_cached(
    proteome: ReferenceProteome, path: Path
) -> Optional[DownloadedReferenceProteome]:
    record_path = _record_path(path)
    if not path.is_file() or not record_path.is_file():
        return None
    try:
        record = json.loads(record_path.read_text())
        if record["proteome"]["proteome_id"] != proteome.proteome_id:
            return None
        digest = _sha256(path)
        if digest != record["sha256"]:
            return None
        proteins, amino_acids = fasta_summary(path)
        if (proteins, amino_acids) != (record["proteins"], record["amino_acids"]):
            return None
        return DownloadedReferenceProteome(
            proteome=proteome,
            path=path,
            sha256=digest,
            proteins=proteins,
            amino_acids=amino_acids,
            source_url=record["source_url"],
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None


def fetch_reference_proteome(
    download_dir: Union[str, os.PathLike],
    host: Union[str, ReferenceProteome],
    *,
    force: bool = False,
    timeout: float = 60.0,
    show_progress: bool = False,
    url: Optional[str] = None,
) -> DownloadedReferenceProteome:
    """Fetch and verify a complete host reference proteome using datacache.

    A content summary and SHA-256 sidecar are written next to the compressed
    FASTA. A valid local cache is reused without a network request; corruption
    or a requested refresh causes datacache to fetch it again. ``url`` exists
    for mirrors and tests; callers normally rely on the pinned UniProt stream.
    """
    from datacache import fetch_file

    proteome = get_reference_proteome(host)
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    path = download_dir / reference_proteome_filename(proteome)
    cached = None if force else _read_cached(proteome, path)
    if cached is not None:
        return cached

    source_url = url or reference_proteome_url(proteome)
    path = Path(
        fetch_file(
            source_url,
            destination=path,
            raw=True,
            force=force or path.exists(),
            timeout=timeout,
            resume="auto",
            show_progress=show_progress,
            record_provenance=True,
        )
    )
    proteins, amino_acids = fasta_summary(path)
    downloaded = DownloadedReferenceProteome(
        proteome=proteome,
        path=path,
        sha256=_sha256(path),
        proteins=proteins,
        amino_acids=amino_acids,
        source_url=source_url,
    )
    record_path = _record_path(path)
    temporary = record_path.with_name(record_path.name + ".partial")
    try:
        temporary.write_text(
            json.dumps(
                {
                    "proteome": asdict(proteome),
                    "source_url": source_url,
                    "sha256": downloaded.sha256,
                    "proteins": proteins,
                    "amino_acids": amino_acids,
                    "downloaded": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                },
                indent=2,
                sort_keys=True,
            )
        )
        os.replace(temporary, record_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return downloaded
