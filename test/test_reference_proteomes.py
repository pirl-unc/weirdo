"""Tests for complete host reference-proteome downloads."""

import gzip
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from weirdo.reference_proteomes import (
    DownloadedReferenceProteome,
    fasta_summary,
    fetch_reference_proteome,
    get_reference_proteome,
    iter_fasta_sequences,
    list_reference_proteome_manifests,
    load_reference_proteome_manifest,
    reference_proteome_from_uniprot,
    reference_proteome_filename,
    reference_proteome_manifest_path,
    reference_proteome_metadata_url,
    reference_proteome_url,
)


def write_fasta(path: Path, text: str) -> Path:
    path.write_bytes(gzip.compress(text.encode("ascii")))
    return path


def test_known_hosts_and_aliases_resolve_to_pinned_proteomes():
    assert get_reference_proteome("dog").proteome_id == "UP000805418"
    assert get_reference_proteome("Canis familiaris") == get_reference_proteome("dog")
    assert get_reference_proteome("rattus_norvegicus").key == "rat"
    assert get_reference_proteome("cattle").proteome_id == "UP000009136"
    assert get_reference_proteome("cow") == get_reference_proteome("cattle")
    assert get_reference_proteome("Bos_taurus") == get_reference_proteome("cattle")
    with pytest.raises(ValueError, match="Unknown host"):
        get_reference_proteome("platypus")


def test_packaged_reference_proteome_selection_is_available():
    path = reference_proteome_manifest_path()
    assert path.name == "reference-proteomes-v1.yaml"
    assert path.is_file()
    with pytest.raises(ValueError, match="Unknown reference-proteome manifest"):
        reference_proteome_manifest_path("not-a-manifest")


def test_frozen_manifest_has_selectable_mammal_and_viral_groups():
    assert list_reference_proteome_manifests() == ("reference-proteomes-v1",)
    manifest = load_reference_proteome_manifest()
    assert len(manifest.entries) == 544
    assert len(manifest.select(("mammals",))) == 32
    assert len(manifest.select(("viruses",))) == 512
    assert len(manifest.select(("cattle",))) == 1
    assert manifest.select(("cattle",))[0].proteome.proteome_id == "UP000009136"
    with pytest.raises(ValueError, match="Unknown group"):
        manifest.select(("not-a-group",))


def test_reference_proteome_url_requests_a_compressed_fasta_stream():
    parsed = urlparse(reference_proteome_url("dog"))
    query = parse_qs(parsed.query)
    assert parsed.scheme == "https"
    assert query == {
        "compressed": ["true"],
        "format": ["fasta"],
        "query": ["proteome:UP000805418"],
    }
    assert reference_proteome_filename("dog") == "dog-UP000805418.fasta.gz"


def test_arbitrary_uniprot_reference_proteome_resolves_from_catalog_metadata(tmp_path):
    metadata = {
        "id": "UP000009137",
        "modified": "2026-01-01",
        "proteomeType": "Reference proteome",
        "taxonomy": {"scientificName": "Bos taurus", "taxonId": 9913},
    }
    source = tmp_path / "metadata.json"
    source.write_text(json.dumps(metadata))
    resolved = reference_proteome_from_uniprot("UP000009137", url=source.as_uri())
    assert resolved.key == "bos_taurus"
    assert resolved.scientific_name == "Bos taurus"
    assert resolved.catalog_modified == "2026-01-01"
    assert resolved.proteome_id == "UP000009137"
    assert reference_proteome_metadata_url("up000009136").endswith("/UP000009136")

    metadata["proteomeType"] = "Redundant proteome"
    source.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="not a reference proteome"):
        reference_proteome_from_uniprot("UP000009136", url=source.as_uri())
    with pytest.raises(ValueError, match="Invalid UniProt proteome ID"):
        reference_proteome_from_uniprot("not-a-upid", url=source.as_uri())


def test_fasta_parser_accepts_gzip_and_validates_structure(tmp_path):
    source = write_fasta(tmp_path / "source.fasta.gz", ">A first\nACDE\nFG\n>B\nMNP\n")
    assert list(iter_fasta_sequences(source)) == [("A", "ACDEFG"), ("B", "MNP")]
    assert fasta_summary(source) == (2, 9)

    bad = write_fasta(tmp_path / "bad.fasta.gz", "ACDE\n")
    with pytest.raises(ValueError, match="before a FASTA header"):
        list(iter_fasta_sequences(bad))


def test_fetches_via_datacache_writes_summary_and_reuses_verified_cache(tmp_path):
    source = write_fasta(tmp_path / "source.fasta.gz", ">A\nACDE\n>B example\nMN\n")
    downloaded = fetch_reference_proteome(
        tmp_path / "cache", "dog", url=source.as_uri(), show_progress=False
    )
    assert isinstance(downloaded, DownloadedReferenceProteome)
    assert downloaded.proteome.key == "dog"
    assert (downloaded.proteins, downloaded.amino_acids) == (2, 6)
    assert downloaded.path.name == "dog-UP000805418.fasta.gz"
    assert downloaded.path.with_name(downloaded.path.name + ".json").is_file()

    source.unlink()
    cached = fetch_reference_proteome(tmp_path / "cache", "dog", url=source.as_uri())
    assert cached == downloaded
