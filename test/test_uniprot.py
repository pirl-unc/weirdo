"""Tests for downloading SwissProt from UniProt and building the sequence table.

The tests serve a small fake UniProt release from a local directory: datacache treats
``file://`` URLs like HTTP ones, so the real download, size/MD5 checking and caching code runs
without network access.
"""

import gzip
import hashlib
import json
import logging

import pytest

from weirdo import uniprot
from weirdo.uniprot import (
    SWISSPROT_DIVISIONS,
    build_sequence_table,
    division_filename,
    fetch_swissprot_divisions,
    iter_swissprot_sequences,
    parse_release_metalink,
    read_sequence_table,
)

ENTRY = """\
ID   {name:<24}Reviewed;{length:>12} AA.
AC   P00001;
DT   01-JAN-2000, integrated into UniProtKB/Swiss-Prot.
DE   RecName: Full=Test protein {name};
OS   Testus organismus.
SQ   SEQUENCE{length:>6} AA;  12345 MW;  0123456789ABCDEF CRC64;
{sequence_lines}
//
"""


def sequence_lines(sequence):
    groups = [sequence[i:i + 10] for i in range(0, len(sequence), 10)]
    rows = [" ".join(groups[i:i + 6]) for i in range(0, len(groups), 6)]
    return "\n".join("     " + row for row in rows)


def dat_text(proteins):
    return "".join(
        ENTRY.format(name=name, length=len(sequence), sequence_lines=sequence_lines(sequence))
        for name, sequence in proteins
    )


PROTEINS = {
    "archaea": [("AAA1_METJA", "MKVLAAGIVGLLLAQPTEK")],
    "bacteria": [("RS19_ECOLI", "MPRSLKKGPFVDDHLLKKVEEA"), ("RL2_ECOLI", "MAVVKCKPTSPGRRHVVKVVN")],
    "human": [("1433B_HUMAN", "MTMDKSELVQKAKLAEQAERYDDMAAAMKAV")],
}


def make_release(directory, version="2099_01", proteins=None, divisions=SWISSPROT_DIVISIONS):
    """Write a fake UniProt directory (division files and RELEASE.metalink); return its URL."""
    directory.mkdir(parents=True, exist_ok=True)
    proteins = PROTEINS if proteins is None else proteins
    entries = []
    for division in divisions:
        data = gzip.compress(dat_text(proteins.get(division, [(f"X{division[:4].upper()}_TEST", "ACDEFGHIKL")])).encode())
        name = division_filename(division)
        (directory / name).write_bytes(data)
        entries.append(
            f'<file name="{name}"><size>{len(data)}</size><verification>'
            f'<hash type="md5">{hashlib.md5(data).hexdigest()}</hash></verification></file>'
        )
    (directory / "RELEASE.metalink").write_text(
        '<metalink xmlns="http://www.metalinker.org/" version="3.0">'
        f"<version>{version}</version><files>{''.join(entries)}</files></metalink>"
    )
    return directory.as_uri()


class TestMetalink:
    def test_version_sizes_and_md5(self):
        xml = (
            '<metalink xmlns="http://www.metalinker.org/" version="3.0"><version>2026_03</version><files>'
            '<file name="uniprot_sprot_human.dat.gz"><size>119657933</size><verification>'
            '<hash type="md5">ABCDEF0123456789abcdef0123456789</hash></verification></file>'
            '<file name="README"><size>4491</size></file></files></metalink>'
        ).encode()
        release = parse_release_metalink(xml)
        assert release.version == "2026_03"
        assert release.files["uniprot_sprot_human.dat.gz"].size == 119657933
        assert release.files["uniprot_sprot_human.dat.gz"].md5 == "abcdef0123456789abcdef0123456789"
        assert release.files["README"].md5 is None

    @pytest.mark.parametrize("bad", [
        b"not xml",
        b'<metalink xmlns="http://www.metalinker.org/"><files/></metalink>',  # no version
    ])
    def test_malformed_listings_are_rejected(self, bad):
        with pytest.raises(ValueError):
            parse_release_metalink(bad)

    def test_entity_declarations_are_refused(self):
        bomb = (b'<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol">]>'
                b'<metalink xmlns="http://www.metalinker.org/"><version>&lol;</version></metalink>')
        with pytest.raises(ValueError, match="DOCTYPE"):
            parse_release_metalink(bomb)


class TestParser:
    def test_reads_names_and_multiline_sequences(self, tmp_path):
        long_sequence = "ACDEFGHIKL" * 13 + "MNP"
        path = tmp_path / "x.dat"
        path.write_text(dat_text([("A_TEST", "MKV"), ("LONG_TEST", long_sequence)]))
        assert list(iter_swissprot_sequences(path)) == [("A_TEST", "MKV"), ("LONG_TEST", long_sequence)]

    def test_reads_gzip(self, tmp_path):
        path = tmp_path / "x.dat.gz"
        path.write_bytes(gzip.compress(dat_text([("A_TEST", "MKVLA")]).encode()))
        assert list(iter_swissprot_sequences(path)) == [("A_TEST", "MKVLA")]

    def test_a_wrong_length_is_an_error(self, tmp_path):
        text = dat_text([("A_TEST", "MKVLA")]).replace("5 AA;", "9 AA;")
        path = tmp_path / "x.dat"
        path.write_text(text)
        with pytest.raises(ValueError, match="states 9 residues but has 5"):
            list(iter_swissprot_sequences(path))

    def test_text_between_entries_is_ignored(self, tmp_path):
        path = tmp_path / "x.dat"
        path.write_text("CC   -----\n" + dat_text([("A_TEST", "MKV")]) + "\n")
        assert list(iter_swissprot_sequences(path)) == [("A_TEST", "MKV")]


class TestSequenceTable:
    def test_round_trip_in_division_order(self, tmp_path):
        paths = {}
        for division, proteins in PROTEINS.items():
            paths[division] = tmp_path / division_filename(division)
            paths[division].write_bytes(gzip.compress(dat_text(proteins).encode()))
        messages = []
        n = build_sequence_table(paths, tmp_path / "out" / "sequences.csv", progress=messages.append)
        assert n == 4 and len(messages) == 3
        frame = read_sequence_table(tmp_path / "out" / "sequences.csv")
        assert list(frame.columns) == ["identifier", "sequence", "category"]
        assert frame["category"].tolist() == ["archaea", "bacteria", "bacteria", "human"]
        assert frame["identifier"].tolist()[1] == "RS19_ECOLI"
        assert not list((tmp_path / "out").glob("*.partial"))

    def test_unknown_division_is_refused_before_anything_is_written(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown SwissProt divisions"):
            build_sequence_table({"martians": tmp_path / "x.dat.gz"}, tmp_path / "out.csv")
        assert not (tmp_path / "out.csv").exists()

    def test_a_bad_file_leaves_no_partial_output(self, tmp_path):
        bad = tmp_path / "bad.dat"
        bad.write_text(dat_text([("A_TEST", "MKV")]).replace("3 AA;", "4 AA;"))
        with pytest.raises(ValueError):
            build_sequence_table({"human": bad}, tmp_path / "out.csv")
        assert list(tmp_path.glob("out.csv*")) == []

    def test_reads_the_older_layout_with_an_index_column(self, tmp_path):
        path = tmp_path / "old.csv"
        path.write_text(",identifier,sequence,category\n0,A_TEST,MKV,human\n1,B_TEST,NA,plants\n")
        frame = read_sequence_table(path)
        assert frame["identifier"].tolist() == ["A_TEST", "B_TEST"]
        assert frame["sequence"].tolist() == ["MKV", "NA"]  # 'NA' is a sequence, not a missing value

    def test_missing_columns_are_reported(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("identifier,sequence\nA,B\n")
        with pytest.raises(ValueError, match="category"):
            read_sequence_table(path)


class TestFetch:
    def test_downloads_every_division_into_a_directory_per_release(self, tmp_path):
        url = make_release(tmp_path / "mirror")
        release = fetch_swissprot_divisions(tmp_path / "downloads", base_url=url)
        assert release.version == "2099_01"
        assert set(release.paths) == set(SWISSPROT_DIVISIONS)
        for division, path in release.paths.items():
            assert path == tmp_path / "downloads" / "release-2099_01" / division_filename(division)
            assert path.read_bytes() == (tmp_path / "mirror" / division_filename(division)).read_bytes()
        record = json.loads((tmp_path / "downloads" / "release-2099_01" / "release.json").read_text())
        assert record["version"] == "2099_01" and len(record["files"]) == 10

    def test_a_cached_file_is_not_downloaded_again(self, tmp_path):
        url = make_release(tmp_path / "mirror")
        fetch_swissprot_divisions(tmp_path / "downloads", base_url=url)
        # remove the source: only the cache can answer now, and the listing still works
        for division in SWISSPROT_DIVISIONS:
            (tmp_path / "mirror" / division_filename(division)).unlink()
        again = fetch_swissprot_divisions(tmp_path / "downloads", base_url=url)
        assert all(path.exists() for path in again.paths.values())

    def test_a_corrupted_cached_file_is_detected(self, tmp_path):
        url = make_release(tmp_path / "mirror")
        release = fetch_swissprot_divisions(tmp_path / "downloads", base_url=url)
        victim = release.paths["human"]
        data = bytearray(victim.read_bytes())
        data[-1] ^= 0xFF  # same size, different content
        victim.write_bytes(bytes(data))
        with pytest.raises(ValueError, match="MD5"):
            fetch_swissprot_divisions(tmp_path / "downloads", base_url=url)
        repaired = fetch_swissprot_divisions(tmp_path / "downloads", base_url=url, force=True)
        assert repaired.paths["human"].read_bytes() == (tmp_path / "mirror" / division_filename("human")).read_bytes()

    def test_a_truncated_download_is_rejected(self, tmp_path):
        url = make_release(tmp_path / "mirror")
        path = tmp_path / "mirror" / division_filename("viruses")
        path.write_bytes(path.read_bytes()[:-5])  # the listing still states the full size
        with pytest.raises(ValueError, match="size|bytes"):
            fetch_swissprot_divisions(tmp_path / "downloads", base_url=url, divisions=["viruses"])
        assert not (tmp_path / "downloads" / "release-2099_01" / division_filename("viruses")).exists()

    def test_subset_of_divisions(self, tmp_path):
        url = make_release(tmp_path / "mirror")
        release = fetch_swissprot_divisions(tmp_path / "downloads", base_url=url, divisions=["viruses", "human"])
        assert set(release.paths) == {"viruses", "human"}
        with pytest.raises(ValueError, match="Unknown SwissProt divisions"):
            fetch_swissprot_divisions(tmp_path / "downloads", base_url=url, divisions=["martians"])

    def test_offline_falls_back_to_the_newest_complete_cached_release(self, tmp_path, caplog):
        url = make_release(tmp_path / "mirror")
        fetch_swissprot_divisions(tmp_path / "downloads", base_url=url)
        with caplog.at_level(logging.WARNING, logger="weirdo.uniprot"):
            release = fetch_swissprot_divisions(
                tmp_path / "downloads", base_url=(tmp_path / "nowhere").as_uri()
            )
        assert release.version == "2099_01"
        assert "Could not reach UniProt" in caplog.text

    def test_offline_without_a_cache_is_an_error(self, tmp_path):
        with pytest.raises(ConnectionError, match="no complete release is cached"):
            fetch_swissprot_divisions(tmp_path / "downloads", base_url=(tmp_path / "nowhere").as_uri())

    def test_a_pinned_release_comes_from_the_cache_without_network(self, tmp_path):
        url = make_release(tmp_path / "mirror")
        fetch_swissprot_divisions(tmp_path / "downloads", base_url=url)
        release = fetch_swissprot_divisions(
            tmp_path / "downloads", release="2099_01", base_url=(tmp_path / "nowhere").as_uri()
        )
        assert release.version == "2099_01"

    def test_an_older_release_that_is_not_cached_cannot_be_fetched(self, tmp_path):
        url = make_release(tmp_path / "mirror")
        with pytest.raises(ValueError, match="current release|serves the division files of release 2099_01"):
            fetch_swissprot_divisions(tmp_path / "downloads", release="2023_02", base_url=url)

    def test_the_url_can_come_from_the_environment(self, tmp_path, monkeypatch):
        url = make_release(tmp_path / "mirror")
        monkeypatch.setenv(uniprot.BASE_URL_ENV, url + "/")
        assert uniprot.resolve_base_url() == url
        release = fetch_swissprot_divisions(tmp_path / "downloads", divisions=["human"])
        assert release.base_url == url

    def test_a_division_missing_from_the_listing_is_reported(self, tmp_path):
        url = make_release(tmp_path / "mirror", divisions=("human",))
        with pytest.raises(ValueError, match="does not list uniprot_sprot_viruses.dat.gz"):
            fetch_swissprot_divisions(tmp_path / "downloads", base_url=url, divisions=["viruses"])

    def test_the_default_location_is_uniprot(self):
        assert uniprot.UNIPROT_BASE_URL.startswith("https://ftp.uniprot.org/")
