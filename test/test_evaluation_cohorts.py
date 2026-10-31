import re
from pathlib import Path

from weirdo.evaluation_cohorts import EVALUATION_COHORTS, cohort_status

ROOT = Path(__file__).resolve().parents[1]


def test_registered_reports_and_builders_exist_in_repo():
    for cohort in EVALUATION_COHORTS.values():
        for report in cohort.reports:
            assert (ROOT / report).is_dir(), (cohort.name, report)
        for script in re.findall(r"scripts/[\w.]+\.py", cohort.builder or ""):
            assert (ROOT / script).is_file(), (cohort.name, script)


def test_roles_are_known_and_sources_nonempty():
    for cohort in EVALUATION_COHORTS.values():
        assert cohort.role in {"development", "validation", "sealed-confirmation", "auxiliary"}
        assert cohort.source and cohort.description


def test_status_reports_presence_and_checksums(tmp_path):
    (tmp_path / "data/improve-validation").mkdir(parents=True)
    target = tmp_path / "data/improve-validation/source.json"
    target.write_text("{}")
    rows = {r["name"]: r for r in cohort_status(tmp_path, with_checksums=True)}
    present = rows["improve"]["present"]
    assert len(present[str(target.relative_to(tmp_path))]) == 64
    assert rows["tesla"]["present"]["data/neoantigen-benchmarks/tesla/TESLA_neoepitopes.xlsx"] is False
