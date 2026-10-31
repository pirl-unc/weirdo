import numpy as np
import pandas as pd

from scripts import evaluate_tcr_visible_external as external
from scripts import train_tcr_visible_response_adapter as adapter


def synthetic_frame():
    rows = []
    for cohort, (_, domain) in adapter.COHORTS.items():
        for index in range(16):
            outcome = index % 2
            rows.append(
                {
                    "cohort": cohort,
                    "outcome": outcome,
                    "peptide": ("AACDEFGY", "CCKLMNPW", "DDEFGHIK")[index % 3],
                    "hla": "HLA-A*02:01" if index % 3 else "HLA-B*07:02",
                    "mhcflurry_2_3_0_presentation_score": 0.2 + 0.6 * outcome,
                    "netmhcpan_ba_rank": 10.0 - 8.0 * outcome,
                    "domain": domain,
                }
            )
    return pd.DataFrame(rows)


def external_frame():
    rows = []
    for benchmark, mutation_class in (
        ("tesla_public_2022", "snv"),
        ("braun_rcc_2025", "snv"),
        ("braun_rcc_2025", "frameshift"),
    ):
        for index in range(24):
            outcome = index % 2
            rows.append(
                {
                    "benchmark": benchmark,
                    "mutation_class": mutation_class,
                    "group": f"{benchmark}-{mutation_class}-{index % 8}",
                    "outcome": outcome,
                    "peptide": "AACDEFGY" if index % 3 else "CCKLMNPW",
                    "hla": "HLA-A*02:01",
                    "mhcflurry_2_3_0_presentation_score": 0.2 + 0.5 * outcome + 0.01 * index,
                    "netmhcpan_ba_rank": 9.0 - 6.0 * outcome + 0.1 * (index % 5),
                }
            )
    return pd.DataFrame(rows)


def test_frozen_adapter_is_scored_on_each_endpoint_against_both_baselines():
    bundle, _ = adapter.fit_adapter(synthetic_frame(), "tumor")
    results = external.evaluate(external_frame(), bundle, draws=50)
    assert set(results["endpoint"]) == {
        "tesla_public_2022",
        "braun_rcc_2025_all",
        "braun_rcc_2025_frameshift",
    }
    assert set(results["baseline"]) == {"mhcflurry_presentation", "netmhcpan_ba_rank"}
    assert len(results) == 6
    assert np.isfinite(results[["adapter_auc", "baseline_auc", "delta_auc"]]).all().all()
    assert (results["delta_auc_low"] <= results["delta_auc_high"]).all()


def test_ba_rank_is_oriented_so_lower_rank_scores_higher():
    frame = pd.DataFrame({"netmhcpan_ba_rank": [1.0, 5.0]})
    assert external.orient(frame, "netmhcpan_ba_rank")[0] > external.orient(frame, "netmhcpan_ba_rank")[1]
