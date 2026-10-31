import pandas as pd

from scripts import score_braun_mutation_aware_candidates as scoring


def test_score_batch_rejects_missing_mutation_aware_flag():
    frame = pd.DataFrame({"peptide": ["SYFPEITHI"], "hla": ["HLA-A*02:01"]})
    try:
        scoring.score_batch(frame, 10, 0)
    except ValueError as error:
        assert "reference_absent_candidate" in str(error)
    else:
        raise AssertionError("Missing candidate flag must be rejected")
