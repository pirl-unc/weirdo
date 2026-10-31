from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from scripts import evaluate_iedb_grouped_learning as grouped

ROOT = Path(__file__).resolve().parents[1]


def toy_frame(n_studies=6, per_study=24, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for study in range(n_studies):
        for i in range(per_study):
            outcome = i % 2
            rows.append({"pmid": str(study), "outcome": outcome, "core": 0.4 + 0.2 * outcome + 0.1 * rng.random()})
    return pd.DataFrame(rows)


def test_margin_is_none_for_plain_and_centred_for_residual():
    matrix = pd.DataFrame({"core": [0.25, 0.5, 0.75]})
    assert grouped.margin(matrix, 0.0) is None
    assert np.allclose(grouped.margin(matrix, 4.0), [-1.0, 0.0, 1.0])


def test_eligible_studies_need_size_and_both_classes():
    frame = pd.DataFrame({"pmid": ["a"] * 30 + ["b"] * 30, "outcome": [0, 1] * 15 + [1] * 30})
    assert grouped.eligible_studies(frame) == ["a"]


def test_residual_ranker_runs_in_its_own_process():
    # xgboost and torch load different OpenMP runtimes, so fit in a subprocess.
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from scripts import evaluate_iedb_grouped_learning as g\n"
        "import numpy as np, pandas as pd\n"
        "rng = np.random.default_rng(0)\n"
        "f = pd.DataFrame([{'pmid': str(s), 'outcome': i %% 2, 'core': 0.4 + 0.2 * (i %% 2) + 0.1 * rng.random()}"
        " for s in range(6) for i in range(24)]); x = f[['core']]\n"
        "s = g.fit_predict(f, x, x, g.CONFIGS['shallow'], seed=0, scale=4.0)\n"
        "gap = g.within_study_gaps(f, pd.DataFrame({'m': s}), f['core'].to_numpy(), ['0', '1'])\n"
        "assert len(s) == len(f) and gap['m'].abs().max() < 0.2\n"
    ) % str(ROOT)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr[-800:]
