"""Check an installed wheel's bundled artifact without source or network access."""

import json
from pathlib import Path
import socket

import numpy as np

import weirdo


def main():
    """Fail if installed-wheel inference requires a download or changes outputs."""
    checkout = Path(__file__).resolve().parents[1]
    if Path(weirdo.__file__).resolve().is_relative_to(checkout):
        raise RuntimeError("Run with the wheel installed in a separate environment, outside the checkout")
    def offline(*args, **kwargs):
        raise AssertionError("Bundled-model inference must be offline")
    socket.create_connection = offline
    socket.socket.connect = offline
    model = weirdo.load_pretrained_model()
    manifest = model.manifest
    peptide = manifest["smoke"]["peptide"]
    frame = model.predict_categories([peptide, peptide.lower(), peptide], batch_size=1)
    np.testing.assert_allclose(
        frame[manifest["categories"]].to_numpy(),
        np.tile(manifest["smoke"]["category_values"], (3, 1)),
        atol=manifest["smoke"]["atol"], rtol=0,
    )
    np.testing.assert_allclose(model.score([peptide]), 1 - frame.human.iloc[:1])
    assert model.score([]).shape == (0,)
    print(json.dumps({"version": weirdo.__version__, "prediction_identity": model.prediction_identity}))


if __name__ == "__main__":
    main()
