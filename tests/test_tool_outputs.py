"""reproject writes through the same no-overwrite path as the other tools; PCA reports its explained variance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.clustering import reduce_dimensions
from tools.spatial import reproject


def test_reproject_twice_keeps_the_first_output(workspace):
    pd.DataFrame({"E": [500000.0, 501000.0], "N": [6000000.0, 6001000.0], "Au": [1, 2]}).to_csv(
        workspace / "pts.csv", index=False
    )
    first = reproject("pts.csv", "EPSG:4326", x_col="E", y_col="N", source_crs="EPSG:32750")
    second = reproject("pts.csv", "EPSG:4326", x_col="E", y_col="N", source_crs="EPSG:32750")
    assert first["output_path"].endswith("pts_EPSG-4326.csv")
    assert second["output_path"].endswith("pts_EPSG-4326-2.csv")
    out = pd.read_csv(first["output_path"])
    assert out["Au"].tolist() == [1, 2]
    assert out["E"].between(116, 118).all() and out["N"].between(-37, -35).all()  # zone 50 south, in degrees


def test_pca_returns_the_explained_variance_of_each_component(workspace):
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    df = pd.DataFrame({"a": a, "b": a + rng.normal(scale=0.1, size=200), "c": rng.normal(size=200)})
    df.loc[0, "c"] = None  # an incomplete row is left out, as before
    df.to_csv(workspace / "geo.csv", index=False)
    out = reduce_dimensions("geo.csv", ["a", "b", "c"], method="pca", n_components=2)
    ratios = out["explained_variance_ratio"]
    assert len(ratios) == 2 and ratios[0] > ratios[1] > 0 and sum(ratios) <= 1
    assert ratios[0] == pytest.approx(2 / 3, abs=0.05)  # a and b move together: one component holds ~2 of 3 units
