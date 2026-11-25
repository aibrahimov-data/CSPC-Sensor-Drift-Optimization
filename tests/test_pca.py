import numpy as np
import pandas as pd

from src.knn import FEATURES, TARGET
from src.pca_analysis import fit_pca, project


def make_demo(n_rows, seed, batch_id):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n_rows, 128)), columns=FEATURES)
    df[TARGET] = np.tile([1, 2], n_rows // 2)
    df["batch_id"] = batch_id
    return df


def test_scaler_and_pca_are_fitted_on_source_only():
    source = make_demo(80, seed=1, batch_id=3)
    future = make_demo(40, seed=2, batch_id=7)

    scaler, pca = fit_pca(source)
    project(future, scaler, pca)

    # Projecting future data must not change what was fitted.
    assert scaler.n_samples_seen_ == len(source)
    assert pca.n_samples_ == len(source)


def test_project_keeps_rows_and_labels():
    source = make_demo(80, seed=1, batch_id=3)
    future = make_demo(40, seed=2, batch_id=7)

    scaler, pca = fit_pca(source)
    projected = project(future, scaler, pca)

    assert len(projected) == len(future)
    assert list(projected.columns) == ["PC1", "PC2", TARGET, "batch_id"]
    assert (projected["batch_id"] == 7).all()