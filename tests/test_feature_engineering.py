import pytest
import pandas as pd
import numpy as np
from yelp_predict.feature_engineering import CreateResidualCat


@pytest.fixture
def df_input():
    return pd.DataFrame(
        {
            "A": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "B": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            "C": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            "D": [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            "E": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            "F": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            "G": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            "H": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            "I": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            "J": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "K": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        }
    )


@pytest.mark.parametrize(
    "threshold", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)
def test_create_res_cat(df_input, threshold):

    test_transformer = CreateResidualCat(threshold)
    df_output = test_transformer.fit_transform(df_input)
    mean_series = df_output.mean()

    residual_cat_test = [1] * int(threshold * 10)
    residual_cat_test.append([0] * int((1 - threshold) * 10))
    residual_cat_test = pd.Series(residual_cat_test)

    for cat in mean_series.index:
        if cat != "residual_category":
            assert mean_series[cat] >= threshold
    assert df_output.shape == (10, 12 - threshold * 10)
    np.array_equal(
        df_output["residual_category"].to_numpy(), residual_cat_test.to_numpy()
    )
