import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from yelp_predict.feature_engineering import CreateResidualCat, Winsorizer
from typing import List


def feature_engineering_pipeline(
    dummies: List[str], categories: List[str], for_model: str
) -> ColumnTransformer:
    """
    This function builds a feature engineering ColumnTransformer for specified model

    Parameters
    ----------
    dummies : List[str]
        the list of columns as dummy variables
    categories : List[str]
        the list of columns as categorical variables
    for_model : str
        the model type, can be "GLM" or "LGBM"

    Returns
    -------
    ColumnTransformer
        the feature engineering pipeline for specified model
    """

    # define pre-setted column groups
    n_length = ["name_length"]
    dist = ["dist"]
    time_d = ["inweek_open_d", "weekend_open_d"]
    time_h = ["inweek_open_h", "weekend_open_h"]
    time_t = ["inweek_open_t", "weekend_open_t"]
    time_missing = ["missing_h"]

    # define the pipeline for different column groups
    dummies_transformer = Pipeline(
        steps=[("create_residual_category", CreateResidualCat(threshold=0.01))]
    )
    categorical_transformer = Pipeline(
        steps=[
            (
                "one_hot_encode",
                OneHotEncoder(drop="first", sparse_output=False),
            )
        ]
    )
    n_length_transformer = Pipeline(
        steps=[
            (
                "winsorizer",
                Winsorizer(lower_quantile=0.01, upper_quantile=0.99),
            )
        ]
    )
    if for_model == "GLM":
        dist_transformer = Pipeline(
            steps=[
                (
                    "winsorizer",
                    Winsorizer(lower_quantile=0.01, upper_quantile=0.99),
                ),
                (
                    "spline",
                    SplineTransformer(knots="quantile", include_bias=False),
                ),
            ]
        )
    else:
        dist_transformer = Pipeline(
            steps=[
                (
                    "winsorizer",
                    Winsorizer(lower_quantile=0.01, upper_quantile=0.99),
                )
            ]
        )
    time_d_transformer = Pipeline(
        [
            (
                "missing_impute_mode",
                SimpleImputer(missing_values=np.nan, strategy="most_frequent"),
            ),
            (
                "one_hot_encode",
                OneHotEncoder(drop="first", sparse_output=False),
            ),
        ]
    )
    time_h_transformer = Pipeline(
        [
            (
                "missing_impute_mean",
                SimpleImputer(missing_values=np.nan, strategy="mean"),
            )
        ]
    )
    if for_model == "GLM":
        time_t_transformer = Pipeline(
            [
                (
                    "missing_impute_mean",
                    SimpleImputer(missing_values=np.nan, strategy="mean"),
                ),
                (
                    "spline",
                    SplineTransformer(knots="quantile", include_bias=False),
                ),
            ]
        )
    else:
        time_t_transformer = Pipeline(
            [
                (
                    "missing_impute_mean",
                    SimpleImputer(missing_values=np.nan, strategy="mean"),
                )
            ]
        )

    feature_engineeerning = ColumnTransformer(
        transformers=[
            ("dummy_transform", dummies_transformer, dummies),
            ("categorical_transform", categorical_transformer, categories),
            ("n_length_transform", n_length_transformer, n_length),
            ("dist_transform", dist_transformer, dist),
            ("time_d_transform", time_d_transformer, time_d),
            ("time_h_transform", time_h_transformer, time_h),
            ("time_t_transform", time_t_transformer, time_t),
            ("keep_missing_h", "passthrough", time_missing),
        ],
        remainder="drop",
    )
    feature_engineeerning.set_output(transform="pandas")
    return feature_engineeerning
