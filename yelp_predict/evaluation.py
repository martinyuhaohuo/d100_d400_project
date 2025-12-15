import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glum import BinomialDistribution
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
import dalex as dx
from pandas.api.types import is_string_dtype
from typing import Optional, List
from pathlib import Path


def compute_bias(pred_col: pd.Series, act_col: pd.Series) -> float:
    """
    This function computes prediction bias

    Parameters:
    -----------
    pred_col : pd.Series
        the column of predicted values
    act_col : pd.Series
        the column of actual observed values

    Returns:
    --------
    float
        the bias
    """

    pred_mean_claim = np.mean(pred_col)
    actual_mean_claim = np.mean(act_col)
    bias = float(pred_mean_claim - actual_mean_claim)
    return bias


def compute_deviance(pred_col: pd.Series, act_col: pd.Series) -> float:
    """
    This function computes binomial deviance
    between predictions and actual values

    Parameters:
    -----------
    pred_col : pd.Series
        the column of predicted values
    act_col : pd.Series
        the column of actual observed values

    Returns:
    --------
    float
        the deviance
    """

    BinomialDist = BinomialDistribution()
    deviance = float(BinomialDist.deviance(act_col, pred_col))
    return deviance


def compute_metrics(
    pred_col: pd.Series, act_col: pd.Series, model: str
) -> pd.DataFrame:
    """
    This function computes evaluation metrics for a given model

    Parameters:
    -----------
    pred_col : pd.Series
        the column of predicted values
    act_col : pd.Series
        the column of actual observed values
    model : str
        the model name

    Returns:
    --------
    pd.DataFrame
        the dataframe containing mae, rmse, deviance, and bias
    """

    mae = mean_absolute_error(act_col, pred_col)
    rmse = mean_squared_error(act_col, pred_col)
    deviance = compute_deviance(pred_col, act_col)
    bias = compute_bias(pred_col, act_col)
    df = pd.DataFrame()
    df["model"] = [model]
    df["mae"] = [mae]
    df["rmse"] = [rmse]
    df["deviance"] = [deviance]
    df["bias"] = [bias]
    return df


def pred_vs_actual(
    pred_col: str,
    actual_col: str,
    dataframe: pd.DataFrame,
    model: str,
    save_path: Optional[Path] = None,
) -> None:
    """
    This function plots predicted values against actual values

    Parameters:
    -----------
    pred_col : str
        the column name of predicted values
    act_col : str
        the column name of actual observed values
    dataframe : pd.DataFrame
        the dataframe with prediction and actual values (test)
    model : str
        the model name
    save_path : Path, optional
        the path designates where the plot will be saved

    Returns:
    --------
    None
        the is a plot function
    """

    plt.figure(figsize=(7, 7))
    x = dataframe[actual_col]
    y = dataframe[pred_col]
    plt.scatter(
        x,
        y,
        facecolors="none",
        edgecolors="grey",
        alpha=0.5,
    )

    dig_x = np.linspace(1, 5, 200)
    dig_y = np.linspace(1, 5, 200)

    plt.plot(
        dig_x,
        dig_y,
        color="red",
        linewidth=1,
        linestyle="--",
        label="Diagonal Line",
    )

    plt.xlabel("Actual (Stars)")
    plt.ylabel("Prediction (Stars)")
    plt.title(f"Actual v.s. Prediction By {model}")
    plt.grid(alpha=0.3)
    plt.legend()

    if save_path is not None:
        filename = f"actual_predict_{model}.png"
        plt.savefig(save_path / filename, dpi=300, bbox_inches="tight")

    plt.show()


def permutation_importance_top(
    cv_pipeline: Pipeline, df_x: pd.DataFrame, y: pd.Series
) -> pd.DataFrame:
    """
    This function computes permutation importance of features

    Parameters:
    -----------
    cv_pipeline : Pipeline
        the fitted CV pipeline
    df_x : pd.DataFrame
        the training set or test set
    y : pd.Series
        the training set target or testing set target

    Returns:
    --------
    pd.DataFrame
        features with permutation importance, sorted descending
    """

    r = permutation_importance(
        cv_pipeline.best_estimator_,
        df_x,
        y,
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=0,
    )
    importance_series = r.importances_mean
    importance_frame = pd.DataFrame(
        {"col": df_x.columns, "importance": importance_series}
    )
    importance_frame = importance_frame.sort_values(
        by="importance", ascending=False
    )
    return importance_frame


def plot_PDP(
    features: List[str],
    model_pipeline: Pipeline,
    model: str,
    df_x: pd.DataFrame,
    y: pd.Series,
    save_path: Path,
) -> None:
    """
    This function plots partial dependece plots for selected features

    Parameters:
    -----------
    features : List[str]
        the list of feature names for plotting
    model_pipeline : Pipeline
        the fitted best model pipeline
    model : str
        the model name
    df_x : pd.DataFrame
        the train set or test set
    y : pd.Series
        the target of train or test set
    save_path : Path
        the path where plots will be saved

    Returns
    -------
    None
        the function save plots to path
    """

    exp = dx.Explainer(model_pipeline, df_x, y)
    for feature in features:
        if is_string_dtype(df_x[feature]):
            plot = exp.model_profile(
                variables=feature, variable_type="categorical"
            ).plot(show=False)
        else:
            plot = exp.model_profile(variables=feature).plot(show=False)
        complete_path = save_path / f"PDP_{model}_{feature}.html"
        plot.write_html(complete_path)


def compare_metrics_by_rating_level(
    df_test: pd.DataFrame, lower_quantile: float, upper_quantile: float
) -> None:
    """
    This function prints accuracy metrics at different quantile of avg_rating

    Parameters:
    -----------
    df_test : pd.DataFrame
        the train set or test set
    lower_quantile : float
        the lower quantile of avg_rating
    upper_quantile : float
        the higher quantile of avg_rating

    Returns
    -------
    None
        the function prints result
    """

    lower_quantile_value = df_test["avg_rating"].quantile(lower_quantile)
    upper_quantile_value = df_test["avg_rating"].quantile(upper_quantile)
    df_temp = df_test[
        (df_test["avg_rating"] >= lower_quantile_value)
        & (df_test["avg_rating"] <= upper_quantile_value)
    ]
    print(
        "Metrics for avg_rating lower_q:"
        + f"{lower_quantile}, higher_q: {upper_quantile}"
    )
    print(
        compute_metrics(
            df_temp["predict_GLM_revert"], df_temp["avg_rating"], model="GLM"
        )
    )
    print(
        compute_metrics(
            df_temp["predict_LGBM_revert"], df_temp["avg_rating"], model="LGBM"
        )
    )


def compare_metrics_by_cat(df_test: pd.DataFrame, category: str) -> None:
    """
    This function prints accuracy metrics w.r.t different categories

    Parameters:
    -----------
    df_test : pd.DataFrame
        the train set or test set
    category : str
        the designated category name

    Returns
    -------
    None
        the function prints result
    """

    df_temp = df_test[df_test[category] == 1]
    print(f"Metrics for category: {category}")
    print(
        compute_metrics(
            df_temp["predict_GLM_revert"], df_temp["avg_rating"], model="GLM"
        )
    )
    print(
        compute_metrics(
            df_temp["predict_LGBM_revert"], df_temp["avg_rating"], model="LGBM"
        )
    )
