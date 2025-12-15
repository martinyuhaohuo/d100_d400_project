from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from typing import Self, Optional


class CreateResidualCat(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """
    This transformer aggregates unfrequent category into one residual category
    """

    def __init__(self, threshold: float = 0.01) -> None:
        """
        Initialize the transformer

        Parameters:
        -----------
        threshold : float, optional
            the percentage threshold used to determine unfrequent category
        """

        self.threshold = threshold

    def set_output(self, *, transform: Optional[str] = None) -> Self:
        """
        Set output configuration (to be compatible with set_output to be pandas)

        Parameters
        ----------
        transform : None
            ignored
        """

        return self

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Self:
        """
        Fit the transformer
        The transformer learn the frequent categories from train set

        Parameters:
        -----------
        X : pd.DataFrame
            the train set
        """

        X_transformed = X.copy()
        pct_series = X_transformed.mean().astype(float)
        self.freq_cat_list_ = pct_series[
            pct_series >= self.threshold
        ].index.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the transformer.
        The transformer turns infrequent categories into a residual category

        Parameters:
        -----------
        X : pd.DataFrame
            the train set or test set
        """

        X_transformed = X.copy()
        unfreq_cat_list = list()
        for cat in X_transformed.columns:
            if cat not in self.freq_cat_list_:
                unfreq_cat_list.append(cat)
        dummies_sum = X_transformed[unfreq_cat_list].sum(axis=1)
        residual_cat = dummies_sum.clip(upper=1).astype(int)
        X_transformed["residual_category"] = residual_cat
        X_transformed = X_transformed.drop(columns=unfreq_cat_list)
        return X_transformed


class Winsorizer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """
    This transformer winsorizes numerical features by
    clipping values to the specified quantiles
    """

    def __init__(self, lower_quantile: float, upper_quantile: float) -> None:
        """
        Initialize the transformer

        Parameters:
        -----------
        lower_quantile : float
            the lower clipping threshold
        higher_quantile : float
            the upper clipping threshold
        """

        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def set_output(self, *, transform: Optional[str] = None) -> Self:
        """
        Set output configuration (to be compatible with set_output to be pandas)

        Parameters
        ----------
        transform : None
            ignored
        """

        return self

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Self:
        """
        Fit the transformer
        The transformer learn feature-wise clipping thresholds

        Parameters:
        -----------
        X : pd.DataFrame
            the train set
        """

        X_transformed = X.copy()
        X_array = np.asarray(X_transformed)
        self.lower_quantile_ = np.quantile(
            X_array, self.lower_quantile, axis=0
        )
        self.upper_quantile_ = np.quantile(
            X_array, self.upper_quantile, axis=0
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the transformer
        The transformer clips feature values to learned quantile thresholds

        Parameters:
        -----------
        X : pd.DataFrame
            the train set or test set
        """

        X_transformed = X.copy()
        X_transformed = X_transformed.clip(
            lower=self.lower_quantile_,
            upper=self.upper_quantile_,
        )
        return X_transformed
