import pandas as pd
from typing import List


def count_bycat(dataframe: pd.DataFrame, cat_column: str) -> pd.DataFrame:
    """
    This function generates counts of rows by type
    designtaed by a catgoriucal variable

    Parameters:
    -----------
    dataframe : pd.DataFrame
        the raw dataframe
    cat_column: str
        the column name of the catgorical variable

    Returns:
    --------
    pd.DataFrame
        a pandas dataframe with index as type name, value as counts by type
    """
    counts = (
        dataframe.groupby(by=cat_column)
        .size()
        .reset_index(name="counts")
        .sort_values(by="counts", ascending=False)
    )
    return counts


def col_dtype(dataframe: pd.DataFrame) -> str:
    """
    This function prints name and datatype of each column of a dataframe

    Parameters:
    -----------
    dataframe : pd.DataFrame
        the dataframe with column names and dtypes need to be printed

    Returns:
    --------
    str
        the formulated output
    """
    output = ""
    i = 0
    for col_name, dtype in dataframe.dtypes.items():
        output += (
            str(i)
            + " " * (5 - len(str(i)))
            + f"{col_name}"
            + " " * (35 - len(str(col_name)))
            + "|"
            + f"{dtype}"
            + "\n"
        )
        i += 1
    return output


def report_uniques(
    dataframe: pd.DataFrame, exclude_col: List[str] = ["None"]
) -> str:
    """
    Summary unique values per column, exlcuding the columns deisgnated

    Parameters
    ----------
    dataframe : pd.DataFrame
        the input dataframe
    exclude_col : List[str], optional
        columns that will be skipped

    Returns
    -------
    str
        the formulated output
    """

    output = ""
    i = 0
    for col_name, dtype in dataframe.dtypes.items():

        if col_name not in exclude_col:
            uniques = dataframe[col_name].unique().tolist()
            uniques_len = len(uniques)
            if uniques_len > 100:
                uniques = ["omitted as n>100"]
        else:
            uniques = "not applicable"
            uniques_len = pd.NA  # type: ignore[assignment]

        Uni_counts = f"Uni_counts: {uniques_len}"

        output += (
            str(i)
            + " " * (5 - len(str(i)))
            + f"{col_name}"
            + " " * (35 - len(str(col_name)))
            + "|"
            + f"{dtype}"
            + " " * (30 - len(str(dtype)))
            + "|"
            + Uni_counts
            + " " * (30 - len(Uni_counts))
            + "|"
            + f"Uni_value: {uniques}"
            + "\n"
        )
        i += 1

    return output


def report_missings(dataframe: pd.DataFrame) -> str:
    """
    Summary missing values per column

    Parameters
    ----------
    dataframe : pd.DataFrame
        the input dataframe

    Returns
    -------
    str
        the formulated output
    """

    output = ""
    i = 0
    for col_name, dtype in dataframe.dtypes.items():

        missing_count = dataframe[col_name].isna().sum()
        missing_pct = missing_count / dataframe.shape[0]
        missing_count_str = f"Mis_count: {missing_count}"
        missing_pct_str = f"Mis_pct: {missing_pct}"

        output += (
            str(i)
            + " " * (5 - len(str(i)))
            + f"{col_name}"
            + " " * (35 - len(str(col_name)))
            + "|"
            + f"{dtype}"
            + " " * (30 - len(str(dtype)))
            + "|"
            + missing_count_str
            + " " * (30 - len(missing_count_str))
            + "|"
            + f"Uni_value: {missing_pct_str}"
            + "\n"
        )
        i += 1

    return output


def get_missing_column(
    attributes: pd.DataFrame, threshold: float
) -> List[str]:
    """
    This function returns a list of columns with missing value percentage > threshold

    Parameters
    ----------
    attributes : pd.DataFrame
        the input dataframe
    threshold : float
        the percentage threshold to drop columns with missing value

    Returns
    -------
    List[str]
        the list of column names with missing percentage > threshold
    """

    missing_cols = []
    for col_name in attributes.columns:

        missing_count = attributes[col_name].isna().sum()
        missing_pct = missing_count / attributes.shape[0]
        if missing_pct > threshold:
            missing_cols.append(col_name)

    return missing_cols
