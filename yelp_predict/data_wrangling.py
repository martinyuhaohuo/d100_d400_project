# import polars as pl
import pandas as pd
import ast


def expand_nested(raw: pd.DataFrame, nested_col: str) -> pd.DataFrame:
    """
    This function expand the nested json objects within deisgnated column
    into new columns.

    Parameters:
    -----------
    raw : pd.DataFrame
        the raw pandas dataframe containing nested json object in one column
    nested_col: str
        the column that contains nested json object

    Returns:
    --------
    pd.DataFrame
        the data frame with nested json object expanded into new columns
    """

    # extract all the attribute names (keys) in the nested json string
    attr_key_list = []
    for _, row in raw.iterrows():
        attr_dict = row[nested_col]
        if attr_dict:
            for key, value in attr_dict.items():
                if "{" not in value:
                    if key not in attr_key_list:
                        attr_key_list.append(key)
                else:
                    sub_attr_dict = ast.literal_eval(value)
                    for sub_key, sub_value in sub_attr_dict.items():
                        if key + "_" + sub_key not in attr_key_list:
                            attr_key_list.append(key + "_" + sub_key)

    # add these attributes names as new columns to the raw dataframe
    # fill these new columns with with NA values
    expanded = raw.copy(deep=True)
    for col_name in attr_key_list:
        expanded[col_name] = pd.NA

    # replace NA values based on values extracted from the nested json string
    for index, row in expanded.iterrows():
        attr_dict = row[nested_col]
        if attr_dict:
            for key, value in attr_dict.items():
                if "{" not in value:
                    expanded.at[index, key] = value  # type: ignore[index]
                else:
                    sub_attr_dict = ast.literal_eval(value)
                    for sub_key, sub_value in sub_attr_dict.items():
                        comp_key = key + "_" + sub_key
                        expanded.at[index, comp_key] = sub_value  # type: ignore[index]

    return expanded


def expand_cats(
    raw: pd.DataFrame, cats_col: str, thres_pct: float
) -> pd.DataFrame:
    """
    This function expand the list of lables in a column into new columns
    Each column is a dummy variable, with 1 means having such label
    0 otherwise

    Parameters:
    -----------
    raw : pd.DataFrame
        the raw pandas dataframe
    cats_col: str
        the column that contains the list of labels
    thres_pct: float
        the percentage threshold of filtering out rare labels

    Returns:
    --------
    pd.DataFrame
        the data frame with list of labels converted to new dummies
    """

    # extract all unique labels in the designated column
    cat_dict: dict[str, int] = {}
    for _, row in raw.iterrows():
        cats = row[cats_col]
        if cats:
            for label in cats.split(", "):
                if label in cat_dict.keys():
                    cat_dict[label] += 1
                else:
                    cat_dict[label] = 1

    # only keep labels that appear in > thres_pct% obs
    total = raw.shape[0]
    cat_count = pd.Series(cat_dict, name="cat_count").sort_values(
        ascending=False
    )
    cat_count = cat_count[cat_count > int(total * thres_pct)]

    # add these labels as new columns to the raw dataframe
    # fill these new columns with 0
    expanded = raw.copy(deep=True)
    for col_name in cat_count.index:
        expanded[col_name] = 0

    # replace o in new columns with 1 if one obs has such label
    for index, row in expanded.iterrows():
        cats = row[cats_col]
        if cats:
            for label in cats.split(", "):
                if label in cat_count.index:
                    expanded.at[index, label] = 1  # type: ignore[index]

    return expanded


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
        the printed output
    """
    output = ""
    for col_name, dtype in dataframe.dtypes.items():
        output += (
            f"{col_name}"
            + " " * (35 - len(str(col_name)))
            + "|"
            + f"{dtype}"
            + "\n"
        )
    return output
