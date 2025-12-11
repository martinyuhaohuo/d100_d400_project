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
