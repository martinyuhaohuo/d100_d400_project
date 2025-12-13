import pandas as pd
import polars as pl
import numpy as np
from geopy.distance import geodesic
import ast
from typing import Tuple, List
from pathlib import Path


def col_contain_str(
    dataframe: pd.DataFrame, col_name: str, contain: str
) -> pd.DataFrame:
    """
    This function filter observations based on a given column
    It checks whether the value in the given column contains deiganted substring

    Parameters:
    -----------
    dataframe : pd.DataFrame
        the raw pandas dataframe
    col_name: str
        the column to filter
    contain: str
        substring to search within column values

    Returns:
    --------
    pd.DataFrame
        the dataframe with filtered observations
    """

    return dataframe[dataframe[col_name].str.contains(contain, na=False)]


def keep_city(raw: pd.DataFrame, cities: list[str]) -> pd.DataFrame:
    """
    This function filter out observations that are not in the city list

    Parameters:
    -----------
    raw : pd.DataFrame
        the raw pandas dataframe
    cities: list of string
        the list of cities that we want to keep

    Returns:
    --------
    pd.DataFrame
        the dataframe with filtered observations
    """

    cities = ["Philadelphia", "Tampa", "Indianapolis", "Nashville", "Tucson"]
    raw = raw[raw["city"].isin(cities)]
    raw.reset_index(drop=True, inplace=True)
    return raw


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
    new_col_frame = pd.DataFrame(
        0, index=expanded.index, columns=cat_count.index
    )
    expanded = pd.concat([expanded, new_col_frame], axis=1)

    # replace o in new columns with 1 if one obs has such label
    for index, row in expanded.iterrows():
        cats = row[cats_col]
        if cats:
            for label in cats.split(", "):
                if label in cat_count.index:
                    expanded.at[index, label] = 1  # type: ignore[index]

    return expanded


def extract_open_close_t(raw: pd.DataFrame) -> pd.DataFrame:
    """
    This function extracts open and closed time from time interval in string

    Parameters:
    -----------
    raw : pd.DataFrame
        the raw pandas dataframe

    Returns:
    --------
    pd.DataFrame
        the dataframe with opening time and closed time
        for each weekday seperately as new columns
    """

    weekdays = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    new_col_name = list()
    for weekday in weekdays:
        new_col_name.append(weekday + "Ot")
        new_col_name.append(weekday + "Ct")

    expanded = raw.copy(deep=True)
    new_col_frame = pd.DataFrame(  # type: ignore[call-overload]
        pd.NA, index=expanded.index, columns=new_col_name
    )
    expanded = pd.concat([expanded, new_col_frame], axis=1)

    for index, row in expanded.iterrows():
        for weekday in weekdays:
            if not pd.isna(row[weekday]):
                time_interval = row[weekday].split("-")
                expanded.at[index, weekday + "Ot"] = time_interval[0]  # type: ignore[index]
                expanded.at[index, weekday + "Ct"] = time_interval[1]  # type: ignore[index]

    expanded[new_col_name] = expanded[new_col_name].apply(
        lambda s: (pd.to_datetime(s, format="%H:%M").dt.time)  # type: ignore[attr-defined]
    )

    return expanded


def split_subframes(
    expanded: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    This function splits the expanded dataframe into four subframes
    based on the type of features descirbed

    Parameters:
    -----------
    expanded : pd.DataFrame
        the expanded pandas dataframe

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        the tuple of four divided dataframes
    """

    basic = expanded.iloc[:, 0:14]
    attributes = expanded.iloc[:, 14:92]
    attributes.insert(0, "business_id", basic["business_id"].values)  # type: ignore[arg-type]
    categories = expanded.iloc[:, 99:653]
    categories.insert(0, "business_id", basic["business_id"].values)  # type: ignore[arg-type]
    times = expanded.iloc[:, 653:]
    times.insert(0, "business_id", basic["business_id"].values)  # type: ignore[arg-type]

    return basic, attributes, categories, times


def unify_NA(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    This function replaces string None with pd.NA for each cell value

    Parameters:
    -----------
    dataframe : pd.DataFrame
        the raw pandas dataframe

    Returns:
    --------
    pd.DataFrame
        the dataframe with replaced NA values
    """

    processed = dataframe.copy(deep=True)
    processed = processed.replace("None", pd.NA)
    return processed


def convert_to_string(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    This function converts all columns of a dataframe to string dtype

    Parameters:
    -----------
    dataframe : pd.DataFrame
        the raw pandas dataframe

    Returns:
    --------
    pd.DataFrame
        the dataframe with converted dtype
    """

    processed = dataframe.copy(deep=True)
    for col_name in processed.columns:
        processed[col_name] = processed[col_name].astype("string")
    return processed


def correct_string(
    dataframe: pd.DataFrame, exclude: List[str] = ["business_id"]
) -> pd.DataFrame:
    """
    This function removes leading 'u' and ' in string column values

    Parameters:
    -----------
    dataframe : pd.DataFrame
        the raw pandas dataframe
    exclude: List[str]
        the list of columns to exclude from the operation

    Returns:
    --------
    pd.DataFrame
        the dataframe with replaced string column values
    """

    processed = dataframe.copy(deep=True)
    for col_name in processed.columns:
        if col_name not in exclude:
            processed[col_name] = (
                processed[col_name].str.replace("u'", "").str.replace("'", "")
            )

    return processed


def detect_open(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    This function returns a dataframe with new columns
    indicating whether the restaurant is open on a specific weekday

    Parameters:
    -----------
    dataframe : pd.DataFrame
        the times subframe

    Returns:
    --------
    pd.DataFrame
        the dataframe with new open-weekday indicators
    """

    weekdays = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    expanded = dataframe.copy(deep=True)
    new_col_frame = pd.DataFrame(  # type: ignore[call-overload]
        pd.NA, index=expanded.index, columns=weekdays
    )
    expanded = pd.concat([expanded, new_col_frame], axis=1)

    for index, row in expanded.iterrows():
        no_hour_flag = False

        for weekday in weekdays:
            if not pd.isna(row[weekday + "Ot"]):
                no_hour_flag = True
                break

        for weekday in weekdays:
            if not pd.isna(row[weekday + "Ot"]):
                expanded.at[index, weekday] = 1  # type: ignore[index]
            elif no_hour_flag:
                expanded.at[index, weekday] = 0  # type: ignore[index]

    return expanded


def name_length(dataframe: pd.DataFrame, name_col: str) -> pd.DataFrame:
    """
    This function adds a new column recording length of restaurant name

    Parameters
    ----------
    dataframe : pd.DataFrame
        the input DataFrame containing the name column
    name_col : str
        the name of the name column

    Returns
    -------
    pd.DataFrame
        the dataframe with a new column recording name length
    """

    processed = dataframe.copy(deep=True)
    processed["name_length"] = processed[name_col].apply(len)
    return processed


def dis_city_center(basic: pd.DataFrame) -> pd.DataFrame:
    """
    This function computes distance from each restaurant to the city center

    Parameters
    ----------
    dataframe : pd.DataFrame
        the input dataframe containing the "city",
        "latitude", and "longitude" columns

    Returns
    -------
    pd.DataFrame
        the dataframe with a new column recording distance to city center
    """

    city_center_dict = {
        "Philadelphia": [39.9526283493744, -75.16375869151712],
        "Tampa": [27.95164234128313, -82.45873880021296],
        "Indianapolis": [39.76850993111294, -86.15795751895246],
        "Nashville": [36.16261434723913, -86.78144809772807],
        "Tucson": [32.2539491542206, -110.9741304965398],
    }

    processed = basic.copy(deep=True)
    processed["dist"] = [0.0] * processed.shape[0]

    for index, row in processed.iterrows():
        city = row["city"]
        center_latitude = city_center_dict[city][0]
        center_longitude = city_center_dict[city][1]
        latitude = row["latitude"]
        longitude = row["longitude"]
        dist = geodesic(
            (latitude, longitude), (center_latitude, center_longitude)
        ).km
        processed.at[index, "dist"] = dist  # type: ignore[index]

    return processed


def cal_avg_rating(basic: pd.DataFrame, review_path: Path) -> pd.DataFrame:
    """
    This function claculate average rating for each restaurant based on historical reviews

    Parameters
    ----------
    basic : pd.DataFrame
        the dataframe containing business_id
    review_path : Path
        the path to review dataset

    Returns
    -------
    pd.DataFrame
        the dataframe with a new column recording average rating
    """

    processed = basic.copy(deep=True)
    review_pl = (
        pl.scan_ndjson(review_path)
        .select(["business_id", "stars"])
        .group_by("business_id")
        .agg(avg_rating=pl.col("stars").mean())
    )

    review = review_pl.collect()
    review_pd = review.to_pandas()
    processed = pd.merge(processed, review_pd, on="business_id", how="inner")
    return processed


def ave_open_days(times: pd.DataFrame) -> pd.DataFrame:
    """
    This function calculates the number of open weekdays and weekend days for each restaurant

    Parameters
    ----------
    times : pd.DataFrame
        the dataframe containing open weekday indicators
        (0: not open, 1: open, NA: info not provided)

    Returns
    ----------
    pd.DataFrame
        the processed dataframe with added open weekdays and weekend days
    """
    processed = times.copy(deep=True)
    inweek = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    weekend = ["Saturday", "Sunday"]

    for index, row in processed.iterrows():
        inweek_days = 0
        weekend_days = 0
        for day in inweek:
            inweek_days += row[day]
        for day in weekend:
            weekend_days += row[day]
        processed.at[index, "inweek_open_d"] = inweek_days  # type: ignore[index]
        processed.at[index, "weekend_open_d"] = weekend_days  # type: ignore[index]

    processed["inweek_open_d"] = processed["inweek_open_d"].replace(
        pd.NA, np.nan
    )
    processed["weekend_open_d"] = processed["weekend_open_d"].replace(
        pd.NA, np.nan
    )

    return processed


def ave_open_hours(times: pd.DataFrame) -> pd.DataFrame:
    """
    This function calculates the average of open hours
    during weekdays and weekend for each restaurant

    Parameters
    ----------
    times : pd.DataFrame
        the dataframe containing open and close time

    Returns
    ----------
    pd.DataFrame
        the processed dataframe with added average openning hours
    """

    processed = times.copy(deep=True)
    inweek = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    weekend = ["Saturday", "Sunday"]

    for index, row in processed.iterrows():
        inweek_hours = 0
        weekend_hours = 0
        inweek_days = 0
        weekend_days = 0

        for day in inweek:
            inweek_days += row[day]
            if not pd.isna(row[day + "Ct"]) and not pd.isna(row[day + "Ot"]):
                if row[day + "Ct"].hour > row[day + "Ot"].hour:
                    inweek_hours += (
                        row[day + "Ct"].hour + row[day + "Ct"].minute / 60
                    ) - (row[day + "Ot"].hour + row[day + "Ot"].minute / 60)
                else:
                    inweek_hours += (
                        row[day + "Ct"].hour + 24 + row[day + "Ct"].minute / 60
                    ) - (row[day + "Ot"].hour + row[day + "Ot"].minute / 60)

        for day in weekend:
            weekend_days += row[day]
            if not pd.isna(row[day + "Ct"]) and not pd.isna(row[day + "Ot"]):
                if row[day + "Ct"].hour > row[day + "Ot"].hour:
                    weekend_hours += (
                        row[day + "Ct"].hour + row[day + "Ct"].minute / 60
                    ) - (row[day + "Ot"].hour + row[day + "Ot"].minute / 60)
                else:
                    weekend_hours += (
                        row[day + "Ct"].hour + 24 + row[day + "Ct"].minute / 60
                    ) - (row[day + "Ot"].hour + row[day + "Ot"].minute / 60)

        if pd.isna(inweek_days):
            processed.at[index, "inweek_open_h"] = np.nan
        elif inweek_days > 0:
            avg_h = inweek_hours / inweek_days
            if avg_h <= 24:
                processed.at[index, "inweek_open_h"] = avg_h  # type: ignore[index]
            else:
                processed.at[index, "inweek_open_h"] = 24  # type: ignore[index]
        else:
            processed.at[index, "inweek_open_h"] = np.nan  # type: ignore[index]

        if pd.isna(weekend_days):
            processed.at[index, "weekend_open_h"] = np.nan
        elif weekend_days > 0:
            avg_h = weekend_hours / weekend_days
            if avg_h <= 24:
                processed.at[index, "weekend_open_h"] = avg_h  # type: ignore[index]
            else:
                processed.at[index, "weekend_open_h"] = 24  # type: ignore[index]
        else:
            processed.at[index, "weekend_open_h"] = np.nan  # type: ignore[index]

    processed["inweek_open_h"] = processed["inweek_open_h"].replace(
        pd.NA, np.nan
    )
    processed["weekend_open_h"] = processed["weekend_open_h"].replace(
        pd.NA, np.nan
    )

    return processed


def ave_open_time(times: pd.DataFrame) -> pd.DataFrame:
    """
    This function calculates the average of openning time
    during weekdays and weekend for each restaurant
    (in hour, e.g. open in avg at 17.5 means open at 17:30 in average)

    Parameters
    ----------
    times : pd.DataFrame
        the dataframe containing open and close time

    Returns
    ----------
    pd.DataFrame
        the processed dataframe with added average openning time
    """

    processed = times.copy(deep=True)
    inweek = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    weekend = ["Saturday", "Sunday"]

    for index, row in processed.iterrows():
        inweek_open_t = 0
        weekend_open_t = 0
        inweek_days = 0
        weekend_days = 0

        for day in inweek:
            inweek_days += row[day]
            if not pd.isna(row[day + "Ot"]):
                inweek_open_t += (
                    row[day + "Ot"].hour + row[day + "Ot"].minute / 60
                )

        for day in weekend:
            weekend_days += row[day]
            if not pd.isna(row[day + "Ot"]):
                weekend_open_t += (
                    row[day + "Ot"].hour + row[day + "Ot"].minute / 60
                )

        if pd.isna(inweek_days):
            processed.at[index, "inweek_open_t"] = np.nan
        elif inweek_days > 0:
            avg_t = inweek_open_t / inweek_days
            processed.at[index, "inweek_open_t"] = avg_t  # type: ignore[index]
        else:
            processed.at[index, "inweek_open_t"] = np.nan  # type: ignore[index]

        if pd.isna(weekend_days):
            processed.at[index, "weekend_open_t"] = np.nan
        elif weekend_days > 0:
            avg_h = weekend_open_t / weekend_days
            processed.at[index, "weekend_open_t"] = avg_h  # type: ignore[index]
        else:
            processed.at[index, "weekend_open_t"] = np.nan  # type: ignore[index]

    processed["inweek_open_t"] = processed["inweek_open_t"].replace(
        pd.NA, np.nan
    )
    processed["weekend_open_t"] = processed["weekend_open_t"].replace(
        pd.NA, np.nan
    )

    return processed
