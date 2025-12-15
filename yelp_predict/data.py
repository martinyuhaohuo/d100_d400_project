import pandas as pd
import hashlib
from pathlib import Path


def load_data(path: Path) -> pd.DataFrame:
    """
    This function load the dataset stored in json format into a pandas dataframe

    Parameters:
    -----------
    path : Path
        designating the path to the json file

    Returns:
    --------
    pd.DataFrame
        the data frame loaded from the json file
    """
    return pd.read_json(path, lines=True)


def hash_int(business_id: str) -> int:
    """
    This function converts a string business_id into a hash interger

    Parameters:
    -----------
    business_id : string
        business_id value

    Returns:
    --------
    interger
        the hash interger representation of the business_id
    """

    string_id = str(business_id)
    return int(hashlib.md5(string_id.encode()).hexdigest(), 16)


def create_sample_split(
    dataframe: pd.DataFrame, id_column: str, training_frac: float = 0.8
) -> pd.DataFrame:
    """
    This function creates sample split based on business_id column

    Parameters
    ----------
    dataframe : pd.DataFrame
        the cleaned dataframe
    id_column : str
        name of the business_id column
    training_frac : float, optional
        fraction to use for the training, by default 0.8

    Returns
    -------
    pd.DataFrame
        the dataframe with a new sample column containing train/test split
    """

    processed = dataframe.copy(deep=True)
    processed["hash"] = processed[id_column].apply(hash_int)
    processed["sample"] = processed["hash"].apply(
        lambda x: "test" if x % 100 >= training_frac * 100 else "train"
    )
    processed = processed.drop("hash", axis=1)
    return processed
