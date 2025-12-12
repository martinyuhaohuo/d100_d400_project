import pandas as pd
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
