import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from yelp_predict.data_loading import expand_nested


@pytest.fixture
def nested_frame():
    return pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [5, 4, 3, 2, 1],
            "C": [
                {
                    "A1": "3",
                    "A2": "1",
                    "A3": "5",
                    "AB": "{'C1': '3', 'C2': '1', 'C3': '5'}",
                },
                {
                    "A1": "3",
                    "A3": "5",
                    "AB": "{'C1': '3', 'C2': '1', 'C3': '5'}",
                },
                {"A1": "3", "A3": "5"},
                None,
                {"A1": "3", "A3": "5", "AB": "{'C1': '3', 'C3': '5'}"},
            ],
        }
    )


@pytest.fixture
def expanded_frame():
    return pd.DataFrame(
        {
            "A": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5},
            "B": {0: 5, 1: 4, 2: 3, 3: 2, 4: 1},
            "C": {
                0: {
                    "A1": "3",
                    "A2": "1",
                    "A3": "5",
                    "AB": "{'C1': '3', 'C2': '1', 'C3': '5'}",
                },
                1: {
                    "A1": "3",
                    "A3": "5",
                    "AB": "{'C1': '3', 'C2': '1', 'C3': '5'}",
                },
                2: {"A1": "3", "A3": "5"},
                3: None,
                4: {"A1": "3", "A3": "5", "AB": "{'C1': '3', 'C3': '5'}"},
            },
            "A1": {0: "3", 1: "3", 2: "3", 3: pd.NA, 4: "3"},
            "A2": {0: "1", 1: pd.NA, 2: pd.NA, 3: pd.NA, 4: pd.NA},
            "A3": {0: "5", 1: "5", 2: "5", 3: pd.NA, 4: "5"},
            "AB_C1": {0: "3", 1: "3", 2: pd.NA, 3: pd.NA, 4: "3"},
            "AB_C2": {0: "1", 1: "1", 2: pd.NA, 3: pd.NA, 4: pd.NA},
            "AB_C3": {0: "5", 1: "5", 2: pd.NA, 3: pd.NA, 4: "5"},
        }
    )


def test_load_expand(nested_frame, expanded_frame):
    result_frame = expand_nested(nested_frame, "C")
    assert_frame_equal(expanded_frame, result_frame, check_dtype=False)
