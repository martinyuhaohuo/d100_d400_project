import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from yelp_predict.pre_processing import expand_nested, expand_cats


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


@pytest.fixture
def cat_list_frame():
    return pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [5, 4, 3, 2, 1],
            "C": [
                "Large, Medium, Small",
                "Large, Small",
                "Large",
                None,
                "Large",
            ],
        }
    )


@pytest.fixture
def cat_dummy_frame():
    return pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [5, 4, 3, 2, 1],
            "C": [
                "Large, Medium, Small",
                "Large, Small",
                "Large",
                None,
                "Large",
            ],
            "Large": [1, 1, 1, 0, 1],
            "Small": [1, 1, 0, 0, 0],
        }
    )


def test_expand_nested(nested_frame, expanded_frame):
    result_frame = expand_nested(nested_frame, "C")
    assert_frame_equal(expanded_frame, result_frame, check_dtype=False)


def test_expand_cats(cat_list_frame, cat_dummy_frame):
    result_frame = expand_cats(cat_list_frame, "C", 0.25)
    assert_frame_equal(cat_dummy_frame, result_frame, check_dtype=False)
