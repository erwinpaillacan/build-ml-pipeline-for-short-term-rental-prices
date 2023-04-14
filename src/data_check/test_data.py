import pandas as pd
import numpy as np
import scipy.stats


def test_column_names(data):

    expected_colums = [
        "id",
        "name",
        "host_id",
        "host_name",
        "neighbourhood_group",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "last_review",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]

    these_columns = data.columns.values

    # This also enforces the same order
    assert list(expected_colums) == list(these_columns)


def test_neighborhood_names(data):

    known_names = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]

    neigh = set(data["neighbourhood_group"].unique())

    # Unordered check
    assert set(known_names) == set(neigh)


def test_proper_boundaries(data: pd.DataFrame):
    """
    Test proper longitude and latitude boundaries for properties in and around NYC
    """
    idx = data["longitude"].between(-74.25, -73.50) & data["latitude"].between(
        40.5, 41.2
    )

    assert np.sum(~idx) == 0


def test_similar_neigh_distrib(
    data: pd.DataFrame, ref_data: pd.DataFrame, kl_threshold: float
):
    """
    Apply a threshold on the KL divergence to detect if the distribution of the new data is
    significantly different than that of the reference dataset
    """
    dist1 = data["neighbourhood_group"].value_counts().sort_index()
    dist2 = ref_data["neighbourhood_group"].value_counts().sort_index()

    assert scipy.stats.entropy(dist1, dist2, base=2) < kl_threshold


########################################################
# Implement here test_row_count and test_price_range   #
########################################################
def test_row_count(data: pd.DataFrame):
    """
    The function checks if the number of rows in a given pandas DataFrame falls between 15000 and
    1000000.

    Args:
      data (pd.DataFrame): The parameter `data` is a pandas DataFrame that contains some data.
    """
    assert 15000 < data.shape[0] < 1000000


def test_price_range(data: pd.DataFrame, min_price: float, max_price: float):
    """
    The function tests whether all prices in a given DataFrame fall within a specified price range.

    Args:
      data (pd.DataFrame): The input data is a pandas DataFrame that contains information about
    products, including their prices.
      min_price (float): The minimum price value that the "price" column in the input DataFrame should
    have.
      max_price (float): The "max_price" parameter is a float value representing the maximum price
    allowed in a price range.
    """
    assert data["price"].between(min_price, max_price).all()
