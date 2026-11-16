"""
Shared helpers for the ADA 2026 project (border premium in Haute-Savoie).

Everything used by more than one stage lives here and is imported
(Requirement 4.9).

Tool declaration: drafted with Claude (Anthropic), asked to write the project
paths, the haversine distance, the distance to the nearest border crossing,
the preparation of the model variables, the evaluation on unseen
municipalities and the conversion of coordinates to kilometres.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict

SEED = 42  # one seed for the whole project, so every number can be reproduced

# Paths are built from the location of this file: no absolute path anywhere.
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"

DEPARTMENT = "74"            # Haute-Savoie
YEARS = range(2021, 2026)    # 2021 to 2025 included (2020 is no longer distributed)

N_FOLDS = 5                  # folds of the cross-validation by municipality

EARTH_RADIUS_KM = 6371.0
REFERENCE_POINT = (46.05, 6.40)  # latitude, longitude near the centre of Haute-Savoie

# Variables describing the home itself, and the municipality controls
PROPERTY_FEATURES = ["log_surface", "is_house", "rooms", "log_land", "n_outbuildings", "new_build"]
CONTROL_FEATURES = ["dist_annecy_km", "median_income", "log_population", "second_home_share"]


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two points given in degrees.

    Works on numbers or numpy arrays (standard haversine formula).
    """
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def load_border_crossings():
    """Hand-compiled list of road border crossings between Haute-Savoie and Switzerland."""
    return pd.read_csv(RAW / "border_crossings.csv")


def distance_to_nearest_crossing(lat, lon, crossings=None):
    """For each (lat, lon), return the distance in km to the nearest border
    crossing and the name of that crossing.

    lat, lon: array-like of the same length (degrees).
    Returns two numpy arrays: distances (km) and crossing names.
    """
    if crossings is None:
        crossings = load_border_crossings()
    lat = np.asarray(lat, dtype=float)[:, None]   # shape (n_sales, 1)
    lon = np.asarray(lon, dtype=float)[:, None]
    c_lat = crossings["lat"].to_numpy()[None, :]   # shape (1, n_crossings)
    c_lon = crossings["lon"].to_numpy()[None, :]
    dist = haversine_km(lat, lon, c_lat, c_lon)    # shape (n_sales, n_crossings)
    nearest = dist.argmin(axis=1)
    return dist.min(axis=1), crossings["name"].to_numpy()[nearest]


def load_model_data():
    """Load the sales with their variables and prepare what every model needs.

    Adds log_surface, log_land (log of 1 + land surface), fills the few missing
    numbers of rooms with the median, and adds the target `price_vs_quarter`:
    the log price per m2 minus the average log price of the same quarter.
    A value of 0.10 means "about 10% more expensive than the average home sold
    in the same quarter". This removes market-wide moves (interest rates,
    inflation) so that homes are only compared with homes sold at the same time.
    """
    sales = pd.read_csv(PROCESSED / "sales_features.csv", dtype={"code_commune": str})
    sales["log_surface"] = np.log(sales["surface"])
    sales["log_land"] = np.log1p(sales["land_surface"])
    sales["rooms"] = sales["rooms"].fillna(sales["rooms"].median())
    quarter_mean = sales.groupby("quarter")["log_price_m2"].transform("mean")
    sales["price_vs_quarter"] = sales["log_price_m2"] - quarter_mean
    return sales


def evaluate_by_commune(model, X, y, communes):
    """Evaluate a model on municipalities it has never seen.

    The municipalities are split in N_FOLDS groups. The model is trained on
    four groups and predicts the fifth, in turn, so every sale is predicted by
    a model that saw no sale from its municipality.
    Returns the scores (R2, RMSE, MAE) and the out-of-sample predictions.
    """
    predictions = cross_val_predict(model, X, y, groups=communes, cv=GroupKFold(N_FOLDS))
    scores = {
        "R2": r2_score(y, predictions),
        "RMSE": mean_squared_error(y, predictions) ** 0.5,
        "MAE": mean_absolute_error(y, predictions),
    }
    return scores, predictions


def to_km(lat, lon):
    """Convert latitude/longitude (degrees) to x/y positions in km, measured
    east and north of REFERENCE_POINT. Accurate enough at the scale of one
    department, and it lets a model treat one km east like one km north.
    """
    lat0, lon0 = REFERENCE_POINT
    x = (np.asarray(lon, dtype=float) - lon0) * 111.32 * np.cos(np.radians(lat0))
    y = (np.asarray(lat, dtype=float) - lat0) * 110.57
    return x, y
