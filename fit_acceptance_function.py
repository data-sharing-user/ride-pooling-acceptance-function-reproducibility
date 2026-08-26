"""Reproduce Eq. 61 and the reported OLS statistics from the released dataset."""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


SEED = 2678
TEST_SIZE = 0.10
DATA_FILE = Path(__file__).with_name("acceptance_estimation_data.xlsx")
SHEET_NAME = "estimation_data"

GROUP_COLUMNS = [
    "Normal_Trip_Distance",
    "Normal_Trip_Minutes",
    "Pickup_Time",
    "Detour_Ratio",
    "Discount",
    "Waiting_Time",
]
MODEL_FEATURES = [
    "Normal_Trip_Minutes",
    "Pickup_Time",
    "Detour_Ratio",
    "Discount",
    "Waiting_Time",
]
TERM_NAMES = [
    "Intercept",
    "Waiting_Time",
    "Discount",
    "Detour_Ratio",
    "Pickup_Time",
    "Normal_Trip_Minutes",
    "Pickup_Time*Discount",
    "Normal_Trip_Minutes*Discount",
    "Detour_Ratio*Discount*Waiting_Time",
]


def load_estimation_sample() -> tuple[np.ndarray, np.ndarray]:
    data = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)
    required = {"record_id", "record_source", "Acceptance", *GROUP_COLUMNS}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    expected_counts = {
        "survey_response": 23_690,
        "synthetic_high_anchor": 140,
        "synthetic_low_anchor": 108,
    }
    actual_counts = data["record_source"].value_counts().to_dict()
    if actual_counts != expected_counts:
        raise ValueError(f"Unexpected record counts: {actual_counts}")

    survey = data.loc[data["record_source"] == "survey_response"].copy()
    grouped = survey.groupby(GROUP_COLUMNS, as_index=False, sort=True)["Acceptance"].mean()
    if len(grouped) != 444:
        raise ValueError(f"Expected 444 grouped survey scenarios; found {len(grouped)}")

    anchors = data.loc[data["record_source"] != "survey_response"].copy()
    x_all = np.concatenate(
        [grouped[MODEL_FEATURES].to_numpy(float), anchors[MODEL_FEATURES].to_numpy(float)]
    )
    y_all = np.concatenate(
        [grouped["Acceptance"].to_numpy(float), anchors["Acceptance"].to_numpy(float)]
    )
    if len(y_all) != 692:
        raise ValueError(f"Expected 692 estimation observations; found {len(y_all)}")
    return x_all, y_all


def design_matrix(base: np.ndarray) -> np.ndarray:
    normal_minutes, pickup, detour, discount, waiting = base.T
    return np.column_stack(
        [
            np.ones(len(base)),
            waiting,
            discount,
            detour,
            pickup,
            normal_minutes,
            pickup * discount,
            normal_minutes * discount,
            detour * discount * waiting,
        ]
    )


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        "R2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    x_all, y_all = load_estimation_sample()
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_all, y_all, test_size=TEST_SIZE, random_state=SEED
    )
    train_matrix = design_matrix(x_train)
    validation_matrix = design_matrix(x_validation)
    model = sm.OLS(y_train, train_matrix).fit()

    train = calculate_metrics(y_train, model.predict(train_matrix))
    validation = calculate_metrics(y_validation, model.predict(validation_matrix))

    print("Passenger acceptance probability function (OLS)")
    print(f"Random seed: {SEED}")
    print(f"Estimation observations: {len(y_all)}")
    print(f"Train/validation observations: {len(y_train)}/{len(y_validation)}")
    print("\nTerm                                      Coefficient       p-value")
    for name, coefficient, p_value in zip(TERM_NAMES, model.params, model.pvalues):
        print(f"{name:40s} {coefficient:12.9f} {p_value:13.8g}")

    print(
        f"\nTrain: MSE={train['MSE']:.6f}, MAPE={train['MAPE']:.2f}%, "
        f"R2={train['R2']:.6f}"
    )
    print(
        f"Validation: MSE={validation['MSE']:.6f}, "
        f"MAPE={validation['MAPE']:.2f}%, R2={validation['R2']:.6f}"
    )


if __name__ == "__main__":
    main()
