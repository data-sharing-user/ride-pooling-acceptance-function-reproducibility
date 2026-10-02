# Passenger acceptance function: reproducibility files

This folder reproduces Eq. 59 and its reported OLS statistics.

## Files

- `acceptance_estimation_data.xlsx`: one worksheet containing 23,690 anonymized survey-response records and 248 explicitly labelled synthetic boundary-anchor observations.


## Run

Python 3.10+ with `numpy`, `pandas`, `openpyxl`, `scikit-learn`, and `statsmodels` is required.

```bash
python fit_acceptance_function.py
```

Expected results are Train R2 = 0.771790, Validation R2 = 0.720300, Train MSE = 0.023062, and Validation MSE = 0.026068.
