# Passenger acceptance function: reproducibility files

This folder reproduces Eq. 59 and its reported OLS statistics.

## Files

- `Ride-pooling-Survey-Shenzhen-main.zip`: survey questionnaire and related survey materials.
- `acceptance_estimation_data.xlsx`: 23,690 anonymized survey responses and 248 synthetic boundary-anchor observations. The `record_source` column identifies the source of each record.
- `fit_acceptance_function.py`: estimation script for the passenger acceptance function. 


## Run

Python 3.10+ with `numpy`, `pandas`, `openpyxl`, `scikit-learn`, and `statsmodels` is required.

```bash
python fit_acceptance_function.py
```

Expected results are Train R2 = 0.771790, Validation R2 = 0.720300, Train MSE = 0.023062, and Validation MSE = 0.026068.
