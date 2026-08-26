# Passenger acceptance function: reproducibility files

This folder reproduces Eq. 61 and its reported OLS statistics.

## Files

- `acceptance_estimation_data.xlsx`: one worksheet containing 23,690 anonymized survey-response records and 248 explicitly labelled synthetic boundary-anchor observations.
- `fit_acceptance_function.py`: reads the Excel file, aggregates the survey responses into 444 scenarios, appends the 248 stored anchors, applies a 90/10 split with random seed 2678, and estimates the final OLS specification.

`record_source` distinguishes `survey_response`, `synthetic_high_anchor`, and `synthetic_low_anchor`. The synthetic records are not questionnaire responses. Direct and indirect personal identifiers are excluded.

## Run

Python 3.10+ with `numpy`, `pandas`, `openpyxl`, `scikit-learn`, and `statsmodels` is required.

```bash
python fit_acceptance_function.py
```

Expected results are Train R2 = 0.771790, Validation R2 = 0.720300, Train MSE = 0.023062, and Validation MSE = 0.026068.
