# Student Exam Score Prediction

A simple machine learning project that predicts a student's exam score from a small set of study and lifestyle inputs. The repository includes:

- a Streamlit app for interactive predictions
- a training notebook used to build the model
- the dataset used for experimentation
- the saved model artifact consumed by the app

## Project Overview

The app loads a trained regression model from `notebooks/final_model.pkl` and predicts an exam score based on five inputs:

- `study_hours_per_day`
- `attendance_percentage`
- `mental_health_rating`
- `sleep_hours`
- `part_time_job`

The output is shown as a predicted exam score capped between `0` and `100`.

## Repository Structure

```text
student-exam-score-prediction-ml/
|-- app.py
|-- README.md
|-- dataset/
|   `-- student_habits_performance.csv
`-- notebooks/
    |-- notebook.ipynb
    `-- final_model.pkl
```

## Dataset

Source file: `dataset/student_habits_performance.csv`

Rows: `1000`

Columns:

- `student_id`
- `age`
- `gender`
- `study_hours_per_day`
- `social_media_hours`
- `netflix_hours`
- `part_time_job`
- `attendance_percentage`
- `sleep_hours`
- `diet_quality`
- `exercise_frequency`
- `parental_education_level`
- `internet_quality`
- `mental_health_rating`
- `extracurricular_participation`
- `exam_score`

Although the dataset contains many features, the current model and app use only five of them.

## Important EDA

The exploratory data analysis in `notebooks/notebook.ipynb` covers both data quality checks and feature-level inspection before modeling.

### EDA steps included in the notebook

- `df.info()` to inspect column types and non-null counts
- duplicate-row check with `df.duplicated().sum()`
- descriptive statistics with `df.describe()`
- categorical distribution checks with `value_counts()`
- numeric feature histograms
- categorical count plots
- numeric correlation matrix
- correlation heatmap
- scatter plots of numeric features against `exam_score`
- box plots of categorical features against `exam_score`

### Key findings from the dataset

- total rows: `1000`
- duplicate rows found: `0`
- total missing values in the dataset: `91`
- the strongest positive numeric correlation with `exam_score` is `study_hours_per_day` (`0.8254`)
- other positive numeric relationships with `exam_score`:
  - `mental_health_rating` (`0.3215`)
  - `exercise_frequency` (`0.1601`)
  - `sleep_hours` (`0.1217`)
  - `attendance_percentage` (`0.0898`)
- negative numeric relationships with `exam_score`:
  - `netflix_hours` (`-0.1718`)
  - `social_media_hours` (`-0.1667`)

### Observations from categorical distributions

- `part_time_job` is imbalanced:
  - `No`: `785`
  - `Yes`: `215`
- `gender` is mostly balanced between `Female` and `Male`, with a smaller `Other` group
- `diet_quality` is led by `Fair`, followed by `Good`, then `Poor`
- `internet_quality` is mostly `Good` or `Average`
- `extracurricular_participation` is mostly `No`

### Why this EDA matters

The EDA helps justify the current feature set used in the app, especially:

- `study_hours_per_day`
- `attendance_percentage`
- `mental_health_rating`
- `sleep_hours`
- `part_time_job`

It also highlights two practical modeling concerns:

- the dataset contains missing values that should be handled explicitly in a production training pipeline
- some categorical classes are imbalanced, which can affect how representative the learned relationships are

## Model Training Summary

Training workflow lives in `notebooks/notebook.ipynb`.

Based on the notebook:

- target column: `exam_score`
- selected features:
  - `study_hours_per_day`
  - `attendance_percentage`
  - `mental_health_rating`
  - `sleep_hours`
  - `part_time_job`
- `part_time_job` is label-encoded before training
- the notebook compares:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor

Recorded notebook results:

| Model | RMSE | R2 |
|---|---:|---:|
| Linear Regression | 7.5490 | 0.8092 |
| Random Forest | 7.8603 | 0.7932 |
| Decision Tree | 8.7541 | 0.7435 |

The saved final model was trained from the notebook's best-performing configuration, which is `Linear Regression`.

## Streamlit App

File: `app.py`

The app:

1. loads the trained model with `joblib`
2. collects user input through sliders and a select box
3. converts `part_time_job` into a numeric value
4. sends the values to the model
5. displays the predicted exam score

### Current Input Controls

- Study Hours: `0.0` to `12.0`
- Attendance Percentage: `0.0` to `100.0`
- Mental Health Rating: `0.0` to `10.0`
- Sleep Hours per Day: `0.0` to `12.0`
- Part-time Job: `Yes` or `No`

### Prediction Encoding

`part_time_job` is encoded as:

- `Yes -> 1`
- `No -> 0`

## Requirements

This repository does not currently include a `requirements.txt`, but the code uses these Python packages:

- `streamlit`
- `numpy`
- `joblib`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`

For a minimal app-only setup, install:

```bash
pip install streamlit numpy joblib scikit-learn pandas
```

If you also want to run the notebook and visualizations, install:

```bash
pip install streamlit numpy joblib pandas scikit-learn matplotlib seaborn notebook
```

## How to Run

From the project root:

```bash
streamlit run app.py
```

By default, Streamlit opens the app locally at:

```text
http://localhost:8501
```

## How to Retrain the Model

1. Open `notebooks/notebook.ipynb`
2. Run the notebook cells in order
3. Confirm a new `final_model.pkl` is produced
4. Keep the saved model in `notebooks/final_model.pkl` so `app.py` can load it

## Troubleshooting

### `NameError: name 'warnings' is not defined`

This happens when `warnings.filterwarnings("ignore")` is used without importing `warnings`. The current `app.py` includes the required import.

### Model file not found

If Streamlit cannot load the model, verify this file exists:

```text
notebooks/final_model.pkl
```

### Prediction input mismatch

If you retrain the model with a different feature list or feature order, you must update `app.py` to send inputs in the same order used during training.

## Current Limitations

- The app uses only 5 features from a dataset with 16 columns.
- There is no `requirements.txt` yet.
- There are no automated tests in the repository.
- The model artifact is stored directly in the repo instead of being generated as part of a repeatable pipeline.

## Next Improvements

- add a `requirements.txt`
- add feature descriptions in the app UI
- show confidence or a prediction range
- validate and version the model artifact
- add tests for model loading and prediction flow
