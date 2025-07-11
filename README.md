 Student Habits & Performance Analysis

This project analyzes student lifestyle habits — like study hours, sleep, diet quality, and social media usage — to understand how they relate to academic performance.

It uses:
-  **DataLoader** — safely loads CSV data
-  **DataCleaner** — checks for missing values, duplicates, and validates ranges
-  **StudentPerformanceAnalysis** — computes study time stats, correlations, and outliers
-  **VisualizationEngine** — generates study time histograms, scatter plots, and boxplots
-  **ScorePredictor** — trains a linear regression model to predict scores
-  **ReportExporter** — generates a Markdown + PDF report

---

Project Structure
project/
│
├── student_habits_performance.csv
├── Student_habits.py
├── report/
│ ├── report.md
│ ├── report.pdf
│ ├── score_model.pkl (optional)
└── README.md

What Each Part Does
DataLoader
Loads CSV data safely.

Handles missing file errors.

DataCleaner
Values_Missing() → checks for missing values.

duplicates_verification() → removes duplicates.

Validate_data_Range() → checks age for valid range.

StudentPerformanceAnalysis
mean_median_study_by_mental_health() → shows average & median study time grouped by mental_health_rating.

exam_sleep_correlation() → correlation between sleep_hours & exam_score.

scoial_media_outliers() → finds outliers in social_media_hours.

VisualizationEngine
study_time_histogram() → histogram of study time.

sleep_vs_exam_scores() → scatter plot: sleep vs. scores.

scores_by_diet_quality() → boxplot: scores vs. diet.

ScorePredictor
train_model() → trains linear regression on chosen features.

ReportExporter
Builds a Markdown report.

Converts it to PDF.

How to Run
1️ Make sure your student_habits_performance.csv is in the same folder.

2️ Adjust your CSV’s column names to match:

study_hours_per_day

sleep_hours

exam_score

diet_quality

social_media_hours

mental_health_rating

Output
Console: prints stats & checks

Plots: histograms, scatter & boxplots shown live

report/report.md + report/report.pdf