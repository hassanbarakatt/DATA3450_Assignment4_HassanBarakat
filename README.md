Assignment 4: Classification - Credit Risk Analysis

This repository contains the completed work for Assignment 4. The project focuses on predicting loan status (default vs. non-default) using various classification models (Logistic Regression, Random Forest).

Project Structure

Asn4_HassanBarakat_Final.ipynb: The main Jupyter Notebook containing the analysis, data cleaning, modeling pipelines, and final deployment testing.

requirements.txt: List of Python libraries required to run the notebook.

train.csv / test.csv: Data files (ensure these are placed in the root directory).

Key Features Implemented

Data Cleaning:

Removed illogical outliers (Age > 100).

Handled missing values using Median Imputation for skewed numeric data.

Exploratory Data Analysis (EDA):

Analyzed Collinearity using Heatmaps.

Analyzed Credit Risk by Loan Grade (Good vs. Bad risk identification).

Modeling Pipeline:

Implemented sklearn.pipeline.Pipeline with ColumnTransformer.

Used StandardScaler for numeric features and OneHotEncoder for categorical ones.

Selected Random Forest (with balanced class weights) as the best performing model based on F1-score.

Deployment:

The model is retrained on the full dataset and ready to predict on unseen data automatically.

How to Run

Install dependencies:

pip install -r requirements.txt


Launch Jupyter Notebook:

jupyter notebook Asn4_HassanBarakat_Final.ipynb


Ensure train.csv and test.csv are in the same directory.

Run all cells to regenerate the analysis and predictions.

Author


Hassan Barakat

***************Asn4_HassanBarakat_Final.ipynb**************

This notebook is a complete machine learning pipeline for a Credit Risk Classification task, designed to predict whether a loan will default (loan_status = 1) or not.

1. Setup and Imports
Libraries: It imports standard data science libraries (pandas, numpy, seaborn, matplotlib) and Scikit-Learn components for modeling, metrics, and preprocessing.

Goal: The markdown defines the objective: predict loan_status while addressing data quality issues like imbalance, missing values, and outliers.

2. Data Cleaning & Exploratory Data Analysis (EDA)
Outlier Removal: It explicitly checks for and removes illogical data, specifically dropping rows where person_age > 100.

Collinearity Check: It generates a heatmap to ensure features aren't too heavily correlated (which could confuse some linear models).

Risk Analysis: It groups data by loan_grade to visualize default rates, answering the assignment's qualitative question about "good vs. bad credit risk" (identifying Grades A/B as safe and D/E as risky).

3. Preprocessing Pipeline
The code builds a robust sklearn.pipeline.Pipeline to handle raw data automatically:

Numeric Features: Missing values are filled with the median, and data is scaled using StandardScaler.

Categorical Features: Missing values are filled with the most frequent value, and text labels are converted to numbers using OneHotEncoder.

ColumnTransformer: These steps are bundled together so they apply to the correct columns automatically.

4. Model Selection & Training
Splitting: The data is split 80/20 into training and validation sets, using stratify=y to maintain the same percentage of defaults in both sets.

Models Compared:

Dummy Classifier: A baseline that always guesses the most frequent class (to see if the other models are actually learning).

Logistic Regression: A standard linear model.

Random Forest: An ensemble method. Notably, it uses class_weight='balanced' to help the model pay more attention to the minority class (defaults).

Metric: It uses the F1-score to judge performance, which is better than accuracy for imbalanced datasets.

5. Final Deployment
Refitting: After finding the best model (typically Random Forest here), it retrains that model on the entire training dataset (X, y) to maximize learning.

Testing: The final cell loads a separate test.csv, passes it through the exact same pipeline, and outputs the final F1 score, Classification Report, and Confusion Matrix to prove the model works on unseen data.


