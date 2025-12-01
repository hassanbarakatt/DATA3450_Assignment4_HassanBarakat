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