# Surviving-the-Algorithm-A-Comparative-ML-Study-of-Classification-Models-Using-Titanic-Dataset
CThis repository contains a comprehensive end-to-end machine learning research project focused on predicting passenger survival from the classic Titanic dataset. The study systematically compares three widely-used supervised classification algorithms — Logistic Regression, Decision Tree, and Random Forest — to identify the most effective model for binary classification tasks.
The project follows a complete machine learning pipeline: starting with data loading from Seaborn's built-in Titanic dataset (891 rows, 15 columns), followed by thorough Exploratory Data Analysis (EDA) including survival distribution, gender-based survival rates, passenger class analysis, and correlation heatmaps. Missing values in key columns such as age, deck, and embark_town are handled using mean imputation, column removal, and backward fill strategies respectively.
Feature engineering involves label encoding of all categorical and boolean variables to prepare the dataset for scikit-learn models. The data is split into 80% training and 20% test sets with a fixed random state to ensure reproducibility. Each model is trained, evaluated, and compared using accuracy score, precision, recall, F1-score, and confusion matrix metrics.
Results Summary:

🌲 Random Forest — 82.1% accuracy (Best Model)<b>
📈 Logistic Regression — 81.6% accuracy
🌿 Decision Tree — 75.4% accuracy

Random Forest outperformed all other models by leveraging ensemble learning and bagging techniques, significantly reducing overfitting compared to a single Decision Tree. The findings confirm that ensemble methods provide superior generalization on real-world datasets.
Tech Stack: Python · pandas · seaborn · matplotlib · scikit-learn
