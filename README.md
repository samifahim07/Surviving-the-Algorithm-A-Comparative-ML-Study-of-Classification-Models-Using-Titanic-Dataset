# Surviving the Algorithm : A Comparative ML Study of Classification Models Using Titanic Dataset
Titanic Survival Prediction — Machine Learning Analysis
1. Project Overview
This project performs end-to-end machine learning on the classic Titanic dataset (loaded from Seaborn's built-in datasets). The goal is to predict passenger survival using Logistic Regression. The workflow covers data loading, exploratory data analysis (EDA), missing-value handling, feature encoding, model training, and evaluation.
2. Libraries Used
Library / Module	Purpose
pandas	Data manipulation and analysis (imported as pd).
seaborn	Statistical visualisation and built-in Titanic dataset (imported as sns).
matplotlib.pyplot	Plotting charts and figures (imported as plt).
sklearn.preprocessing.LabelEncoder	Encodes categorical string columns into integer labels.
pandas.api.types.is_numeric_dtype	Helper to check whether a column is numeric.
sklearn.model_selection.train_test_split	Splits data into training and test sets.
sklearn.linear_model.LinearRegression	Imported but Logistic Regression is used for the final model.
sklearn.linear_model.LogisticRegression	Binary classifier used to predict survival.
sklearn.metrics.accuracy_score	Computes classification accuracy.
sklearn.metrics.classification_report	Produces precision, recall, F1-score summary.
sklearn.metrics.confusion_matrix	Produces the confusion matrix.
3. Dataset Loading
The Titanic dataset is loaded directly from Seaborn:
df = sns.load_dataset("Titanic")
The dataset contains 891 rows and 15 columns. The first five rows (df.head()) and last five rows (df.tail()) are inspected to get an initial feel for the data.


3.1 Column Descriptions
Column	Type	Description
survived	int	Target variable — 0 = did not survive, 1 = survived
pclass	int	Passenger class: 1 = First, 2 = Second, 3 = Third
sex	object	Gender of the passenger
age	float	Age in years (has 177 missing values)
sibsp	int	Number of siblings / spouses aboard
parch	int	Number of parents / children aboard
fare	float	Ticket fare paid
embarked	object	Port of embarkation code (S / C / Q) — dropped later
class	object	Text version of pclass (First / Second / Third)
who	object	Passenger type: man, woman, or child
adult_male	bool	True if passenger is an adult male
deck	object	Deck letter — 688 missing values, dropped later
embark_town	object	Full name of embarkation port
alive	object	Text version of survived (yes / no) — dropped later
alone	bool	True if the passenger is travelling alone
4. Statistical Summary (df.describe())
df.describe() produces summary statistics for the numeric columns. Key observations:
•	Survival rate: mean survived ≈ 0.38, meaning roughly 38 % of passengers survived.
•	Passenger class (pclass): ranges 1–3; mean ≈ 2.31 indicates more third-class passengers.
•	Age: mean ≈ 29.7, ranging from 0.42 to 80 years.
•	Fare: highly skewed — min 0, max 512.33, mean ≈ 32.20.
•	sibsp: max 8; parch: max 6 — most passengers travelled with few or no family members.
5. Missing Value Handling
5.1 Identifying Missing Values
df.isnull().sum() reveals three columns with missing values:
Column	Missing Count	Strategy
age	177	Filled with column mean
embarked	2	Dropped (column removed)
deck	688	Dropped (column removed)
embark_town	2	Filled with backward fill (bfill)
alive	0	Dropped (column removed)

5.2  Dropping Unnecessary Columns
The columns deck, embarked, and alive are dropped using:
df = df.drop(['deck', 'embarked', 'alive'], axis=1)
Reasoning: deck has too many missing values (77 %); embarked and alive are redundant because embark_town and survived already capture the same information.
5.3  Filling age NaN with Mean
df['age'] = df['age'].fillna(df['age'].mean())
Mean imputation is used because age is a continuous numeric variable and the missing proportion (~20 %) is moderate.
5.4  Filling embark_town with Backward Fill
df['embark_town'] = df['embark_town'].bfill()
Only 2 values are missing, so backward fill propagates the next valid observation upward — a simple, low-impact fix.
After these steps, df.isnull().sum() returns 0 for every column — no missing values remain.
6. Exploratory Data Analysis (EDA)
Survival Count Plot
sns.countplot(x='survived', data=df)
A bar chart showing the raw count of survivors (1) vs non-survivors (0). More passengers did not survive than survived. 


 
Survival by Sex
sns.countplot(x='survived', hue='sex', data=df)
Female passengers had a dramatically higher survival rate than males, reflecting the "women and children first" evacuation policy.
 
Survival by Passenger Class
sns.countplot(x='survived', hue='class', data=df)
First-class passengers had the highest survival rates; third-class the lowest.
 
Correlation Matrix
df.corr()
A numeric correlation table is produced. Key correlations with survived: sex (–0.54), adult_male (–0.56), pclass (–0.34), fare (+0.26), who (+0.33).
Correlation Heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
A colour-coded heatmap of the correlation matrix makes relationships visually intuitive. Strong negative correlation between sex/adult_male and survival is clearly visible.
 

7. Feature Engineering & Label Encoding
The dataset contains several categorical (object / bool) columns that must be converted to numeric values before feeding into scikit-learn. LabelEncoder is applied to each non-numeric column:
le = LabelEncoder()
for col in df.columns:
    if not is_numeric_dtype(df[col]):
        df[col] = le.fit_transform(df[col])
After encoding, every column is numeric and ready for model training. Boolean columns (adult_male, alone) are automatically numeric already.
8. Train / Test Split
x = df.drop("survived", axis=1)  # Features
y = df["survived"]               # Target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
80 % of the data is used for training (≈ 712 rows) and 20 % for testing (179 rows). random_state=42 ensures reproducibility.
9. Model Training — Logistic Regression
lor = LogisticRegression()
model = lor.fit(x_train, y_train)
print("Train score :", model.score(x_train, y_train))
A default Logistic Regression model is trained on the training set. The solver did not fully converge within the default 100 iterations (a ConvergenceWarning is shown), suggesting data scaling or more iterations could improve results further.
Algorithm Descriptions
1. Logistic Regression
Logistic Regression is a fundamental statistical algorithm used for binary and multi-class classification tasks. Despite its name, it is a classification algorithm — not a regression one.

How it works:
Logistic Regression models the probability that a given input belongs to a certain class. It applies the sigmoid (logistic) function to a linear combination of input features, squashing the output to a value between 0 and 1. A threshold (usually 0.5) is then applied to make the final class prediction.

Key Characteristics:
•	Works best when the relationship between features and target is linear
•	Very fast to train and easy to interpret
•	Outputs probability scores, not just class labels
•	Sensitive to outliers and requires feature scaling
•	Assumes features are independent of each other

Result in this project:
Accuracy: 81.6% — Strong performance for a simple linear model. This suggests the dataset has some degree of linear separability.
 
2. Decision Tree
A Decision Tree is a flowchart-like model that makes decisions by splitting data based on feature values. It mimics human decision-making in a tree structure — starting from a root node and branching down to leaf nodes (final decisions).

How it works:
The algorithm selects the feature that best splits the data at each node (using criteria like Gini Impurity or Information Gain). It recursively divides the dataset until all items in a node belong to the same class, or a stopping criterion is met.

Key Characteristics:
•	Highly interpretable — easy to visualize and understand
•	Handles both numerical and categorical features
•	No need for feature scaling or normalization
•	Prone to overfitting, especially on deep trees
•	Sensitive to small changes in data (high variance)

Result in this project:
Accuracy: 75.4% — The lowest among the three models. This is likely due to overfitting on the training data, which reduced generalization to unseen examples.
 

3. Random Forest
BEST PERFORMING MODEL (Accuracy: 82.1%)
Random Forest is an ensemble learning method that builds multiple Decision Trees and combines their outputs to produce a more accurate, stable, and robust prediction. It is one of the most widely used and powerful algorithms in machine learning.

How it works:
Random Forest trains many Decision Trees on random subsets of the training data (a technique called bagging). Each tree also considers only a random subset of features at each split. During prediction, all trees vote on the result and the majority vote wins. This reduces overfitting dramatically compared to a single Decision Tree.

Key Characteristics:
•	Very high accuracy due to ensemble averaging
•	Robust to overfitting and noise in the data
•	Can handle missing values and high-dimensional data
•	Provides feature importance rankings
•	Slower to train than a single Decision Tree, but much more accurate

Result in this project:
Accuracy: 82.1% — Best performance among all models. The ensemble nature of Random Forest successfully reduced the variance and overfitting seen in the single Decision Tree.

 
Model Comparison Report
Decision Tree  |  Random Forest  |  Logistic Regression
Executive Summary
This report presents the results of training and evaluating three supervised machine learning classification algorithms on a given dataset. The goal was to identify the best-performing model based on accuracy. Below is a quick overview:
Model	Accuracy	Complexity	Status
Logistic Regression	81.6%	Low	Runner-up
Decision Tree	75.4%	Low	Baseline
Random Forest	82.1%	High	BEST MODEL

Model Performance Chart
The bar chart below shows the accuracy comparison of all three models. Random Forest achieved the highest accuracy, highlighted in blue.
 

Analysis & Recommendation
Why Random Forest Won
Random Forest outperformed the other models because it combines the predictions of many weak learners (Decision Trees) to form a strong collective prediction. This ensemble approach reduces both variance and bias — the two key sources of prediction error.

The Decision Tree suffered from overfitting (high variance), while Logistic Regression performed well but was limited by the linear boundary assumption. Random Forest avoided both of these pitfalls.

Recommendations
•	Deploy Random Forest as the final production model for this classification task
•	Tune hyperparameters (n_estimators, max_depth, min_samples_split) to potentially improve accuracy further
•	Apply cross-validation to ensure model generalizes well across different data splits
•	Investigate feature importance from the Random Forest to understand key predictors
•	Consider trying Gradient Boosting (XGBoost) as a next step for even higher performance


Conclusion
This project successfully evaluated three machine learning algorithms for a classification problem. The results clearly demonstrate that Random Forest is the most suitable model, achieving 82.1% accuracy — outperforming both Logistic Regression (81.6%) and Decision Tree (75.4%).

The experiment confirms that ensemble methods like Random Forest offer a significant advantage over single classifiers when dealing with real-world datasets. The relatively small gap between Logistic Regression and Random Forest also suggests that the data has meaningful linear separability, which Logistic Regression was able to partially exploit.

Future work may involve feature engineering, hyperparameter optimization, and testing additional ensemble methods to push accuracy beyond the 82.1% benchmark achieved in this study.

