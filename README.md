# Surviving the Algorithm : A Comparative ML Study of Classification Models Using Titanic Dataset
# 🚢 Titanic Survival Prediction — Machine Learning Project

## 📌 Overview
This project performs an end-to-end machine learning analysis on the Titanic dataset to predict passenger survival.

---

## 📊 Dataset
- Source: Seaborn Titanic Dataset
- Rows: 891
- Columns: 15

---

## 🧹 Data Preprocessing
- Filled missing `age` with mean
- Dropped `deck`, `embarked`, `alive`
- Filled `embark_town` using backward fill
- Applied Label Encoding to categorical columns



## ⚙️ Feature Engineering
- Converted categorical columns using LabelEncoder

---

## 🔀 Train-Test Split
- 80% Training
- 20% Testing

---

## 🤖 Models Used

### Logistic Regression
- Accuracy: **81.6%**

### Decision Tree
- Accuracy: **75.4%**

### Random Forest ✅ (Best Model)
- Accuracy: **82.1%**

---

## 📊 Model Comparison

| Model | Accuracy |
|------|---------|
| Logistic Regression | 81.6% |
| Decision Tree | 75.4% |
| Random Forest | 82.1% |



## 🧠 Conclusion
- Random Forest performed best
- Logistic Regression also strong
- Decision Tree overfitted

---

## 🚀 Future Work
- Hyperparameter tuning
- Cross-validation
- Try XGBoost

---

## 📂 Project Structure
```
├── data/
├── notebooks/
├── images/
├── README.md
```

---

## ⭐ Give a star if you like this project!

