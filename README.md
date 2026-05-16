# 🎓 Student Performance Predictor — Machine Learning

> Predicts student pass/fail outcomes using 4 ML models with full comparison, ROC curves, feature importance, and a visual dashboard.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?style=flat-square)
![Accuracy](https://img.shields.io/badge/Best%20AUC-0.87+-green?style=flat-square)

---

## 📌 Project Overview

A supervised machine learning project that predicts whether a student will pass or fail based on behavioral and demographic features. Four classification models are trained and compared using accuracy, AUC-ROC score, and 5-fold cross-validation.

**Skills demonstrated:** Feature engineering · Model training & comparison · Cross-validation · ROC analysis · Confusion matrix · Feature importance

---

## 🤖 Models Compared

| Model | Type |
|-------|------|
| Logistic Regression | Linear baseline |
| Decision Tree | Interpretable tree |
| Random Forest | Ensemble (bagging) |
| Gradient Boosting | Ensemble (boosting) |

---

## 📊 Features Used

| Feature | Description |
|---------|-------------|
| Study_Hours | Daily study time (hrs) |
| Attendance_Pct | Class attendance percentage |
| Prev_Grade | Previous semester grade |
| Parental_Edu | Parent's education level |
| Internet_Access | Has internet at home |
| Extracurricular | Participates in activities |
| Sleep_Hours | Average daily sleep |
| Part_Time_Job | Works part-time |
| Gender | Student gender |
| Region | Urban / Rural / Suburban |

---

## 🚀 How to Run

```bash
git clone https://github.com/TheAishwarya-a/student-performance-ml.git
cd student-performance-ml

pip install pandas numpy matplotlib scikit-learn

python student_performance_predictor.py
```

**Output:**
- `student_data.csv` — dataset
- `ml_dashboard.png` — 6-panel comparison dashboard
- Console: accuracy table, classification report, key insights

---

## 📁 Project Structure

```
student-performance-ml/
├── student_performance_predictor.py
├── student_data.csv        # auto-generated
├── ml_dashboard.png        # auto-generated
└── README.md
```

---

## 👩‍💻 Author

**Aishwarya Bollipogu** · B.Tech AI & Data Science · VRSEC
[LinkedIn](https://www.linkedin.com/in/aishwarya-bollipogu-4a2b20259) · [GitHub](https://github.com/TheAishwarya-a)
