"""
Student Performance Predictor — Machine Learning Project
Author: Aishwarya Bollipogu
GitHub: https://github.com/TheAishwarya-a

Predicts whether a student will pass or fail based on study habits,
attendance, parental education, and demographic features.
Uses multiple ML models with comparison and feature importance analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

ROSE  = "#C2716F"
TEAL  = "#3D7D7A"
GOLD  = "#B8860B"
INK   = "#1C1917"
MUTED = "#78716C"

plt.rcParams.update({
    "figure.facecolor": "#FAF7F2", "axes.facecolor": "#FAF7F2",
    "axes.edgecolor": "#DDD8D0",   "text.color": INK,
    "xtick.color": MUTED,          "ytick.color": MUTED,
    "grid.color": "#EDE7DC",       "grid.linestyle": "--", "grid.alpha": 0.7,
})

# ── 1. Generate synthetic student dataset ─────────────────────────────────────
np.random.seed(0)
n = 800

study_hours       = np.random.uniform(0, 10, n)
attendance_pct    = np.random.uniform(40, 100, n)
prev_grades       = np.random.uniform(30, 95, n)
parental_edu      = np.random.choice(["None", "High School", "Bachelor's", "Master's"], n,
                                      p=[0.10, 0.35, 0.40, 0.15])
internet_access   = np.random.choice([0, 1], n, p=[0.20, 0.80])
extracurricular   = np.random.choice([0, 1], n, p=[0.45, 0.55])
sleep_hours       = np.random.uniform(4, 10, n)
part_time_job     = np.random.choice([0, 1], n, p=[0.70, 0.30])
gender            = np.random.choice(["Male", "Female"], n)
region            = np.random.choice(["Urban", "Rural", "Suburban"], n, p=[0.50, 0.25, 0.25])

# Realistic pass probability
pass_prob = (
    0.30 * (study_hours / 10) +
    0.25 * (attendance_pct / 100) +
    0.25 * (prev_grades / 95) +
    0.08 * internet_access +
    0.05 * extracurricular +
    0.04 * (sleep_hours / 10) -
    0.03 * part_time_job +
    np.random.normal(0, 0.07, n)
)
pass_prob = np.clip(pass_prob, 0, 1)
passed    = (pass_prob > 0.52).astype(int)

df = pd.DataFrame({
    "Study_Hours":      np.round(study_hours, 1),
    "Attendance_Pct":   np.round(attendance_pct, 1),
    "Prev_Grade":       np.round(prev_grades, 1),
    "Parental_Edu":     parental_edu,
    "Internet_Access":  internet_access,
    "Extracurricular":  extracurricular,
    "Sleep_Hours":      np.round(sleep_hours, 1),
    "Part_Time_Job":    part_time_job,
    "Gender":           gender,
    "Region":           region,
    "Passed":           passed,
})

df.to_csv("student_data.csv", index=False)
print(f"✓ Dataset saved → student_data.csv  ({len(df)} rows)")
print(f"  Pass rate: {passed.mean()*100:.1f}%  |  Fail rate: {(1-passed.mean())*100:.1f}%\n")

# ── 2. Preprocessing ──────────────────────────────────────────────────────────
le = LabelEncoder()
df_enc = df.copy()
for col in ["Parental_Edu", "Gender", "Region"]:
    df_enc[col] = le.fit_transform(df[col])

X = df_enc.drop("Passed", axis=1)
y = df_enc["Passed"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ── 3. Train multiple models ───────────────────────────────────────────────────
models = {
    "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]),
    "Decision Tree":       DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, random_state=42),
}

results = {}
print("=" * 55)
print(f"{'Model':<22} {'Accuracy':>9} {'AUC-ROC':>9} {'CV Mean':>9}")
print("=" * 55)
for name, model in models.items():
    model.fit(X_train, y_train)
    preds    = model.predict(X_test)
    proba    = model.predict_proba(X_test)[:, 1]
    acc      = accuracy_score(y_test, preds)
    auc      = roc_auc_score(y_test, proba)
    cv       = cross_val_score(model, X, y, cv=5, scoring="accuracy").mean()
    results[name] = {"model": model, "preds": preds, "proba": proba,
                     "acc": acc, "auc": auc, "cv": cv}
    print(f"  {name:<20} {acc*100:>8.1f}%  {auc:>9.3f}  {cv*100:>8.1f}%")
print("=" * 55)

best_name  = max(results, key=lambda k: results[k]["auc"])
best       = results[best_name]
print(f"\n  Best model: {best_name} (AUC = {best['auc']:.3f})\n")
print(classification_report(y_test, best["preds"], target_names=["Fail", "Pass"]))

# ── 4. Dashboard ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.suptitle("Student Performance Prediction — ML Dashboard", fontsize=16, fontweight="bold", color=INK, y=0.98)
gs  = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)
ax1 = fig.add_subplot(gs[0, 0])   # model comparison
ax2 = fig.add_subplot(gs[0, 1])   # confusion matrix
ax3 = fig.add_subplot(gs[0, 2])   # ROC curves
ax4 = fig.add_subplot(gs[1, 0])   # feature importance
ax5 = fig.add_subplot(gs[1, 1])   # study hours distribution
ax6 = fig.add_subplot(gs[1, 2])   # attendance distribution

# — Model accuracy bar —
names = list(results.keys())
accs  = [results[n]["auc"] for n in names]
bar_c = [ROSE if n == best_name else "#C4A8A7" for n in names]
bars  = ax1.barh(names, accs, color=bar_c, edgecolor="none", height=0.55)
ax1.set_xlim(0.5, 1.0)
ax1.set_title("Model AUC-ROC Comparison", fontsize=11, fontweight="bold", color=INK, pad=8)
ax1.set_xlabel("AUC-ROC Score", fontsize=9)
ax1.grid(True, axis="x")
for bar, val in zip(bars, accs):
    ax1.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
             f"{val:.3f}", va="center", fontsize=8.5, color=INK)

# — Confusion matrix heatmap —
cm = confusion_matrix(y_test, best["preds"])
im = ax2.imshow(cm, cmap="RdPu", aspect="auto")
ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
ax2.set_xticklabels(["Predicted Fail", "Predicted Pass"], fontsize=8)
ax2.set_yticklabels(["Actual Fail", "Actual Pass"], fontsize=8)
ax2.set_title(f"Confusion Matrix — {best_name}", fontsize=11, fontweight="bold", color=INK, pad=8)
for i in range(2):
    for j in range(2):
        ax2.text(j, i, str(cm[i, j]), ha="center", va="center",
                 fontsize=16, fontweight="bold",
                 color="white" if cm[i, j] > cm.max()/2 else INK)

# — ROC curves —
colors_roc = [ROSE, TEAL, GOLD, "#9B8EA0"]
for (name, res), c in zip(results.items(), colors_roc):
    fpr, tpr, _ = roc_curve(y_test, res["proba"])
    ax3.plot(fpr, tpr, color=c, lw=2, label=f"{name} ({res['auc']:.2f})")
ax3.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
ax3.set_title("ROC Curves", fontsize=11, fontweight="bold", color=INK, pad=8)
ax3.set_xlabel("False Positive Rate", fontsize=9)
ax3.set_ylabel("True Positive Rate", fontsize=9)
ax3.legend(fontsize=7.5, frameon=True, framealpha=0.9)
ax3.grid(True)

# — Feature importance (Random Forest) —
rf      = results["Random Forest"]["model"]
feat_imp= pd.Series(rf.feature_importances_, index=X.columns).sort_values()
colors_fi = [TEAL if v > feat_imp.median() else "#A8C4C2" for v in feat_imp.values]
ax4.barh(feat_imp.index, feat_imp.values, color=colors_fi, edgecolor="none", height=0.65)
ax4.set_title("Feature Importance (Random Forest)", fontsize=11, fontweight="bold", color=INK, pad=8)
ax4.set_xlabel("Importance Score", fontsize=9)
ax4.grid(True, axis="x")

# — Study hours vs pass/fail —
passed_hours = df[df["Passed"] == 1]["Study_Hours"]
failed_hours = df[df["Passed"] == 0]["Study_Hours"]
ax5.hist(passed_hours, bins=20, alpha=0.7, color=TEAL, label="Pass", edgecolor="none")
ax5.hist(failed_hours, bins=20, alpha=0.7, color=ROSE, label="Fail", edgecolor="none")
ax5.set_title("Study Hours Distribution", fontsize=11, fontweight="bold", color=INK, pad=8)
ax5.set_xlabel("Study Hours per Day", fontsize=9)
ax5.set_ylabel("Count", fontsize=9)
ax5.legend(fontsize=9)
ax5.grid(True, axis="y")

# — Attendance vs pass/fail —
passed_att = df[df["Passed"] == 1]["Attendance_Pct"]
failed_att = df[df["Passed"] == 0]["Attendance_Pct"]
ax6.hist(passed_att, bins=20, alpha=0.7, color=TEAL, label="Pass", edgecolor="none")
ax6.hist(failed_att, bins=20, alpha=0.7, color=ROSE, label="Fail", edgecolor="none")
ax6.set_title("Attendance % Distribution", fontsize=11, fontweight="bold", color=INK, pad=8)
ax6.set_xlabel("Attendance (%)", fontsize=9)
ax6.set_ylabel("Count", fontsize=9)
ax6.legend(fontsize=9)
ax6.grid(True, axis="y")

plt.savefig("ml_dashboard.png", dpi=150, bbox_inches="tight", facecolor="#FAF7F2")
print("✓ Dashboard saved → ml_dashboard.png")
plt.close()
print("\nProject complete!")
