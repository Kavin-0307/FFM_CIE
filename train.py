import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

DATA = "data/data.csv"
TARGET_RATE = 0.20

EXPENSES = [
    "Rent", "Loan_Repayment", "Insurance", "Groceries", "Transport",
    "Eating_Out", "Entertainment", "Utilities", "Healthcare",
    "Education", "Miscellaneous"
]
MONEY = ["Income"] + EXPENSES
NUMERIC = ["Age", "Dependents"]
CATEGORICAL = ["Occupation", "City_Tier"]

df = pd.read_csv(DATA)
df = df[df["Income"] > 0].copy()
df["savings_rate"] = df["Disposable_Income"] / df["Income"]
df["dti"] = df["Loan_Repayment"] / df["Income"]
df["approved"] = (
    (df["savings_rate"] >= TARGET_RATE) &
    (df["dti"] <= 0.25) &
    (df["Disposable_Income"] > 0)
).astype(int)

X = df[MONEY + NUMERIC + CATEGORICAL].copy()
X[MONEY] = np.log1p(X[MONEY])
X = pd.get_dummies(X, columns=CATEGORICAL, dtype=int)
feature_columns = X.columns.tolist()

potential_rates = {
    c: float((df["Potential_Savings_" + c] / df[c]).mean())
    for c in EXPENSES if "Potential_Savings_" + c in df.columns
}

X_tr, X_te, y_tr, y_te = train_test_split(
    X, df["approved"], test_size=0.2, random_state=42, stratify=df["approved"]
)

clf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
clf.fit(X_tr, y_tr)

preds = clf.predict(X_te)
probs = clf.predict_proba(X_te)[:, 1]

acc = accuracy_score(y_te, preds)
prec = precision_score(y_te, preds, zero_division=0)
rec = recall_score(y_te, preds, zero_division=0)
auc = roc_auc_score(y_te, probs)

print("AI Loan Classifier Metrics:")
print(f"Accuracy:  {acc:.1%}")
print(f"ROC AUC:   {auc:.3f}")
print(f"Precision: {prec:.1%}")
print(f"Recall:    {rec:.1%}")

rf_importances = dict(sorted(
    zip(feature_columns, [float(v) for v in clf.feature_importances_]),
    key=lambda x: x[1], reverse=True
))

shares = df[EXPENSES].div(df["Income"], axis=0)
scaler = StandardScaler().fit(shares)
Z = scaler.transform(shares)

kmeans = KMeans(n_clusters=3, n_init=10, random_state=42).fit(Z)
df["persona"] = kmeans.labels_

overall = shares.mean()
persona_names = {}
for c in range(3):
    dev = (shares[df["persona"] == c].mean() - overall).sort_values(ascending=False)
    persona_names[c] = f"High {dev.index[0]} & {dev.index[1]}"

persona_summary = df.groupby("persona").agg(
    size=("savings_rate", "size"),
    mean_savings_rate=("savings_rate", "mean"),
    approval_rate=("approved", "mean"),
).round(3).to_dict(orient="index")

persona_shares = shares.groupby(df["persona"]).mean().to_dict(orient="index")

joblib.dump({
    "classifier": clf,
    "columns": feature_columns,
    "money_cols": MONEY,
    "categorical": CATEGORICAL,
    "target_rate": TARGET_RATE,
    "kmeans": kmeans,
    "shares_scaler": scaler,
    "persona_names": persona_names,
    "persona_summary": persona_summary,
    "persona_shares": persona_shares,
    "potential_rates": potential_rates,
    "rf_importances": rf_importances,
    "metrics": {
        "Accuracy": acc,
        "ROC_AUC": auc,
        "Precision": prec,
        "Recall": rec,
    },
}, "models/model.joblib")

print("Saved model bundle to models/model.joblib")