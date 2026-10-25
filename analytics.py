import numpy as np
import pandas as pd
from config import EXPENSES, NEEDS, WANTS, INDIAN_EXPENSE_LABELS, format_inr

def predict_loan_approval(bundle, applicant_dict, disposable):
    clf = bundle["classifier"]
    columns = bundle["columns"]
    money_cols = bundle["money_cols"]
    categorical = bundle["categorical"]
    kmeans = bundle["kmeans"]
    shares_scaler = bundle["shares_scaler"]
    persona_names = bundle["persona_names"]

    row = pd.DataFrame([applicant_dict])
    row[money_cols] = np.log1p(row[money_cols])
    row = pd.get_dummies(row, columns=categorical, dtype=int)
    row = row.reindex(columns=columns, fill_value=0)

    approval_prob = float(clf.predict_proba(row)[0][1])

    income = applicant_dict["Income"]
    max_safe_emi = max(0.0, disposable - (0.20 * income))
    eligible_loan_amount = max_safe_emi * 30

    user_shares = pd.DataFrame([{k: applicant_dict[k] / income for k in EXPENSES}])
    persona_id = int(kmeans.predict(shares_scaler.transform(user_shares))[0])
    persona_name = persona_names[persona_id]

    return {
        "approval_prob": approval_prob,
        "is_approved": approval_prob >= 0.60,
        "max_safe_emi": max_safe_emi,
        "eligible_loan_amount": eligible_loan_amount,
        "persona_name": persona_name,
    }

def calculate_health_score(income, expenses):
    total_exp = sum(expenses[k] for k in EXPENSES if k in expenses)
    disposable = income - total_exp
    sr = disposable / income
    dti = expenses["Loan_Repayment"] / income
    needs_val = sum(expenses[k] for k in NEEDS)
    wants_val = sum(expenses[k] for k in WANTS)
    needs_ratio = needs_val / income
    wants_ratio = wants_val / income

    if sr >= 0.30:
        s_pts = 40.0
    elif sr >= 0.20:
        s_pts = 30.0 + 10.0 * (sr - 0.20) / 0.10
    elif sr >= 0.10:
        s_pts = 15.0 + 15.0 * (sr - 0.10) / 0.10
    elif sr > 0:
        s_pts = max(0.0, 15.0 * (sr / 0.10))
    else:
        s_pts = 0.0

    if dti == 0:
        d_pts = 25.0
    elif dti <= 0.10:
        d_pts = 20.0 + 5.0 * (1 - dti / 0.10)
    elif dti <= 0.20:
        d_pts = 10.0 + 10.0 * (1 - (dti - 0.10) / 0.10)
    elif dti <= 0.35:
        d_pts = max(0.0, 10.0 * (1 - (dti - 0.20) / 0.15))
    else:
        d_pts = 0.0

    n_pts = 10.0 if needs_ratio <= 0.50 else max(0.0, 10.0 * (1 - (needs_ratio - 0.50) / 0.20))
    w_pts = 10.0 if wants_ratio <= 0.30 else max(0.0, 10.0 * (1 - (wants_ratio - 0.30) / 0.15))

    cushion = disposable / total_exp if total_exp > 0 else 0
    if cushion >= 0.35:
        c_pts = 15.0
    elif cushion >= 0.20:
        c_pts = 10.0 + 5.0 * (cushion - 0.20) / 0.15
    elif cushion > 0:
        c_pts = max(0.0, 10.0 * (cushion / 0.20))
    else:
        c_pts = 0.0

    score = int(round(s_pts + d_pts + n_pts + w_pts + c_pts))
    score = max(0, min(100, score))

    if score >= 85:
        grade, status = "A", "Prime Borrower (Excellent)"
    elif score >= 70:
        grade, status = "B", "Healthy & Low Risk (Good)"
    elif score >= 55:
        grade, status = "C", "Moderate Strain (Fair)"
    elif score >= 40:
        grade, status = "D", "Stretched / High Burden"
    else:
        grade, status = "F", "Critical / High Risk of Default"

    return {
        "score": score,
        "grade": grade,
        "status": status,
        "savings_pts": round(s_pts, 1),
        "debt_pts": round(d_pts, 1),
        "balance_pts": round(n_pts + w_pts, 1),
        "cushion_pts": round(c_pts, 1),
    }

def get_50_30_20_summary(income, expenses, disposable):
    needs_total = sum(expenses[k] for k in NEEDS)
    wants_total = sum(expenses[k] for k in WANTS)
    needs_pct = needs_total / income
    wants_pct = wants_total / income
    savings_pct = max(0.0, disposable / income)

    budget_df = pd.DataFrame([
        {
            "Category": "Needs (Rent, Kirana, Bills, EMI)",
            "Actual": needs_pct,
            "Target": 0.50,
            "Status": "Safe (≤50%)" if needs_pct <= 0.50 else "High (>50%)",
        },
        {
            "Category": "Wants (Dining, Swiggy, OTT, Fun)",
            "Actual": wants_pct,
            "Target": 0.30,
            "Status": "Safe (≤30%)" if wants_pct <= 0.30 else "High (>30%)",
        },
        {
            "Category": "Savings & Emergency Reserve",
            "Actual": savings_pct,
            "Target": 0.20,
            "Status": "Safe (≥20%)" if savings_pct >= 0.20 else "Low (<20%)",
        },
    ])

    donut_df = pd.DataFrame([
        {
            "Category": INDIAN_EXPENSE_LABELS[k],
            "Amount": v,
            "Spend_INR": format_inr(v),
            "Share": v / sum(expenses.values()) if sum(expenses.values()) > 0 else 0,
        }
        for k, v in expenses.items() if v > 0
    ])

    return budget_df, donut_df, needs_total, wants_total, needs_pct, wants_pct, savings_pct

def get_cohort_benchmark(df_full, applicant_income, applicant_expenses, applicant_tier, applicant_occ):
    cohort_df = df_full[(df_full["City_Tier"] == applicant_tier) & (df_full["Occupation"] == applicant_occ)]
    if len(cohort_df) == 0:
        cohort_df = df_full[df_full["City_Tier"] == applicant_tier]

    actual_sr = (applicant_income - sum(applicant_expenses.values())) / applicant_income
    sr_pct = (cohort_df["savings_rate"] < actual_sr).mean() * 100

    bench_data = []
    for cat in EXPENSES:
        user_cat_share = applicant_expenses[cat] / applicant_income
        cohort_med_share = (cohort_df[cat] / cohort_df["Income"]).median()
        bench_data.append({
            "Category": INDIAN_EXPENSE_LABELS.get(cat, cat),
            "Your Share (% of Salary)": user_cat_share,
            "Typical Peer (% of Salary)": cohort_med_share,
            "Diff": user_cat_share - cohort_med_share,
        })
    b_df = pd.DataFrame(bench_data)
    return cohort_df, sr_pct, b_df

def get_optimization_plan(applicant_expenses, gap, potential_rates):
    options = sorted(
        ((cat, rate * applicant_expenses[cat]) for cat, rate in potential_rates.items()),
        key=lambda x: x[1],
        reverse=True,
    )
    plan = []
    remaining = gap
    for cat, available in options:
        if remaining <= 0:
            break
        cut = min(available, remaining)
        plan.append({
            "Category to Trim": INDIAN_EXPENSE_LABELS.get(cat, cat),
            "Current Spend (₹)": format_inr(applicant_expenses[cat]),
            "Recommended Cut (₹)": format_inr(cut),
            "Target Spend (₹)": format_inr(applicant_expenses[cat] - cut),
        })
        remaining -= cut
    return pd.DataFrame(plan), remaining
