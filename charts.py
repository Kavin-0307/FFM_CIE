import altair as alt
import pandas as pd
from config import INDIAN_EXPENSE_LABELS

def plot_budget_bar(budget_df):
    melted_b = budget_df.melt(
        id_vars=["Category", "Status"],
        value_vars=["Actual", "Target"],
        var_name="Type",
        value_name="Percentage",
    )
    return alt.Chart(melted_b).mark_bar(cornerRadius=4).encode(
        y=alt.Y("Category:N", title=None),
        x=alt.X("Percentage:Q", axis=alt.Axis(format="%", title="Share of Income")),
        color=alt.Color(
            "Type:N",
            scale=alt.Scale(domain=["Actual", "Target"], range=["#1976d2", "#90caf9"]),
        ),
        tooltip=[
            alt.Tooltip("Category:N"),
            alt.Tooltip("Type:N"),
            alt.Tooltip("Percentage:Q", format=".1%"),
        ],
    ).properties(height=180)

def plot_expense_donut(donut_df):
    return alt.Chart(donut_df).mark_arc(innerRadius=45).encode(
        theta=alt.Theta("Amount:Q", title="Spend (₹)"),
        color=alt.Color(
            "Category:N",
            scale=alt.Scale(scheme="tableau20"),
            legend=alt.Legend(orient="right", columns=2),
        ),
        tooltip=[
            alt.Tooltip("Category:N"),
            alt.Tooltip("Spend_INR:N", title="Monthly Spend"),
            alt.Tooltip("Share:Q", format=".1%", title="Share of Total"),
        ],
    ).properties(height=200)

def plot_peer_comparison(b_df):
    melted_bench = b_df.melt(
        id_vars=["Category", "Diff"],
        value_vars=["Your Share (% of Salary)", "Typical Peer (% of Salary)"],
        var_name="Who",
        value_name="Percentage",
    )
    return alt.Chart(melted_bench).mark_bar().encode(
        y=alt.Y("Category:N", sort=alt.EncodingSortField(field="Diff", order="descending"), title=None),
        x=alt.X("Percentage:Q", axis=alt.Axis(format="%", title="Share of Salary")),
        color=alt.Color(
            "Who:N",
            scale=alt.Scale(domain=["Your Share (% of Salary)", "Typical Peer (% of Salary)"], range=["#2e7d32", "#757575"]),
        ),
        tooltip=[alt.Tooltip("Category:N"), alt.Tooltip("Who:N"), alt.Tooltip("Percentage:Q", format=".1%")],
    ).properties(height=360)

def plot_tier_living_costs(sub_df):
    tier_summary = sub_df.groupby("City_Tier").agg(
        Rent_Share=("rent_share", "mean"),
        Savings_Rate=("savings_rate", "mean"),
        High_Risk_Rate=("high_risk", "mean"),
    ).reset_index()

    tier_melted = tier_summary.melt(
        id_vars=["City_Tier"],
        value_vars=["Rent_Share", "Savings_Rate", "High_Risk_Rate"],
        var_name="Metric",
        value_name="Proportion",
    )
    return alt.Chart(tier_melted).mark_bar(cornerRadius=4).encode(
        y=alt.Y("City_Tier:N", title=None),
        x=alt.X("Proportion:Q", axis=alt.Axis(format="%", title="Proportion of Salary")),
        color=alt.Color(
            "Metric:N",
            scale=alt.Scale(domain=["Rent_Share", "Savings_Rate", "High_Risk_Rate"], range=["#e53935", "#1e88e5", "#fb8c00"]),
        ),
        tooltip=[alt.Tooltip("City_Tier:N"), alt.Tooltip("Metric:N"), alt.Tooltip("Proportion:Q", format=".1%")],
    ).properties(height=220)

def plot_savings_distribution(sub_df):
    hist = alt.Chart(sub_df).mark_bar(color="#1976d2", opacity=0.7).encode(
        x=alt.X("savings_rate:Q", bin=alt.Bin(maxbins=30), axis=alt.Axis(format="%", title="Savings Rate")),
        y=alt.Y("count()", title="Number of Indian Applicants"),
        tooltip=[alt.Tooltip("count()", title="Applicants")],
    ).properties(height=260)
    rule = alt.Chart(pd.DataFrame({"t": [0.20]})).mark_rule(color="red", strokeDash=[4, 4], strokeWidth=2).encode(x="t:Q")
    return hist + rule

def plot_risk_heatmap(df_full):
    heat_data = df_full.groupby(["Occupation", "City_Tier"])["high_risk"].mean().reset_index()
    return alt.Chart(heat_data).mark_rect().encode(
        x=alt.X("City_Tier:N", title="City Tier"),
        y=alt.Y("Occupation:N", title="Occupation"),
        color=alt.Color(
            "high_risk:Q",
            scale=alt.Scale(scheme="orangered"),
            legend=alt.Legend(format="%", title="Risk Rate"),
        ),
        tooltip=[alt.Tooltip("Occupation:N"), alt.Tooltip("City_Tier:N"), alt.Tooltip("high_risk:Q", format=".1%")],
    ).properties(height=260)

def plot_feature_importance(rf_importances):
    top_feats = pd.DataFrame(list(rf_importances.items())[:6], columns=["Factor", "Importance"])
    friendly_names = {
        "Loan_Repayment": "Existing Loan EMIs (Debt Load)",
        "City_Tier_Tier_1": "Living in Tier 1 Metro (High Rent)",
        "Education": "Education & School Fees",
        "Dependents": "Family Dependents",
        "Income": "Monthly Salary Level",
        "Rent": "Base House Rent",
    }
    top_feats["Factor"] = top_feats["Factor"].map(lambda x: friendly_names.get(x, x.replace("_", " ")))
    return alt.Chart(top_feats).mark_bar(color="#2e7d32", cornerRadius=4).encode(
        x=alt.X("Importance:Q", title="Relative Weight", axis=alt.Axis(format="%")),
        y=alt.Y("Factor:N", sort="-x", title=None),
        tooltip=[alt.Tooltip("Factor:N"), alt.Tooltip("Importance:Q", format=".1%")],
    ).properties(height=260)

def plot_elasticity(potential_rates):
    pot_df = pd.DataFrame(list(potential_rates.items()), columns=["Category", "Potential Rate"])
    pot_df["Category"] = pot_df["Category"].map(lambda x: INDIAN_EXPENSE_LABELS.get(x, x))
    pot_df["Type"] = pot_df["Potential Rate"].apply(lambda x: "Flexible (Easy to trim)" if x > 0.10 else "Inelastic (Hard to cut)")
    return alt.Chart(pot_df).mark_bar(cornerRadius=4).encode(
        x=alt.X("Potential Rate:Q", title="Realistic Savings Potential", axis=alt.Axis(format="%")),
        y=alt.Y("Category:N", sort="-x", title=None),
        color=alt.Color(
            "Type:N",
            scale=alt.Scale(domain=["Flexible (Easy to trim)", "Inelastic (Hard to cut)"], range=["#1976d2", "#757575"]),
        ),
        tooltip=[alt.Tooltip("Category:N"), alt.Tooltip("Potential Rate:Q", format=".1%")],
    ).properties(height=260)
