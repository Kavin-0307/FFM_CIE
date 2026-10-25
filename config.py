EXPENSES = [
    "Rent", "Loan_Repayment", "Insurance", "Groceries", "Transport",
    "Eating_Out", "Entertainment", "Utilities", "Healthcare",
    "Education", "Miscellaneous"
]

NEEDS = ["Rent", "Loan_Repayment", "Insurance", "Groceries", "Utilities", "Healthcare"]
WANTS = ["Transport", "Eating_Out", "Entertainment", "Education", "Miscellaneous"]

INDIAN_EXPENSE_LABELS = {
    "Rent": "House Rent",
    "Loan_Repayment": "Existing Loan EMI",
    "Insurance": "Life & Health Insurance",
    "Groceries": "Groceries & Kirana",
    "Transport": "Commute & Fuel",
    "Eating_Out": "Dining & Swiggy/Zomato",
    "Entertainment": "Entertainment & OTT",
    "Utilities": "Electricity, Water & Internet",
    "Healthcare": "Healthcare & Medical",
    "Education": "Education & School Fees",
    "Miscellaneous": "Miscellaneous & Shopping",
}

OCCUPATION_MAP = {
    "Salaried Professional (IT / Corporate)": "Professional",
    "Self-Employed / Business Owner": "Self_Employed",
    "Student": "Student",
    "Retired Senior Citizen": "Retired",
}
REV_OCCUPATION_MAP = {v: k for k, v in OCCUPATION_MAP.items()}

CITY_TIER_MAP = {
    "Tier 1 Metros (Mumbai, Delhi-NCR, Bengaluru, etc.)": "Tier_1",
    "Tier 2 Cities (Jaipur, Lucknow, Pune, Ahmedabad, etc.)": "Tier_2",
    "Tier 3 Towns & Rural Districts": "Tier_3",
}
REV_CITY_TIER_MAP = {v: k for k, v in CITY_TIER_MAP.items()}

DEFAULT_EXPENSES = {
    "Rent": 6400.0,
    "Loan_Repayment": 0.0,
    "Insurance": 1000.0,
    "Groceries": 3700.0,
    "Transport": 1900.0,
    "Eating_Out": 1000.0,
    "Entertainment": 1000.0,
    "Utilities": 1750.0,
    "Healthcare": 1200.0,
    "Education": 1700.0,
    "Miscellaneous": 580.0,
}

PRESETS = {
    "👤 Salaried Professional (Tier 2 City - Balanced)": {
        "income": 50000.0,
        "age": 35,
        "dependents": 1,
        "occupation": "Professional",
        "city_tier": "Tier_2",
        "expenses": DEFAULT_EXPENSES,
    },
    "🏙️ Metro High-Rent (Tier 1 - Mumbai / Bengaluru / Delhi)": {
        "income": 45000.0,
        "age": 28,
        "dependents": 0,
        "occupation": "Professional",
        "city_tier": "Tier_1",
        "expenses": {
            "Rent": 13500.0,
            "Loan_Repayment": 0.0,
            "Insurance": 1500.0,
            "Groceries": 5600.0,
            "Transport": 3500.0,
            "Eating_Out": 3200.0,
            "Entertainment": 2500.0,
            "Utilities": 2700.0,
            "Healthcare": 1400.0,
            "Education": 0.0,
            "Miscellaneous": 1200.0,
        },
    },
    "💳 High Loan EMI Burden (Self-Employed - Strained)": {
        "income": 52000.0,
        "age": 42,
        "dependents": 2,
        "occupation": "Self_Employed",
        "city_tier": "Tier_2",
        "expenses": {
            "Rent": 10400.0,
            "Loan_Repayment": 14000.0,
            "Insurance": 2000.0,
            "Groceries": 6500.0,
            "Transport": 3000.0,
            "Eating_Out": 1800.0,
            "Entertainment": 1200.0,
            "Utilities": 3100.0,
            "Healthcare": 2200.0,
            "Education": 4000.0,
            "Miscellaneous": 1000.0,
        },
    },
    "💰 Tier 3 Prudent Saver (Low Cost of Living)": {
        "income": 38000.0,
        "age": 32,
        "dependents": 1,
        "occupation": "Professional",
        "city_tier": "Tier_3",
        "expenses": {
            "Rent": 5700.0,
            "Loan_Repayment": 0.0,
            "Insurance": 1200.0,
            "Groceries": 4500.0,
            "Transport": 1800.0,
            "Eating_Out": 800.0,
            "Entertainment": 600.0,
            "Utilities": 2000.0,
            "Healthcare": 1000.0,
            "Education": 2000.0,
            "Miscellaneous": 500.0,
        },
    },
}

def format_inr(number):
    try:
        num = float(number)
    except (ValueError, TypeError):
        return f"₹{number}"
    sign = "-" if num < 0 else ""
    num = abs(round(num))
    s = str(num)
    if len(s) <= 3:
        return f"{sign}₹{s}"
    last3 = s[-3:]
    rest = s[:-3]
    groups = []
    while len(rest) > 2:
        groups.append(rest[-2:])
        rest = rest[:-2]
    if rest:
        groups.append(rest)
    groups.reverse()
    return f"{sign}₹{','.join(groups)},{last3}"
