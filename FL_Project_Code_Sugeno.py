import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import ast

# Define universe
bmi = np.linspace(0,50,500)
age = np.linspace(0, 100, 500)
sugar = np.linspace(60, 200, 500)
ldl = np.linspace(0, 200, 500)
bp = np.linspace(0, 250, 500)

# Membership functions using np.piecewise
# BMI
# Underweight
def bmi_uw(x): return np.piecewise(x, [x < 16, (x >= 16) & (x < 18.5), x >= 18.5],
                    [1, lambda x: (18.5-x) / 2.5, 0])
# Healthy Weight
def bmi_normal(x): return np.piecewise(x, [x < 17, (x >= 17) & (x < 18.5), (x >= 18.5) & (x < 24.0), (x >= 24.0) & (x < 26), x > 26],
                       [0, lambda x: (x-17) / 1.5, 1, lambda x: (26-x) / 2, 0])
# overweight
def bmi_ow(x): return np.piecewise(x, [x < 24, (x >= 24) & (x < 26), (x >= 26) & (x < 28), (x >= 28) & (x < 30), x > 30],
                       [0, lambda x: (x-24) / 2, 1, lambda x: (30-x) / 2, 0])
# Obese
def bmi_obese(x): return np.piecewise(x, [x < 28, (x >= 28) & (x < 32),(x >= 32)],
                       [0, lambda x: np.clip((x-28)/4,0,1), 1])

# Show plots for BMI Membership Functions
x = np.linspace(10, 40, 300)
plt.plot(x, bmi_uw(x), label='Underweight')
plt.plot(x, bmi_normal(x), label='Normal')
plt.plot(x, bmi_ow(x), label='Overweight')
plt.plot(x, bmi_obese(x), label='Obese')
plt.legend()
plt.title("BMI Membership Functions")
plt.show()

# Age
# Young
# Young: peak at 20, zero at 40
def young(x):
    x = np.asarray(x)
    return np.piecewise(
        x,
        [x <= 30, (x > 30) & (x < 40), x >= 40],
        [1.0, lambda x: (40 - x) / 10.0, 0.0]
    )

# Midage: peak at 45, zero at 30 and 60
def midage(x):
        x = np.asarray(x)
        return np.piecewise(
            x,
            [x <= 30, (x > 30) & (x < 40), (x >= 40) & (x <= 50), (x > 50) & (x < 60), x >= 60],
            [0.0, lambda x: (x - 30) / 10.0, 1.0, lambda x: (60 - x) / 10.0, 0.0]
        )

# Senior: peak at 60, zero at 50 and 70
def senior(x):
        x = np.asarray(x)
        return np.piecewise(
        x,
        [x <= 55, (x > 55) & (x < 65), x >= 65],
        [0.0, lambda x: (x - 55) / 10.0, 1.0]
    )

# Plotting
x = np.linspace(0, 100, 500)
plt.figure(figsize=(8,5))
plt.plot(x, young(x), label='Young', linewidth=2)
plt.plot(x, midage(x), label='Midage', linewidth=2)
plt.plot(x, senior(x), label='Senior', linewidth=2)

plt.ylim(-0.05, 1.05)
plt.xlim(0, 100)
plt.xlabel("Age")
plt.ylabel("Membership Degree")
plt.title("Age Membership Functions (20–40, 30–60, 50–70)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()


# Normal LDL: full ≤100, decreases to 0 by 130
def normal_ldl(x):
        x = np.asarray(x)
        return np.piecewise(
        x,
        [x <= 100, (x > 100) & (x < 130), x >= 130],
        [1.0, lambda x: (130 - x) / 30.0, 0.0]
    )

# Borderline LDL: rises 120–130, full 130–150, falls 150–160
def borderline_ldl(x):
        x = np.asarray(x)
        return np.piecewise(
        x,
        [x < 120, (x >= 120) & (x < 130), (x >= 130) & (x <= 150), (x > 150) & (x < 160), x >= 160],
        [0.0, lambda x: (x - 120) / 10.0, 1.0, lambda x: (160 - x) / 10.0, 0.0]
    )

# High LDL: rises 150–170, full ≥170
def high_ldl(x):
        x = np.asarray(x)
        return np.piecewise(
        x,
        [x < 150, (x >= 150) & (x < 170), x >= 170],
        [0.0, lambda x: (x - 150) / 20.0, 1.0]
    )

# Plotting
x = np.linspace(75, 200, 500)
plt.figure(figsize=(8,5))
plt.plot(x, normal_ldl(x), label='Normal (<130)', linewidth=2)
plt.plot(x, borderline_ldl(x), label='Borderline (130–159)', linewidth=2)
plt.plot(x, high_ldl(x), label='High (≥160)', linewidth=2)

plt.ylim(-0.05, 1.05)
plt.xlim(75, 200)
plt.xlabel("LDL Cholesterol (mg/dL)")
plt.ylabel("Membership Degree")
plt.title("LDL Cholesterol Membership Functions")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()


# Low BP: full <= 80, fades out to 90
def low_bp(x):
        x = np.asarray(x)
        return np.piecewise(
        x,
        [x <= 80, (x > 80) & (x < 90), x >= 90],
        [1.0, lambda x: (90 - x) / 10.0, 0.0]
    )

# Normal BP: rises 85–90, full 90–115, fades 115–120
def normal_bp(x):
        x = np.asarray(x)
        return np.piecewise(
        x,
        [x < 85, (x >= 85) & (x < 90), (x >= 90) & (x <= 115), (x > 115) & (x < 120), x >= 120],
        [0.0, lambda x: (x - 85) / 5.0, 1.0, lambda x: (120 - x) / 5.0, 0.0]
    )

# Elevated BP: rises 118–120, full 120–125, fades 125–129
def elevated_bp(x):
        x = np.asarray(x)
        return np.piecewise(
        x,
        [x < 118, (x >= 118) & (x < 120), (x >= 120) & (x <= 125), (x > 125) & (x < 129), x >= 129],
        [0.0, lambda x: (x - 118) / 2.0, 1.0, lambda x: (129 - x) / 4.0, 0.0]
    )

# High BP: rises 128–130, full >= 130
def high_bp(x):
        x = np.asarray(x)
        return np.piecewise(
        x,
        [x < 128, (x >= 128) & (x < 130), x >= 130],
        [0.0, lambda x: (x - 128) / 2.0, 1.0]
    )

# Plotting
x = np.linspace(60, 200, 500)
plt.figure(figsize=(8,5))
plt.plot(x, low_bp(x), label='Low (<90)', linewidth=2)
plt.plot(x, normal_bp(x), label='Normal (90–120)', linewidth=2)
plt.plot(x, elevated_bp(x), label='Elevated (120–129)', linewidth=2)
plt.plot(x, high_bp(x), label='High (≥130)', linewidth=2)

plt.ylim(-0.05, 1.05)
plt.xlim(60, 200)
plt.xlabel("Systolic BP (mmHg)")
plt.ylabel("Membership Degree")
plt.title("Systolic Blood Pressure Membership Functions")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# Low Sugar (Hypoglycemia) – full <60, fade 60–70
def low_gl(x):
    x = np.asarray(x)
    return np.piecewise(
        x,
        [x <= 60, (x > 60) & (x < 70), x >= 70],
        [1.0, lambda x: (70 - x) / 10.0, 0.0]
    )

# Normal Sugar – rise 65–70, full 70–95, fade 95–100
def normal_gl(x):
    x = np.asarray(x)
    return np.piecewise(
        x,
        [x < 65, (x >= 65) & (x < 70), (x >= 70) & (x <= 95), (x > 95) & (x < 100), x >= 100],
        [0.0, lambda x: (x - 65) / 5.0, 1.0, lambda x: (100 - x) / 5.0, 0.0]
    )

# Pre-Diabetic Sugar – rise 95–100, full 100–120, fade 120–125
def prediabetic_gl(x):
    x = np.asarray(x)
    return np.piecewise(
        x,
        [x < 95, (x >= 95) & (x < 100), (x >= 100) & (x <= 120), (x > 120) & (x < 126), x >= 126],
        [0.0, lambda x: (x - 95) / 5.0, 1.0, lambda x: (126 - x) / 6.0, 0.0]
    )

# High Sugar (Diabetes) – rise 125–130, full ≥130
def high_gl(x):
    x = np.asarray(x)
    return np.piecewise(
        x,
        [x < 125, (x >= 125) & (x < 130), x >= 130],
        [0.0, lambda x: (x - 125) / 5.0, 1.0]
    )

# Plot
x = np.linspace(50, 200, 500)
plt.figure(figsize=(8,5))
plt.plot(x, low_gl(x), label='Low (<70)', linewidth=2)
plt.plot(x, normal_gl(x), label='Normal (70–99)', linewidth=2)
plt.plot(x, prediabetic_gl(x), label='Pre-Diabetic (100–125)', linewidth=2)
plt.plot(x, high_gl(x), label='High (≥126)', linewidth=2)

plt.ylim(-0.05, 1.05)
plt.xlabel("Fasting Blood Glucose (mg/dL)")
plt.ylabel("Membership Degree")
plt.title("Fasting Blood Glucose Membership Functions")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

from enum import Enum
class AgeStatus(Enum):
    Young = "Young Age"
    Mid = "Mid Age"
    Senior = "Senior Age"
class SugarStatus(Enum):
    Low = "Low Sugar"
    Normal = "Normal Sugar"
    Prediabetic = "Prediabetic Sugar"
    Diabetes = "Diabetes Sugar"
class LDLStatus(Enum):
    Normal = "Normal LDL"
    Borderline = "Borderline LDL"
    High = "High LDL"
class BMIStatus(Enum):
    Underweight = "Underweight BMI"
    Normal = "Normal BMI"
    Overweight = "Overweight BMI"
    Obese = "Obese BMI"
class BPStatus(Enum):
    Low = "Low BP"
    Normal = "Normal BP"
    Elevated = "Elevated BP"
    High = "High BP"
class RiskStatus(Enum):
    Low = "Low Risk"
    Moderate = "Moderate Risk"
    High = "High Risk"

# --- Rule List ---
# the first one of each line is Consequents
rules = [
    #         "w1": "LOW risk – Young, Sugar Normal, BMI Normal",
    [RiskStatus.Low.value , AgeStatus.Young.value, SugarStatus.Normal.value, BMIStatus.Normal.value],
    #         "w2": "MODERATE risk – Prediabetic, Overweight",
    [RiskStatus.Moderate.value , SugarStatus.Prediabetic.value, BMIStatus.Overweight.value],
    #         "w3": "HIGH risk – Diabetes, Obese",
    [RiskStatus.High.value , SugarStatus.Diabetes.value, BMIStatus.Obese.value],
    #         "w4": "HIGH risk – Senior, High BP, High LDL",
    [RiskStatus.High.value , AgeStatus.Senior.value, BPStatus.High.value, LDLStatus.High.value],
    #         "w5": "LOW risk – Normal Sugar, Normal LDL, Normal BP",
    [RiskStatus.Low.value , SugarStatus.Normal.value, LDLStatus.Normal.value, BPStatus.Normal.value],
    #         "w6": "HIGH risk – Midage, Diabetes, Obese",
    [RiskStatus.High.value , AgeStatus.Mid.value, SugarStatus.Diabetes.value, BMIStatus.Obese.value],
    #         "w7": "MODERATE risk – Prediabetic, Borderline LDL",
    [RiskStatus.Moderate.value , SugarStatus.Prediabetic.value, LDLStatus.Borderline.value],
    #         "w8": "HIGH risk – Diabetes, Elevated BP",
    [RiskStatus.High.value , SugarStatus.Diabetes.value, BPStatus.Elevated.value],
    #         "w9": "HIGH risk – Senior, Diabetes, Obese",
    [RiskStatus.High.value , AgeStatus.Senior.value, SugarStatus.Diabetes.value, BMIStatus.Obese.value],
    #         "w10": "MODERATE risk – Young, Diabetes, Normal LDL",
    [RiskStatus.Moderate.value , AgeStatus.Young.value, SugarStatus.Diabetes.value, LDLStatus.Normal.value],
    #         "w11": "HIGH risk – High Sugar, High BP",
    [RiskStatus.High.value , SugarStatus.Diabetes.value, BPStatus.High.value],
    #         "w12": "HIGH risk – High Sugar, High LDL",
    [RiskStatus.High.value , SugarStatus.Diabetes.value, LDLStatus.High.value],
    #         "w13": "HIGH risk – High Sugar, Obese",
    [RiskStatus.High.value , SugarStatus.Diabetes.value, BMIStatus.Obese.value],
    #         "w14": "HIGH risk – Very High Sugar",
    [RiskStatus.High.value , SugarStatus.Diabetes.value],
]

def firing(memberships):
    weights = []
    for rule in rules:
        # index 0 is Consequents
        values = [memberships[key] for key in rule[1:]]
        w = min(values)
        weights.append(w)
    return weights

def RuleDescriptions():
    rule_desc = {}
    for i in range(len(rules)):
        rule = rules[i]
        rule_desc[f"w{i+1}"]=f"{rule[0]} - {' , '.join(rule[1:])}"
    return rule_desc

# Define fuzzy rules manually
def sugeno_rules(age_val, sugar_val, ldl_val, bmi_val, bp_val):
    # --- Fuzzy memberships ---
    bmi_u, bmi_n, bmi_o, bmi_h = bmi_uw(bmi_val).item(), bmi_normal(bmi_val).item(), bmi_ow(bmi_val).item(), bmi_obese(bmi_val).item()
    age_y, age_m, age_s = young(age_val).item(), midage(age_val).item(), senior(age_val).item()
    ldl_n, ldl_b, ldl_h = normal_ldl(ldl_val).item(), borderline_ldl(ldl_val).item(), high_ldl(ldl_val).item()
    bp_l, bp_n, bp_e, bp_h = low_bp(bp_val).item(), normal_bp(bp_val).item(), elevated_bp(bp_val).item(), high_bp(bp_val).item()
    sugar_l, sugar_n, sugar_p, sugar_d = low_gl(sugar_val).item(), normal_gl(sugar_val).item(), prediabetic_gl(sugar_val).item(), high_gl(sugar_val).item()

    # Print active memberships
    memberships = {
        AgeStatus.Young.value: age_y, AgeStatus.Mid.value: age_m, AgeStatus.Senior.value: age_s,
        SugarStatus.Low.value: sugar_l, SugarStatus.Normal.value: sugar_n, SugarStatus.Prediabetic.value: sugar_p,SugarStatus.Diabetes.value: sugar_d,
        LDLStatus.Normal.value: ldl_n, LDLStatus.Borderline.value: ldl_b, LDLStatus.High.value: ldl_h,
        BMIStatus.Underweight.value: bmi_u, BMIStatus.Normal.value: bmi_n, BMIStatus.Overweight.value: bmi_o, BMIStatus.Obese.value: bmi_h,
        BPStatus.Low.value: bp_l, BPStatus.Normal.value: bp_n, BPStatus.Elevated.value: bp_e, BPStatus.High.value: bp_h,
    }
    print("Active memberships:")
    for k, v in memberships.items():
        if v > 0:
            print(f"  {k}: {v:.2f}")

    # --- Rule firing strengths ---
    weights = firing(memberships)

    if sum(weights) == 0:
        return 0.0, {f"w{i+1}": 0.0 for i in range(14)}

    # --- Consequents ---
    f = consequent([age_val, sugar_val, ldl_val, bmi_val, bp_val])

    # --- Sugeno weighted average ---
    numerator = sum(w * fx for w, fx in zip(weights, f))
    denominator = sum(weights)
    risk_score = numerator / denominator

    # --- Rule descriptions ---
    rule_desc = RuleDescriptions()

    print("Rule Strengths:")
    for i, w in enumerate(weights, 1):
        if w > 0:
            print(f"  w{i} ({rule_desc[f'w{i}']}): {w:.2f}")

    return risk_score, {f"w{i+1}": weights[i] for i in range(14)}

# Parameter matrix of all rule consequents
consequent_matrix = [
    # If it is a linear function, the values of the 5 columns on the left are not 0.
    # If it is a constant, the intercept of the rightmost column is not 0.
    #1age,2sugar,3ldl,4bmi,5bp,6intercept
    [0,0,0,0,0,1.5],# R1 consequent
    [0,0,0,0,0,5.0],# R2 consequent
    [0,0,0,0,0,8.0],# R3 consequent
    [0,0,0,0,0,9.0],# R4 consequent
    [0,0,0,0,0,0.5],# R5 consequent
    [0,0,0,0,0,8.0],# R6 consequent
    [0,0,0,0,0,5.5],# R7 consequent
    [0,0,0,0,0,8.0],# R8 consequent
    [0,0,0,0,0,9.0],# R9 consequent
    [0,0,0,0,0,5.0],# R10 consequent
    [0,0,0,0,0,9.5],# R11 consequent
    [0,0,0,0,0,9.0],# R12 consequent
    [0,0,0,0,0,8.5],# R13 consequent
    [0,0,0,0,0,7.5],# R14 consequent
]

# Calculate all rule consequences uniformly
def consequent(input_value):
    inputs = list(input_value) + [1]  # copy safely
    fs = []
    for ws_i in consequent_matrix:
        f = sum(w * v for w, v in zip(ws_i, inputs))
        fs.append(f)
    return fs

# Evaluate
input_vals = {
    "age": 35,
    "sugar": 244,
    "ldl": 180,
    "bmi": 20,
    "bp": 147
}

risk, rule_strengths = sugeno_rules(
    input_vals["age"],
    input_vals["sugar"],
    input_vals["ldl"],
    input_vals["bmi"],
    input_vals["bp"]
)

print(f"📊 Risk Score: {risk:.2f}")
print("🔍 Rule Firing Strengths:")
for rule, strength in rule_strengths.items():
    print(f"  {rule}: {strength:.2f}")


# Load CSV
df = pd.read_csv("Diabetes_Final_Data_V2.csv")

#Converting gulucose level to mmol as per the fuzzy sets
df["glucose_mgdl"] = df["glucose"] * 18.008

# Function to run fuzzy logic on whole dataset
def evaluate_dataset(df):
    results = []
    for idx, row in df.iterrows():
        age_val = row["age"]
        sugar_val = row["glucose_mgdl"]
        ldl_val = row.get("ldl", 120)
        bmi_val = row["bmi"]
        bp_val = row["systolic_bp"]

        risk_score, rule_strengths = sugeno_rules(age_val, sugar_val, ldl_val, bmi_val, bp_val)
        results.append({
            "index": idx,
            "age": age_val,
            "sugar_mgdl": sugar_val,
            "ldl": ldl_val,
            "bmi": bmi_val,
            "bp": bp_val,
            "risk_score": risk_score,
            "rule_strengths": rule_strengths  # Optional: for deeper analysis
        })
        if idx % 100 == 0:
            print(f"Processed {idx} rows...")

    return pd.DataFrame(results)

# Run evaluation
results_df = evaluate_dataset(df)

plt.scatter(results_df["age"], results_df["risk_score"], alpha=0.6)
plt.xlabel("Age")
plt.ylabel("Risk Score")
plt.title("Risk Score vs Age")
plt.grid(True)
plt.show()
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"risk_scores_{timestamp}.csv"
results_df.to_csv(filename, index=False)

# ========= LOAD DATA =========
# Change filename to your CSV path
df = pd.read_csv(filename)

# Expand rule strengths (convert string dicts into columns)
df_rules = df.copy()
df_rules['rule_strengths'] = df_rules['rule_strengths'].apply(ast.literal_eval)
rules_expanded = pd.json_normalize(df_rules['rule_strengths'])
df_rules = pd.concat([df_rules.drop(columns=['rule_strengths']), rules_expanded], axis=1)

# ========= DASHBOARD =========
plt.figure(figsize=(18, 12))

# 1. Distribution of Risk Scores
plt.subplot(2, 2, 1)
sns.histplot(df['risk_score'], bins=30, kde=True, color="steelblue")
plt.title("Distribution of Risk Scores")
plt.xlabel("Risk Score")
plt.ylabel("Count")

# 2. Sugar vs Risk Score (colored by Age)
plt.subplot(2, 2, 2)
sns.scatterplot(data=df, x='sugar_mgdl', y='risk_score', hue='age',
                palette='coolwarm', alpha=0.6, edgecolor=None)
plt.title("Sugar vs Risk Score (colored by Age)")
plt.xlabel("Sugar (mg/dL)")
plt.ylabel("Risk Score")

# 3. Correlation Heatmap
plt.subplot(2, 2, 3)
corr = df[['age', 'sugar_mgdl', 'ldl', 'bmi', 'bp', 'risk_score']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt=".2f")
plt.title("Correlation Heatmap")

# 4. Average Rule Activations
plt.subplot(2, 2, 4)
rules_mean = df_rules.drop(columns=['index','age','sugar_mgdl','ldl','bmi','bp','risk_score']).mean().sort_values(ascending=False)
rules_mean.plot(kind='bar', color='skyblue')
plt.title("Average Rule Activation Levels")
plt.ylabel("Activation Strength")

plt.tight_layout()
plt.show()

print("Risk Score Summary:")
print(results_df["risk_score"].describe())

#Printing first few values
# print(results_df.head(18))
