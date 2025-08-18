import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
# Age
# Young
def young(x): return np.piecewise(x, [x <= 20, (x > 20) & (x < 30), x >= 30],
                    [1, lambda x: (30 - x) / 10, 0])
# Middle-Aged
def midage(x): return np.piecewise(x, [x < 30, (x > 20) & (x < 35), (x >= 35) & (x < 45), (x > 45) & (x < 50), x >= 50],
                       [0, lambda x: (x - 20) / 15, 1, lambda x: (50 - x) / 5, 0])
# senior
def senior(x): return np.piecewise(x, [x < 50, (x > 50) & (x < 60), (x >= 60)],
                       [0, lambda x: np.clip((x - 50) / 10,0,1), 1])
# Cholesterol - LDL
# Normal Cholesterol - LDL
def normal_ldl(x): return np.piecewise(x, [x < 100, (x >= 100) & (x < 129), x >= 129],
                    [1, lambda x: (129 - x) / 29, 0])

# Borderline Cholesterol - LDL
def border_ldl(x): return np.piecewise(x, [x < 100, (x >= 100) & (x < 120), (x >= 120) & (x < 140), (x >= 140) & (x < 160), x >= 160],
                       [0, lambda x: (x-100) / 20, 1, lambda x: (160 - x) / 20, 0])

# High Cholesterol - LDL
def high_ldl(x): return np.piecewise(x, [x < 160, (x >= 160) & (x < 170),(x >= 170)],
                       [0, lambda x: np.clip((170-x)/30,0,1), 1])
# Blood Pressure - Systolic
# Low BP
def low_bp(x): return np.piecewise(x, [x < 80, (x >= 80) & (x < 120), x >= 120],
                    [1, lambda x: (120 - x) / 40, 0])

# Normal BP
def normal_bp(x): return np.piecewise(x, [x < 80, (x >= 80) & (x < 90),(x >= 90) & (x < 121), (x >= 121) & (x < 129), x >= 130],
                    [0, lambda x: ( x-80 ) / 10,1,lambda x: (129 - x) / 10, 0])

# Elevated BP
def elevated_bp(x): return np.piecewise(x, [x < 121, (x >= 121) & (x < 126), (x >= 126) & (x < 135), (x >= 135) & (x < 140), x >= 140],
                       [0, lambda x: (x-121) / 5, 1, lambda x: (140 - x) / 5, 0])

# High BP
def high_bp(x): return np.piecewise(x, [x < 135, (x >= 135) & (x < 140),(x >= 140)],
                       [0, lambda x: np.clip((x-135)/5,0,1), 1])
# Blood Sugar (Glucose)
# Low Sugar (0-69)
def low_gl(x): return np.piecewise(x,[x < 60, (x >= 60) & (x < 70), x >= 70],
    [1, lambda x: (70 - x) / 10, 0])
# Normal Sugar (70-99)
def normal_gl(x): return np.piecewise(x, [x < 65, (x >= 65) & (x < 85),(x >= 85) & (x < 105),x >= 105],
                    [0, lambda x: (x-65 ) / 20, lambda x: (105-x) / 20, 0])
# Pre-Diabetic Sugar (100-125)
def prediabetic_gl(x): return np.piecewise(x, [x < 95, (x >= 95) & (x < 115), (x >=115) & (x < 130), x > 130],
                       [0, lambda x: (x-95)/20,lambda x: (130-x)/15, 0])
# High Sugar (126+)
def high_gl(x): return np.piecewise(x, [x < 120, (x >= 120) & (x < 140), (x >= 140)],
                       [0, lambda x: np.clip((x-120)/20,0,1), 1])

# to remove the above  piece
def age_young(x):     return np.piecewise(x, [x <= 20, (x > 20) & (x < 30), x >= 30],
                    [1, lambda x: (30 - x) / 10, 0])

def age_senior(x):    return np.piecewise(x, [x < 50, (x > 50) & (x < 60), (x >= 60)],
                       [0, lambda x: np.clip((x - 50) / 10,0,1), 1])

def sugar_normal(x):  return np.piecewise(x, [x < 65, (x >= 65) & (x < 85),(x >= 85) & (x < 105),x >= 105],
                    [0, lambda x: (x-65 ) / 20, lambda x: (105-x) / 20, 0])

def sugar_high(x):    return np.piecewise(x, [x < 120, (x >= 120) & (x < 140), (x >= 140)],
                       [0, lambda x: np.clip((x-120)/20,0,1), 1])

def ldl_normal(x):    return np.piecewise(x, [x < 100, (x >= 100) & (x < 129), x >= 129],
                    [1, lambda x: (129 - x) / 29, 0])

def ldl_high(x):      return np.piecewise(x, [x < 160, (x >= 160) & (x < 190),(x >= 190)],
                       [0, lambda x: np.clip((x-160)/30,0,1), 1])

# to remove the above  piece

# Define fuzzy rules manually
def sugeno_rules(age_val, sugar_val, ldl_val, bmi_val, bp_val):
    # Fuzzy memberships
    # BMI
    bmi_u = bmi_uw(np.array([bmi_val]))[0]
    bmi_n = bmi_normal(np.array([bmi_val]))[0]
    bmi_o = bmi_ow(np.array([bmi_val]))[0]
    bmi_h = bmi_obese(np.array([bmi_val]))[0]

    # AGE
    age_y = young(np.array([age_val]))[0]
    age_m = midage(np.array([age_val]))[0]
    age_s = senior(np.array([age_val]))[0]

    # CHOLESTEROL
    ldl_n = normal_ldl(np.array([ldl_val]))[0]
    ldl_b = border_ldl(np.array([ldl_val]))[0]
    ldl_h = high_ldl(np.array([ldl_val]))[0]

    # BLOOD PRESSURE
    bp_l = low_bp(np.array([bp_val]))[0]
    bp_n = normal_bp(np.array([bp_val]))[0]
    bp_e = elevated_bp(np.array([bp_val]))[0]
    bp_h = high_bp(np.array([bp_val]))[0]

    # BLOOD SUGAR
    sugar_l = low_gl(np.array([sugar_val]))[0]
    sugar_n = normal_gl(np.array([sugar_val]))[0]
    sugar_p = prediabetic_gl(np.array([sugar_val]))[0]
    sugar_d = high_gl(np.array([sugar_val]))[0]

    print("Memberships:")
    for name, val in zip(
            ['age_y', 'age_m', 'age_s', 'sugar_n', 'sugar_h', 'ldl_n', 'ldl_b', 'ldl_h', 'bmi_n', 'bmi_h', 'bp_n', 'bp_h'],
            [age_y, age_m, age_s, sugar_n, sugar_d, ldl_n, ldl_b, ldl_h, bmi_n, bmi_h, bp_n, bp_h]):
        print(f"  {name}: {val:.2f}")


    # SET RULES AND FIRING STRENGTHS
    # W1 = SUGAR_N & BMI_N & AGE_Y  : LOW RISK         YOUTHFULL WELLNESS
    # W2 = SUGAR_P & BMI_O          : MODERATE RISK    OBESITY INDUCED T2D
    # W3 = SUGAR_D & BMI_H          : HIGH RISK        OBESITY INDUCED T2D
    # W4 = AGE_S & BP_H & LDL_H     : HIGH RISK        STRESS INDUCED T2D
    # W5 = SUGAR_N & LDL_N & BP_N   : LOW RISK         WELL MAINTAINED
    # W6 = SUGAR_D & AGE_M & BMI_H  : HIGH RISK        OBESITY INDUCED T2D
    # W7 = SUGAR_P & LDL_B          : MODERATE RISK    TGL INDUCED T2D
    # W8 = SUGAR_D & BP_E           : HIGH RISK        STRESS INDUCED T2D
    # W9 = SUGAR_D & AGE_S & BMI_H  : HIGH RISK        OBESITY INDUCED T2D
    # W10 = AGE_Y & SUGAR_D & LDL_N : MODERATE RISK    YOUTH T2D

    # Rule firing strengths
    w1 = min(sugar_n, bmi_n,age_y)      # Low risk
    w2 = min(sugar_p, bmi_o)            # Moderate risk
    w3 = min(sugar_d, bmi_h)            # High Risk
    w4 = min(age_s, bp_h, ldl_h)        # High risk
    w5 = min(sugar_n, ldl_n, bp_n)      # Low Risk
    w6 = min(sugar_d, age_m, bmi_h)     # High risk
    w7 = min(sugar_p, ldl_b)            # Moderate risk
    w8 = min(sugar_d, bp_e)             # High risk
    w9 = min(sugar_d, age_s, bmi_h)     # High risk
    w10 = min(age_y, sugar_d, ldl_n)    # Moderate risk

    # Consequents (constants)
    f = [1.5, 5.0, 8.0, 9.0, 0.5, 8.0, 5.5, 8.0, 9.0, 5.0]

    # Sugeno weighted average defuzzification
    weights = [w1, w2, w3, w4, w5, w6, w7, w8, w9, w10]
    numerator = sum(w * fx for w, fx in zip(weights, f))
    denominator = sum(weights) + 1e-6

    risk_score = numerator / denominator

    print("Rule Strengths:")
    for i, w in enumerate(weights, 1):
        print(f"  w{i}: {w:.2f}")

    return risk_score, {f"w{i+1}": weights[i] for i in range(10)}

# Evaluate
input_vals = {
    "age": 73,
    "sugar": 89.31,
    "ldl": 170,
    "bmi": 20.54,
    "bp": 130
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
df = pd.read_csv("./Diabetes_Final_Data_V2.csv")

#Converting gulucose level to mmol as per the fuzzy sets
df["glucose_mmol"] = df["glucose"] * 18.008

# Function to run fuzzy logic on whole dataset
def evaluate_dataset(df):
    results = []
    for idx, row in df.iterrows():
        age_val = row["age"]
        sugar_val = row["glucose_mmol"] 
        ldl_val = row.get("ldl", 120)     
        bmi_val = row["bmi"]
        bp_val = row["systolic_bp"]

        risk_score, rule_strengths = sugeno_rules(age_val, sugar_val, ldl_val, bmi_val, bp_val)
        results.append({
            "index": idx,
            "age": age_val,
            "sugar_mmol": sugar_val,
            "ldl": ldl_val,
            "bmi": bmi_val,
            "bp": bp_val,
            "risk_score": risk_score
        })
    return pd.DataFrame(results)

# Run evaluation
results_df = evaluate_dataset(df)

#Printing first few values
print(results_df.head(18))