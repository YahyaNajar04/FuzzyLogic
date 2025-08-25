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
    return np.piecewise(
        x,
        [x <= 30, (x > 30) & (x < 40), x >= 40],
        [1.0, lambda x: (40 - x) / 10.0, 0.0]
    )

# Midage: peak at 45, zero at 30 and 60
def midage(x):
        return np.piecewise(
            x,
            [x <= 30, (x > 30) & (x < 40), (x >= 40) & (x <= 50), (x > 50) & (x < 60), x >= 60],
            [0.0, lambda x: (x - 30) / 10.0, 1.0, lambda x: (60 - x) / 10.0, 0.0]
        )

# Senior: peak at 60, zero at 50 and 70
def senior(x):
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
    return np.piecewise(
        x,
        [x <= 100, (x > 100) & (x < 130), x >= 130],
        [1.0, lambda x: (130 - x) / 30.0, 0.0]
    )

# Borderline LDL: rises 120–130, full 130–150, falls 150–160
def borderline_ldl(x):
    return np.piecewise(
        x,
        [x < 120, (x >= 120) & (x < 130), (x >= 130) & (x <= 150), (x > 150) & (x < 160), x >= 160],
        [0.0, lambda x: (x - 120) / 10.0, 1.0, lambda x: (160 - x) / 10.0, 0.0]
    )

# High LDL: rises 150–170, full ≥170
def high_ldl(x):
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
    return np.piecewise(
        x,
        [x <= 80, (x > 80) & (x < 90), x >= 90],
        [1.0, lambda x: (90 - x) / 10.0, 0.0]
    )

# Normal BP: rises 85–90, full 90–115, fades 115–120
def normal_bp(x):
    return np.piecewise(
        x,
        [x < 85, (x >= 85) & (x < 90), (x >= 90) & (x <= 115), (x > 115) & (x < 120), x >= 120],
        [0.0, lambda x: (x - 85) / 5.0, 1.0, lambda x: (120 - x) / 5.0, 0.0]
    )

# Elevated BP: rises 118–120, full 120–125, fades 125–129
def elevated_bp(x):
    return np.piecewise(
        x,
        [x < 118, (x >= 118) & (x < 120), (x >= 120) & (x <= 125), (x > 125) & (x < 129), x >= 129],
        [0.0, lambda x: (x - 118) / 2.0, 1.0, lambda x: (129 - x) / 4.0, 0.0]
    )

# High BP: rises 128–130, full >= 130
def high_bp(x):
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
    return np.piecewise(
        x,
        [x <= 60, (x > 60) & (x < 70), x >= 70],
        [1.0, lambda x: (70 - x) / 10.0, 0.0]
    )

# Normal Sugar – rise 65–70, full 70–95, fade 95–100
def normal_gl(x):
    return np.piecewise(
        x,
        [x < 65, (x >= 65) & (x < 70), (x >= 70) & (x <= 95), (x > 95) & (x < 100), x >= 100],
        [0.0, lambda x: (x - 65) / 5.0, 1.0, lambda x: (100 - x) / 5.0, 0.0]
    )

# Pre-Diabetic Sugar – rise 95–100, full 100–120, fade 120–125
def prediabetic_gl(x):
    return np.piecewise(
        x,
        [x < 95, (x >= 95) & (x < 100), (x >= 100) & (x <= 120), (x > 120) & (x < 126), x >= 126],
        [0.0, lambda x: (x - 95) / 5.0, 1.0, lambda x: (126 - x) / 6.0, 0.0]
    )

# High Sugar (Diabetes) – rise 125–130, full ≥130
def high_gl(x):
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
    ldl_b = borderline_ldl(np.array([ldl_val]))[0]
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
df = pd.read_csv("Diabetes_Final_Data_V2.csv")

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