from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# ================= LOAD MODEL =================

model = joblib.load("model/best_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# Feature names (same order as training)
feature_names = [
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "FLAG_MOBIL",
    "FLAG_WORK_PHONE",
    "FLAG_PHONE",
    "FLAG_EMAIL",
    "OCCUPATION_TYPE",
    "CNT_FAM_MEMBERS"
]

# Numerical columns that were scaled during training
numeric_cols = [
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "CNT_FAM_MEMBERS"
]


# ================= HOME PAGE =================

@app.route("/")
def home():
    return render_template("home.html")

# ================= PREDICTION PAGE =================

@app.route("/predict")
def predict():
    return render_template("index.html")


# ================= RESULT PAGE =================

@app.route("/result", methods=["POST"])
def result():
    # ===========================
    # Basic Eligibility Rules
    # ===========================

    # Validate hidden fields
    if request.form["f10"] == "" or request.form["f11"] == "":
        return render_template(
            "result.html",
            prediction="Invalid Input",
            confidence=0
        )

    age = abs(float(request.form["f10"])) / 365
    income = float(request.form["f5"])
    employment = float(request.form["f6"])

    if age < 21:
        return render_template(
            "result.html",
            prediction="Rejected",
            confidence=100
        )

    if income <= 0:
        return render_template(
            "result.html",
            prediction="Rejected",
            confidence=100
        )

    if employment == 3 and income < 100000:
        return render_template(
            "result.html",
            prediction="Rejected",
            confidence=100
        )
        
    try:
        days_employed = float(request.form["f11"])

        if days_employed == 365243:
            days_employed = -1825

        values = [
            float(request.form["f1"]),
            float(request.form["f2"]),
            float(request.form["f3"]),
            float(request.form["f4"]),
            float(request.form["f5"]),
            float(request.form["f6"]),
            float(request.form["f7"]),
            float(request.form["f8"]),
            float(request.form["f9"]),
            float(request.form["f10"]),
            days_employed,
            float(request.form["f12"]),
            float(request.form["f13"]),
            float(request.form["f14"]),
            float(request.form["f15"]),
            float(request.form["f16"]),
            float(request.form["f17"])
        ]
    except (ValueError, KeyError):
        return render_template(
            "result.html",
            prediction="Invalid Input",
            confidence=0
        )

    # Create DataFrame
    input_df = pd.DataFrame([values], columns=feature_names)

    # Scale numerical features
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    
    # Prediction
    prediction = model.predict(input_df)[0]

    # Prediction probabilities
    probabilities = model.predict_proba(input_df)[0]

    # Confidence of predicted class
    confidence = round(probabilities[int(prediction)] * 100, 2)

    # Output text
    output = "Approved" if prediction == 1 else "Rejected"

    return render_template(
        "result.html",
        prediction=output,
        confidence=confidence
    )


# ================= RUN APPLICATION =================

if __name__ == "__main__":
    app.run(debug=True)