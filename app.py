import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from collections import Counter

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Heart Disease Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.title("Heart Disease Predictor")

# ======================================================
# LOAD MODELS
# ======================================================
@st.cache_resource
def load_models():
    return {
        "Logistic Regression": pickle.load(open("LogisticRegression.pkl", "rb")),
        "Random Forest": pickle.load(open("RandomForest.pkl", "rb")),
        "Decision Tree": pickle.load(open("DecisionTree.pkl", "rb")),
        "Support Vector Machine": pickle.load(open("SVM.pkl", "rb"))
    }

models = load_models()

# ======================================================
# LOAD MODEL ACCURACIES (PRE-COMPUTED)
# ======================================================
# Example values – replace with your actual test accuracies if needed
model_accuracies = {
    "Logistic Regression": 0.85,
    "Random Forest": 0.88,
    "Decision Tree": 0.82,
    "Support Vector Machine": 0.86
}

# ======================================================
# TABS
# ======================================================
tab1, tab2, tab3 = st.tabs(["Predict", "Bulk Predict", "Model Information"])

# ======================================================
# TAB 1: SINGLE PREDICTION (ALL MODELS + FINAL DECISION)
# ======================================================
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", 1, 120)
        sex = st.selectbox("Sex", ["Male", "Female"])
        chest_pain = st.selectbox(
            "Chest Pain Type",
            ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
        )
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 0, 300)
        cholesterol = st.number_input("Serum Cholesterol (mg/dl)", 0, 600)
        fasting_bs = st.selectbox("Fasting Blood Sugar", ["≤ 120 mg/dl", "> 120 mg/dl"])

    with col2:
        resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST", "LVH"])
        max_hr = st.number_input("Maximum Heart Rate Achieved", 60, 202)
        exercise_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
        oldpeak = st.number_input("Oldpeak (ST Depression)", 0.0, 6.0)
        st_slope = st.selectbox(
            "Slope of Peak Exercise ST Segment",
            ["Upsloping", "Flat", "Downsloping"]
        )

    # -----------------------------
    # ENCODING (heart.csv format)
    # -----------------------------
    sex = 1 if sex == "Male" else 0
    chest_pain = {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-Anginal Pain": 2,
        "Asymptomatic": 3
    }[chest_pain]
    fasting_bs = 1 if fasting_bs == "> 120 mg/dl" else 0
    resting_ecg = {"Normal": 0, "ST": 1, "LVH": 2}[resting_ecg]
    exercise_angina = 1 if exercise_angina == "Yes" else 0
    st_slope = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}[st_slope]

    input_data = np.array([[age, sex, chest_pain, resting_bp, cholesterol,
                            fasting_bs, resting_ecg, max_hr,
                            exercise_angina, oldpeak, st_slope]])

    if st.button("Submit"):
        st.markdown("---")
        predictions = []

        # Individual model predictions
        for model_name, model in models.items():
            pred = model.predict(input_data)[0]
            predictions.append(pred)

            st.subheader(model_name)
            if pred == 1:
                st.error("Heart disease detected.")
            else:
                st.success("No heart disease detected.")

        # -----------------------------
        # FINAL DECISION (MAJORITY VOTE)
        # -----------------------------
        final_decision = Counter(predictions).most_common(1)[0][0]

        st.markdown("---")
        st.subheader("Final Decision (Majority Voting)")
        if final_decision == 1:
            st.error("⚠️ Final Result: Heart Disease Detected")
        else:
            st.success("✅ Final Result: No Heart Disease Detected")

# ======================================================
# TAB 2: BULK PREDICTION (UNCHANGED)
# ======================================================
with tab2:
    st.subheader("Bulk Prediction Instructions")

    st.markdown("""
**Important Rules:**
- ❌ No missing (NaN) values  
- ✅ Exactly **11 features only**
- ✅ Columns must be in the exact order used during training
""")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview", df.head())

        # Majority voting for bulk
        all_preds = []

        for model in models.values():
            all_preds.append(model.predict(df))

        final_preds = np.round(np.mean(all_preds, axis=0)).astype(int)
        df["HeartDisease_Prediction"] = final_preds

        st.success("Prediction Completed")
        st.dataframe(df)

        st.download_button(
            "Download Result CSV",
            df.to_csv(index=False),
            "PredictedHeart.csv",
            "text/csv"
        )

# ======================================================
# TAB 3: MODEL INFORMATION + ACCURACY BAR GRAPH
# ======================================================
with tab3:
    st.subheader("Model Accuracy Comparison")

    acc_df = pd.DataFrame({
        "Model": model_accuracies.keys(),
        "Accuracy (%)": [v * 100 for v in model_accuracies.values()]
    }).set_index("Model")

    st.bar_chart(acc_df)

    st.markdown("""
### Final Decision Logic
- Each model predicts independently
- Final output is based on **majority voting**
- This improves robustness compared to a single model

### Models Used
- Logistic Regression  
- Random Forest  
- Decision Tree  
- Support Vector Machine  

### Target Variable
- HeartDisease  
  - 0 = No  
  - 1 = Yes  
""")
