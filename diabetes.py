import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import  sklearn
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder,RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.neighbors import KNeighborsRegressor
from PIL import Image
from sklearn.metrics import accuracy_score

# Load model
with open("elite_31_Diabetes.pkl", "rb") as f:
    final_model = pickle.load(f)

# App Title
st.markdown(
    "<h1 style='text-align: center; color: crimson;'>Pima Indians Diabetes Prediction🩸</h1>", 
    unsafe_allow_html=True
)

st.image(
    "https://cdn.prod.website-files.com/63c5ae549b033b01227743c2/64be15e1ff307d68abc6bc2f_TYPE%202%20DIABETES.jpg", 
    use_column_width=True
)

# Input layout
with st.form("diabetes_form"):
    st.subheader("📝 Patient Information")
    col1, col2 = st.columns(2)

    with col1:
        Pregnancies = st.number_input("👶 Number of Pregnancies", 0, 17)
        Glucose = st.number_input("🧪 Glucose Level", 0, 199)
        BloodPressure = st.number_input("💓 Blood Pressure", 0, 122)
        SkinThickness = st.number_input("📏 Skin Thickness", 0, 99)

    with col2:
        Insulin = st.number_input("💉 Insulin Level", 0, 846)
        BMI = st.number_input("📊 BMI", 0.0, 67.1)
        DiabetesPedigreeFunction = st.number_input("👨‍👩‍👧‍👦 Diabetes Pedigree Function", 0.078, 2.42)
        Age = st.number_input("🎂 Age", 21, 81)

    submitted = st.form_submit_button("Drop Blood 💉🩸")

# Animation + Prediction
if submitted:
    drop_area = st.empty()
    for i in range(15):
        drop_area.markdown("🩸" * (i + 1))
        time.sleep(0.05)

    # Prediction
    pred = final_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                                 Insulin, BMI, DiabetesPedigreeFunction, Age]])

    st.success("✅ Prediction Completed!")

    if pred[0] == 1:
        st.markdown(
            "<h3 style='color:red;'>🔴 The patient is likely to have Diabetes.</h3>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<h3 style='color:green;'>🟢 The patient is unlikely to have Diabetes.</h3>",
            unsafe_allow_html=True
        )

    # Show summary
    st.markdown("---")
    st.subheader("📋 Prediction Summary")
    st.write(pd.DataFrame({
        "Feature": [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ],
        "Value": [
            Pregnancies, Glucose, BloodPressure, SkinThickness,
            Insulin, BMI, DiabetesPedigreeFunction, Age
        ]
    }))

    # Health Tip
    st.markdown("---")
    st.info("💡 *Tip: Regular exercise, a balanced diet, and routine health checks help manage or prevent diabetes.*")

 
 