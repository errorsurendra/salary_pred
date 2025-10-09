import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.title("Salary Prediction Application")
st.write("This app predicts employee salary based on *Years of Experience*.")

model_path = "model.pkl"
data_path = "Salary Data.csv"

with open(model_path, "rb") as file:
    model = pickle.load(file)

df = pd.read_csv(data_path)

st.subheader("🔢 Enter Years of Experience")
years_exp = st.number_input("Years of Experience:", min_value=0.0, max_value=50.0, value=2.0, step=0.1)

if st.button("Predict Salary 💰"):
    prediction = model.predict(np.array([[years_exp]]))[0]
    st.success(f"*Predicted Salary:* Rs.{prediction}")

st.markdown("---")
st.markdown("""<div style='text-align: center; color: grey;'>👨‍💻 Developed by <b>Surendra Gupta</b></div>""",
    unsafe_allow_html=True
)