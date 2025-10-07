import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.title("Salary Prediction Application")
st.write("This app predicts employee salary based on *Years of Experience*.")

model_path = "model.pkl"
data_path = "Salary Data.csv"

# Load dataset first (used for training fallback and checking)
try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    st.error(f"Data file not found: {data_path}. Make sure the CSV is in the app folder.")
    df = None

# Try to load the trained model. If it fails, train a simple model from the CSV (if available).
model = None
try:
    with open(model_path, "rb") as file:
        model = pickle.load(file)
except Exception as e:
    st.warning(f"Could not load model from {model_path}: {e}")
    if df is not None:
        # Try a minimal training fallback so the app still works on deployment
        try:
            from sklearn.linear_model import LinearRegression
            X = df[["YearsExperience"]].values.reshape(-1, 1) if "YearsExperience" in df.columns else df.iloc[:, 0].values.reshape(-1, 1)
            y = df["Salary"].values if "Salary" in df.columns else df.iloc[:, 1].values
            lr = LinearRegression()
            lr.fit(X, y)
            model = lr
            # Save fallback model to disk so subsequent runs use it
            try:
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
                st.info(f"Trained a fallback model and saved to {model_path}.")
            except Exception:
                st.info("Trained a fallback model but couldn't save it to disk (permission or filesystem issue).")
        except Exception as train_e:
            st.error(f"Failed to train fallback model: {train_e}")
    else:
        st.error("No data available to train a fallback model.")

st.subheader("🔢 Enter Years of Experience")
years_exp = st.number_input("Years of Experience:", min_value=0.0, max_value=50.0, value=2.0, step=0.1)

if st.button("Predict Salary 💰"):
    prediction = model.predict(np.array([[years_exp]]))[0]
    st.success(f"*Predicted Salary:* Rs.{prediction}")

st.markdown("---")
st.markdown("""<div style='text-align: center; color: grey;'>👨‍💻 Developed by <b>Surendra Gupta</b></div>""",
    unsafe_allow_html=True
)