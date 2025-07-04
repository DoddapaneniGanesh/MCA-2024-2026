import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

st.set_page_config(layout="centered")
st.title("📈 Predict Future Stock Close Price (Manual Date & Features)")

# File upload
uploaded_file = st.file_uploader("Upload Excel File (with 'Close' column)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.tail())

    df = df.dropna()

    if "Close" not in df.columns:
        st.error("❌ 'Close' column is required in the dataset.")
    else:
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != "Close"]

        st.subheader("🔧 Select Features for Prediction")
        features = st.multiselect("Select at least 2 numeric features", numeric_cols, default=numeric_cols[:2])

        if len(features) >= 2:
            # Prepare data
            X = df[features]
            y = df["Close"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.subheader("📊 Model Performance")
            st.write(f"**R² Score:** {r2_score(y_test, y_pred):.4f}")
            st.write(f"**Mean Squared Error:** {mean_squared_error(y_test, y_pred):.4f}")

            st.subheader("📅 Predict Close Price for a Future Date")

            future_date = st.text_input("Enter a future date (e.g. 2022-04-11):")

            st.markdown("### ✏️ Enter Feature Values for That Date")
            future_data = []
            for feature in features:
                value = st.number_input(f"{feature} value:", value=float(df[feature].iloc[-1]))
                future_data.append(value)

            if st.button("Predict Close Price"):
                input_df = pd.DataFrame([future_data], columns=features)
                prediction = model.predict(input_df)[0]
                st.success(f"📌 Predicted Close Price for {future_date}: **{prediction:.2f}**")

                st.write("🔍 Prediction Detail")
                st.write(pd.DataFrame({
                    "Date": [future_date],
                    "Predicted Close Price": [prediction]
                }))
        else:
            st.warning("⚠️ Please select at least **two** numeric features.")
else:
    st.info("Upload an Excel file to begin.")
