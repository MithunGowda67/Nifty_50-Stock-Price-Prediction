import pandas as pd
import numpy as np
import streamlit as st
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

# Load NIFTY 50 stock data
@st.cache_data
def load_data():
    try:
        return pd.read_csv("nifty50_stocks_data.csv")
    except FileNotFoundError:
        st.error("❌ Data file not found! Make sure 'nifty50_stocks_data.csv' is in the correct directory.")
        return pd.DataFrame()

# Streamlit App Config
st.set_page_config(page_title="NIFTY 50 Stock Price Prediction", layout="wide")
st.title("📈 NIFTY 50 Stock Price Prediction")

# Load stock data
df = load_data()

# If data failed to load, stop execution
if df.empty:
    st.stop()

# Sidebar for Stock Selection
st.sidebar.header("Select Stock")
stock_symbol = st.sidebar.selectbox("Choose a stock", df["Symbol"].unique())
forecast_days = st.sidebar.number_input("Forecast Days", min_value=1, max_value=30, step=1, value=7)

# Display Stock Info
stock_data = df[df["Symbol"] == stock_symbol]
st.subheader(f"Stock Details: {stock_symbol}")
st.dataframe(stock_data)

# Train the SVR Model
def train_model(data, days):
    X = np.array(data["Last Sale"]).reshape(-1, 1)
    y = np.array(data["Last Sale"].shift(-days))

    # Ensure enough data before splitting
    if len(X) <= days or len(y) <= days:
        st.warning("⚠️ Not enough historical data to predict. Try a lower forecast period.")
        return None

    X, y = X[:-days], y[:-days]

    if len(X) < 2:  # Ensure at least 2 samples
        st.warning("⚠️ Dataset is too small for training.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = SVR(kernel="rbf", C=1000.0, gamma=0.0001)
    model.fit(X_train, y_train)

    return model.predict(X[-days:].reshape(-1, 1))

# Predict and Display Results
st.subheader("Predicted Stock Prices")
predicted_prices = train_model(stock_data, forecast_days)

# Only display results if predictions exist
if predicted_prices is not None:
    for i, price in enumerate(predicted_prices):
        st.write(f"📅 Day {i+1}: ₹{round(price, 2)}")
else:
    st.info("🔹 Try selecting a different stock or reducing forecast days.")

st.info("🔹 This is a basic model for educational purposes. Not for financial decisions.")
