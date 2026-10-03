import streamlit as st
import pandas as pd
import joblib


# -----------------------------
# Load Model
# -----------------------------

model = joblib.load("retail_sales_rf_model.pkl")
model_columns = joblib.load("rf_model_columns.pkl")


# -----------------------------
# Page Settings
# -----------------------------

st.set_page_config(
    page_title="Retail Sales Prediction",
    page_icon="🛒",
    layout="wide"
)


# -----------------------------
# Title
# -----------------------------

st.title("🛒 Retail Sales Prediction Dashboard")

st.write(
    "Enter the customer and product details to predict the total sales amount."
)


# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.header("Customer & Product Details")


age = st.sidebar.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=30
)


gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)


product_category = st.sidebar.selectbox(
    "Product Category",
    ["Beauty", "Clothing", "Electronics"]
)


quantity = st.sidebar.number_input(
    "Quantity",
    min_value=1,
    max_value=100,
    value=1
)


price_per_unit = st.sidebar.number_input(
    "Price Per Unit",
    min_value=0.0,
    value=100.0
)


# -----------------------------
# Prediction Button
# -----------------------------

predict_button = st.sidebar.button(
    "🔮 Predict Total Amount"
)


# -----------------------------
# Main Dashboard
# -----------------------------

if predict_button:

    # Create input data
    input_data = pd.DataFrame({

        "Age": [age],

        "Quantity": [quantity],

        "Price per Unit": [price_per_unit],

        "Gender": [gender],

        "Product Category": [product_category]

    })


    # Convert categorical columns
    input_data = pd.get_dummies(
        input_data,
        columns=["Gender", "Product Category"],
        drop_first=True
    )


    # Match training columns
    input_data = input_data.reindex(
        columns=model_columns,
        fill_value=0
    )


    # -----------------------------
    # Prediction
    # -----------------------------

    prediction = model.predict(input_data)

    predicted_amount = prediction[0]


    # -----------------------------
    # KPI Cards
    # -----------------------------

    st.subheader("📊 Prediction Summary")

    col1, col2, col3, col4 = st.columns(4)


    col1.metric(
        "Predicted Amount",
        f"₹ {predicted_amount:.2f}"
    )


    col2.metric(
        "Quantity",
        quantity
    )


    col3.metric(
        "Price Per Unit",
        f"₹ {price_per_unit:.2f}"
    )


    col4.metric(
        "Customer Age",
        age
    )


    # -----------------------------
    # Prediction Result
    # -----------------------------

    st.divider()

    st.subheader("🎯 Prediction Result")

    st.success(
        f"Predicted Total Amount: ₹ {predicted_amount:.2f}"
    )


    # -----------------------------
    # Order Information
    # -----------------------------

    st.subheader("🛍️ Order Information")

    col1, col2 = st.columns(2)


    with col1:

        st.write("**Customer Details**")

        st.write(f"Age: {age}")

        st.write(f"Gender: {gender}")


    with col2:

        st.write("**Product Details**")

        st.write(f"Category: {product_category}")

        st.write(f"Quantity: {quantity}")

        st.write(f"Price Per Unit: ₹ {price_per_unit:.2f}")


else:

    st.info(
        "👈 Enter the details in the sidebar and click "
        "**Predict Total Amount**."
    )

