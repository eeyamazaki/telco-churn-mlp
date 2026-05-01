import streamlit as st
import requests
import pandas as pd
from io import BytesIO

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Telco Churn Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Telco Churn Prediction")

# ─────────────────────────────────────────────────────────────
# LOGIN
# ─────────────────────────────────────────────────────────────

if "token" not in st.session_state:
    st.session_state.token = None

with st.sidebar:

    st.header("🔐 Login")

    username = st.text_input("Usuário")
    password = st.text_input("Senha", type="password")

    if st.button("Entrar"):

        response = requests.post(
            f"{API_URL}/login",
            json={
                "username": username,
                "password": password
            }
        )

        if response.status_code == 200:

            data = response.json()

            st.session_state.token = data["access_token"]

            st.success("Login realizado!")

        else:
            st.error("Credenciais inválidas")

headers = {}

if st.session_state.token:
    headers = {
        "Authorization": f"Bearer {st.session_state.token}"
    }

# ─────────────────────────────────────────────────────────────
# STATUS API
# ─────────────────────────────────────────────────────────────

st.subheader("🩺 API Status")

try:

    response = requests.get(f"{API_URL}/health")

    if response.status_code == 200:
        st.success("API Online")
        st.json(response.json())

    else:
        st.error("API Offline")

except Exception as e:
    st.error(str(e))

# ─────────────────────────────────────────────────────────────
# PREVISÃO INDIVIDUAL
# ─────────────────────────────────────────────────────────────

st.subheader("🔮 Previsão Individual")

with st.form("predict_form"):

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            ["Yes", "No"]
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        internet_service_type = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )

    with col2:

        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        tenure_months = st.number_input(
            "Tenure Months",
            min_value=0,
            value=24
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=65.5
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=1570.0
        )

    submitted = st.form_submit_button("Prever Churn")

    if submitted:

        payload = {
            "contract": contract,
            "dependents": dependents,
            "device_protection": device_protection,
            "gender": gender,
            "internet_service_type": internet_service_type,
            "monthly_charges": monthly_charges,
            "multiple_lines": multiple_lines,
            "online_backup": online_backup,
            "online_security": online_security,
            "paperless_billing": paperless_billing,
            "partner": partner,
            "payment_method": payment_method,
            "phone_service": phone_service,
            "senior_citizen": senior_citizen,
            "streaming_movies": streaming_movies,
            "streaming_tv": streaming_tv,
            "tech_support": tech_support,
            "tenure_months": tenure_months,
            "total_charges": total_charges
        }

        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            headers=headers
        )

        if response.status_code == 200:

            result = response.json()

            prediction = result["prediction"]

            st.success("Predição realizada!")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Prediction",
                    prediction["prediction"]
                )

            with col2:
                st.metric(
                    "Churn Probability",
                    f'{prediction["churn_probability"]:.2%}'
                )

            with col3:
                st.metric(
                    "Confidence",
                    f'{prediction["confidence"]:.2%}'
                )

            st.json(result)

        else:
            st.error(response.text)

# ─────────────────────────────────────────────────────────────
# BATCH PREDICTION
# ─────────────────────────────────────────────────────────────

st.subheader("📂 Batch Prediction")

uploaded_file = st.file_uploader(
    "Upload CSV ou XLSX",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    if st.button("Processar Arquivo"):

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }

        response = requests.post(
            f"{API_URL}/predict/batch",
            headers=headers,
            files=files
        )

        if response.status_code == 200:

            st.success("Arquivo processado!")

            st.download_button(
                label="⬇️ Download Predictions",
                data=response.content,
                file_name="predictions.csv",
                mime="text/csv"
            )

            df = pd.read_csv(BytesIO(response.content))

            st.dataframe(df.head())

        else:
            st.error(response.text)