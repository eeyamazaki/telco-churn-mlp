"""Interface Streamlit para a API de previsão de churn."""

import io

import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📡",
    layout="wide",
)


# ── Autenticação ──────────────────────────────────────────────────────────────


def login(username: str, password: str) -> str | None:
    """Chama POST /login e retorna o token ou None se falhar."""
    try:
        resp = requests.post(
            f"{API_URL}/login",
            json={"username": username, "password": password},
            timeout=5,
        )
        if resp.status_code == 200:
            return resp.json()["access_token"]
        return None
    except requests.exceptions.ConnectionError:
        return None


def auth_headers() -> dict:
    return {"Authorization": f"Bearer {st.session_state.token}"}


# ── Tela de Login ─────────────────────────────────────────────────────────────


def pagina_login():
    st.title("📡 Telco Churn Prediction")
    st.subheader("Login")

    with st.form("login_form"):
        username = st.text_input("Usuário")
        password = st.text_input("Senha", type="password")
        submitted = st.form_submit_button("Entrar")

    if submitted:
        if not username or not password:
            st.error("Preencha usuário e senha.")
            return

        token = login(username, password)
        if token:
            st.session_state.token = token
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Credenciais inválidas ou API indisponível.")


# ── Predição Single ───────────────────────────────────────────────────────────


def pagina_predicao_single():
    st.header("Predição Individual")
    st.markdown("Preencha os dados do cliente para obter a previsão de churn.")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Dados Pessoais")
            gender = st.selectbox("Gênero", ["Female", "Male"])
            senior_citizen = st.selectbox("Sênior", ["No", "Yes"])
            partner = st.selectbox("Tem parceiro", ["No", "Yes"])
            dependents = st.selectbox("Tem dependentes", ["No", "Yes"])

        with col2:
            st.subheader("Conta")
            tenure_months = st.number_input("Meses como cliente", min_value=1, max_value=72, value=12)
            monthly_charges = st.number_input("Cobrança mensal (USD)", min_value=0.1, max_value=119.0, value=65.0)
            total_charges = st.number_input("Cobrança total (USD)", min_value=0.0, max_value=8690.0, value=780.0)
            contract = st.selectbox("Contrato", ["Month-to-month", "One year", "Two year"])
            payment_method = st.selectbox("Forma de pagamento", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)",
            ])
            paperless_billing = st.selectbox("Fatura eletrônica", ["No", "Yes"])

        with col3:
            st.subheader("Serviços")
            phone_service = st.selectbox("Serviço de telefone", ["Yes", "No"])
            multiple_lines = st.selectbox("Múltiplas linhas", ["No", "Yes", "No phone service"])
            internet_service_type = st.selectbox("Internet", ["Fiber optic", "DSL", "No"])
            internet_options = ["No internet service", "No", "Yes"]
            online_security = st.selectbox("Segurança online", internet_options)
            online_backup = st.selectbox("Backup online", internet_options)
            device_protection = st.selectbox("Proteção de dispositivo", internet_options)
            tech_support = st.selectbox("Suporte técnico", internet_options)
            streaming_tv = st.selectbox("Streaming TV", internet_options)
            streaming_movies = st.selectbox("Streaming filmes", internet_options)

        submitted = st.form_submit_button("Prever Churn", use_container_width=True)

    if submitted:
        payload = {
            "gender": gender,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "phone_service": phone_service,
            "multiple_lines": multiple_lines,
            "paperless_billing": paperless_billing,
            "payment_method": payment_method,
            "contract": contract,
            "internet_service_type": internet_service_type,
            "online_security": online_security,
            "online_backup": online_backup,
            "device_protection": device_protection,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
        }

        with st.spinner("Calculando..."):
            resp = requests.post(
                f"{API_URL}/predict",
                json=payload,
                headers=auth_headers(),
                timeout=10,
            )

        if resp.status_code == 200:
            pred = resp.json()["prediction"]
            is_churn = pred["prediction"] == "Churn"

            st.divider()
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.metric("Decisão", pred["prediction"])
            with col_b:
                st.metric("Probabilidade", f"{pred['churn_probability']:.1%}")
            with col_c:
                st.metric("Threshold", f"{pred['threshold_used']:.2f}")

            if is_churn:
                st.error("⚠️ Cliente com alto risco de churn — acionar campanha de retenção.")
            else:
                st.success("✅ Cliente com baixo risco de churn.")

        elif resp.status_code == 422:
            st.error(f"Dados inválidos: {resp.json().get('detail', resp.text)}")
        elif resp.status_code == 401:
            st.warning("Sessão expirada. Faça login novamente.")
            st.session_state.clear()
            st.rerun()
        else:
            st.error(f"Erro na API: {resp.status_code}")


# ── Predição Batch ────────────────────────────────────────────────────────────


def pagina_predicao_batch():
    st.header("Predição em Lote")
    st.markdown("Envie um arquivo **CSV ou XLSX** com os dados dos clientes.")

    uploaded = st.file_uploader("Selecione o arquivo", type=["csv", "xlsx"])

    if uploaded is not None:
        st.info(f"Arquivo carregado: **{uploaded.name}** ({uploaded.size / 1024:.1f} KB)")

        if st.button("Enviar para predição", use_container_width=True):
            with st.spinner("Processando..."):
                resp = requests.post(
                    f"{API_URL}/predict/batch",
                    headers=auth_headers(),
                    files={"file": (uploaded.name, uploaded.getvalue(), "application/octet-stream")},
                    timeout=60,
                )

            if resp.status_code == 200:
                df_result = io.StringIO(resp.text)
                import pandas as pd
                df = pd.read_csv(df_result)

                st.success(f"Predição concluída — {len(df)} clientes processados.")

                n_churn = int(df["churn_prediction"].sum())
                taxa = n_churn / len(df)
                col1, col2, col3 = st.columns(3)
                col1.metric("Total de clientes", len(df))
                col2.metric("Previsão de churn", n_churn)
                col3.metric("Taxa de churn", f"{taxa:.1%}")

                st.dataframe(
                    df[["churn_probability", "churn_prediction"]].describe().round(3),
                    use_container_width=True,
                )

                st.download_button(
                    label="Baixar CSV com predições",
                    data=resp.content,
                    file_name="predicoes_churn.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            elif resp.status_code == 401:
                st.warning("Sessão expirada. Faça login novamente.")
                st.session_state.clear()
                st.rerun()
            else:
                st.error(f"Erro na API: {resp.status_code} — {resp.text[:300]}")


# ── App principal ─────────────────────────────────────────────────────────────


def main():
    if "token" not in st.session_state:
        pagina_login()
        return

    with st.sidebar:
        st.title("📡 Churn Prediction")
        st.markdown(f"Logado como **{st.session_state.get('username', '')}**")
        st.divider()
        pagina = st.radio("Navegação", ["Predição Individual", "Predição em Lote"])
        st.divider()
        if st.button("Sair", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    if pagina == "Predição Individual":
        pagina_predicao_single()
    else:
        pagina_predicao_batch()


if __name__ == "__main__":
    main()
