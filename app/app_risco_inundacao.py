import streamlit as st
import pandas as pd
import joblib
from datetime import date

# Mapeamento numérico -> rótulo de texto
RISK_LABELS = {
    0: "Sem risco",
    1: "Risco de Inundação Transitável",
    2: "Risco de Inundação Intransitável",
}


def classificar_risco_chuva(mm: float) -> int:
    """
    Define a classe de risco com base na chuva do dia (mm).

    0 -> Sem risco            (chuva < 20 mm)
    1 -> Risco Transitável    (20 mm <= chuva < 60 mm)
    2 -> Risco Intransitável  (chuva >= 60 mm)
    """
    if mm < 20.0:
        return 0
    elif mm < 60.0:
        return 1
    else:
        return 2


@st.cache_resource
def load_model():
    """Carrega o modelo Random Forest treinado (3 classes)."""
    pacote = joblib.load("models/modelo_risco_inundacao_3classes.joblib")
    model = pacote["model"]
    features = pacote["features"]
    labels = pacote.get("labels", RISK_LABELS)  # fallback caso não exista
    return model, features, labels


def main():
    st.title("Protótipo – IA de Risco de Inundação no Butantã (3 classes)")
    st.write(
        "Este protótipo usa um modelo de classificação treinado com chuvas diárias "
        "do CGE Butantã para estimar o risco de inundação em **3 níveis de severidade**, "
        "a partir da intensidade de chuva e do histórico recente."
    )

    model, features, labels = load_model()

    st.subheader("Entrada manual de dados de chuva")

    data_ref = st.date_input("Data de referência", value=date.today())

    col1, col2 = st.columns(2)
    with col1:
        chuva_hoje = st.number_input("Chuva HOJE (mm)", min_value=0.0, step=0.1, value=0.0)
        chuva_lag1 = st.number_input("Chuva ONTEM (mm)", min_value=0.0, step=0.1, value=0.0)
    with col2:
        chuva_lag2 = st.number_input("Chuva 2 dias atrás (mm)", min_value=0.0, step=0.1, value=0.0)
        chuva_lag3 = st.number_input("Chuva 3 dias atrás (mm)", min_value=0.0, step=0.1, value=0.0)

    if st.button("Calcular risco"):
        mes = data_ref.month
        dia_semana = data_ref.weekday()  # 0 = segunda

        # Monta o dataframe com as MESMAS features usadas no treino
        X_novo = pd.DataFrame([{
            "chuva_mm": chuva_hoje,
            "chuva_lag1": chuva_lag1,
            "chuva_lag2": chuva_lag2,
            "chuva_lag3": chuva_lag3,
            "mes": mes,
            "dia_semana": dia_semana,
        }])

        # Garante a ordem correta das colunas
        X_novo = X_novo[features]

        # Predição do modelo
        probs = model.predict_proba(X_novo)[0]
        classe_pred = int(model.predict(X_novo)[0])

        # Classe teórica apenas pelas faixas de chuva de hoje
        classe_regra = classificar_risco_chuva(chuva_hoje)

        st.subheader("Resultado")

        st.markdown(
            f"**Classe prevista pelo modelo (Random Forest):** "
            f"{RISK_LABELS.get(classe_pred, 'Desconhecido')}"
        )

        st.markdown(
            f"**Classe teórica pelas faixas de chuva (apenas chuva de hoje = {chuva_hoje:.1f} mm):** "
            f"{RISK_LABELS.get(classe_regra, 'Desconhecido')}"
        )

        st.markdown("**Probabilidade por classe (modelo):**")
        for cls, p in zip(model.classes_, probs):
            st.write(f"- {RISK_LABELS[int(cls)]}: {p:.1%}")

        st.info(
            "As classes de risco atuais são definidas por faixas de chuva diária: "
            "**Sem risco** (< 20 mm), "
            "**Risco de Inundação Transitável** (20–60 mm) e "
            "**Risco de Inundação Intransitável** (≥ 60 mm). "
            "O modelo aprende padrões usando a chuva do dia, o histórico recente "
            "(lags) e variáveis de calendário. "
            "No futuro, esses limites poderão ser refinados com base em dados "
            "hidrológicos físicos (escoamento e capacidade do sistema de drenagem)."
        )


if __name__ == "__main__":
    main()
